"""从新浪财经抓取增量行情，补进 qlib 的 cn_data。

qlib 官方数据集更新滞后（本机上落后了 11 个交易日），而 10 日调仓的策略每次调仓前
都需要当日行情。这个脚本只补最后一段缺口，不重建历史。

    python production/update_data.py --check          # 只看差多少天
    python production/update_data.py                  # 抓取并写入
    python production/update_data.py --days 30        # 多抓一些以防漏

关于价格口径：新浪返回的是**不复权价**，而 qlib 的 $close 是复权价，两者关系是
$close = 真实价 x $factor。分红送股当日 factor 会跳变，脚本无法从新浪的日线接口得知，
所以沿用每只股票最后一个已知的 factor。这在两次除权之间是准确的；跨越除权日的那几天
会有偏差，因此 --check 会提示距上次全量更新的天数，超过一个季度应该用 qlib 官方数据
重新拉一次全量。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
CALENDAR = QLIB_DIR / "calendars/day.txt"
FEATURES = QLIB_DIR / "features"
SINA = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen={n}"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}
FIELDS = ["open", "close", "high", "low", "volume", "factor"]


def qlib_calendar() -> list[pd.Timestamp]:
    return [pd.Timestamp(x) for x in CALENDAR.read_text().split()]


def universe(market: str = "csi300") -> list[str]:
    """当前仍在册的成分股。只更新在交易的池子，不必碰全市场。"""
    path = QLIB_DIR / f"instruments/{market}.txt"
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    latest = max(r[2] for r in rows)
    return sorted({r[0] for r in rows if r[2] >= latest})


def fetch(code: str, days: int) -> pd.DataFrame:
    """抓单只股票的日线。code 形如 SH600000，新浪要 sh600000。"""
    resp = requests.get(SINA.format(sym=code.lower(), n=days), headers=HEADERS, timeout=20)
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or text[0] != "[":
        raise ValueError(f"返回不是 JSON 数组: {text[:80]}")
    rows = json.loads(text)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["day"])
    for col in ("open", "close", "high", "low", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 新浪的 volume 按股，qlib 按手
    df["volume"] = df["volume"] / 100.0
    return df.set_index("date")[["open", "close", "high", "low", "volume"]].dropna()


def read_bin(path: Path) -> tuple[int, np.ndarray]:
    """qlib .bin 格式：首个 float32 是起始日在日历中的下标，其后是数值。"""
    raw = path.read_bytes()
    start = int(struct.unpack("<f", raw[:4])[0])
    return start, np.frombuffer(raw[4:], dtype="<f4")


def write_bin(path: Path, start: int, values: np.ndarray) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("<f", float(start)))
        fh.write(values.astype("<f4").tobytes())


def update_symbol(code: str, cal: list[pd.Timestamp], new_dates: list[pd.Timestamp], days: int) -> str:
    """把 new_dates 这几天的数据追加到该股票的各个 .bin。返回状态说明。"""
    folder = FEATURES / code.lower()
    if not folder.exists():
        return "无历史数据"

    close_bin = folder / "close.day.bin"
    if not close_bin.exists():
        return "缺 close.day.bin"
    start, close_vals = read_bin(close_bin)
    have_until = start + len(close_vals) - 1
    if have_until >= cal.index(new_dates[-1]):
        return "已是最新"

    quotes = fetch(code, days)
    if quotes.empty:
        return "新浪无数据"

    # factor 沿用最后一个已知值：日线接口看不到除权信息
    factor_bin = folder / "factor.day.bin"
    factor = 1.0
    if factor_bin.exists():
        _, fvals = read_bin(factor_bin)
        if len(fvals):
            factor = float(fvals[-1])

    appended = 0
    for field in FIELDS:
        bin_path = folder / f"{field}.day.bin"
        if not bin_path.exists():
            continue
        f_start, vals = read_bin(bin_path)
        vals = list(vals)
        cursor = f_start + len(vals) - 1
        for date in new_dates:
            idx = cal.index(date)
            if idx <= cursor:
                continue
            # 日历上有、但该股当日无行情（停牌）时补 nan，保持下标对齐
            while cursor + 1 < idx:
                vals.append(np.nan)
                cursor += 1
            if date in quotes.index:
                row = quotes.loc[date]
                if field == "factor":
                    vals.append(factor)
                elif field == "volume":
                    vals.append(float(row["volume"]))
                else:
                    # qlib 存复权价
                    vals.append(float(row[field]) * factor)
            else:
                vals.append(np.nan)
            cursor += 1
            if field == "close":
                appended += 1
        write_bin(bin_path, f_start, np.array(vals, dtype="f4"))
    return f"补 {appended} 日"


def extend_membership(market: str, until: pd.Timestamp) -> None:
    """把当前在册成分的结束日顺延到 until。

    instruments/*.txt 是时点成分表（code / 起 / 止）。只补行情不补它，D.instruments()
    在新日期上会返回空集，因子源数据就停在旧日期上，症状是"行情更新了但信号没动"。
    这里只顺延已在册的成分，不猜测指数调整——真正的成分变更要等官方数据。
    """
    path = QLIB_DIR / f"instruments/{market}.txt"
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    latest = max(r[2] for r in rows)
    stamp = until.strftime("%Y-%m-%d")
    if latest >= stamp:
        return
    out = [[r[0], r[1], stamp if r[2] == latest else r[2]] for r in rows]
    path.write_text("\n".join("\t".join(r) for r in out) + "\n")
    n = sum(1 for r in rows if r[2] == latest)
    print(f"成分表 {market}: {n} 只在册成分顺延至 {stamp}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告缺口，不写入")
    ap.add_argument("--days", type=int, default=30, help="向新浪索取的天数")
    ap.add_argument("--market", default="csi300")
    ap.add_argument("--sleep", type=float, default=0.15, help="每只之间的间隔，别把对方打疼")
    args = ap.parse_args()

    cal = qlib_calendar()
    print(f"qlib 日历最新 {cal[-1].date()}，共 {len(cal)} 个交易日")

    # 用一只活跃股探测最新交易日
    probe = fetch("SH600000", args.days)
    if probe.empty:
        print("新浪无返回，检查网络", file=sys.stderr)
        return 1
    market_last = probe.index.max()
    print(f"新浪最新交易日 {market_last.date()}")

    missing = [d for d in probe.index if d > cal[-1]]
    if not missing:
        print("数据已是最新，无需更新")
        return 0
    print(f"缺 {len(missing)} 个交易日: {missing[0].date()} ~ {missing[-1].date()}")
    if args.check:
        return 0

    # 日历必须先补，各 .bin 的下标都相对它
    cal = cal + missing
    CALENDAR.write_text("\n".join(d.strftime("%Y-%m-%d") for d in cal) + "\n")
    print(f"日历已补至 {cal[-1].date()}")

    # 成分表也要顺延，否则 D.instruments() 在新日期上返回空 —— 行情补了但因子算不出来
    extend_membership(args.market, cal[-1])

    codes = universe(args.market)
    print(f"更新 {len(codes)} 只 {args.market} 成分股…")
    stats: dict[str, int] = {}
    for i, code in enumerate(codes, 1):
        try:
            note = update_symbol(code, cal, missing, args.days)
        except Exception as exc:  # 单只失败不该中断整批
            note = f"失败: {type(exc).__name__}"
        stats[note] = stats.get(note, 0) + 1
        if i % 50 == 0:
            print(f"  {i}/{len(codes)}")
        time.sleep(args.sleep)

    print("\n结果:")
    for note, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {note}: {n} 只")
    print("\n更新后请重跑 generate_data_folder_from_qlib() 刷新因子源数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

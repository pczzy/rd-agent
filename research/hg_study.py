"""WRB + HG 首次回踩研究：只回答"HG 第一次被回踩，是否有统计优势"。

    python research/hg_study.py SH603986            # 60m（新浪）+ 1D（qlib）
    python research/hg_study.py SH603986 --refresh  # 重新抓 60m
    python research/hg_study.py --pool csi300       # 当前沪深300成分股合并统计

定义（第一轮固定，不优化）：
  WRB   body > max(body[前5根]) 且 body > 1.2 x ATR14
  HG    多：K[-1] 是阳 WRB 且 low[0] > high[-2]，区间 [high[-2], low[0]]
        空：K[-1] 是阴 WRB 且 high[0] < low[-2]，区间 [high[0], low[-2]]
        K[0] 收盘才确认，此前不可交易
  首测  多：之后第一根 low <= HG_high 的 K 线；只看这一根，之后该 HG 作废
  信号  多：首测 K 收盘 > HG_mid 且收阳；空对称
  成交  信号后下一根开盘；一字涨停开盘买不进则跳过
  止损  A: HG_low - 0.1ATR   B: WRB low（空头对称）
  止盈  k x R，R = entry - stop，R < 0.2ATR 的跳过（开盘贴着止损）；最长持有 20 根，到期按收盘平
  A 股  多头 T+1：买入当天不能卖，次日起按开盘跳空/盘中触价处理；
        同一根既碰止损又碰止盈按先止损算（保守）

60m 是新浪不复权价，窗口内若有除权会制造假缺口；1D 用 qlib 复权价。
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data"
QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
SINA = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "CN_MarketData.getKLineData?symbol={sym}&scale={scale}&ma=no&datalen={n}")

LOOKBACK, WRB_ATR, MAX_HOLD = 5, 1.2, 20
MIN_RISK_ATR = 0.2
COST = 0.0015  # 往返：佣金 2x0.025% + 印花税 0.05% + 滑点约 0.05%


# --------------------------------------------------------------------------- 数据


def fetch_sina(code: str, scale: int, n: int = 5000) -> pd.DataFrame:
    """不走代理：trust_env=False 忽略环境里的 http(s)_proxy。"""
    s = requests.Session()
    s.trust_env = False
    r = s.get(SINA.format(sym=code.lower(), scale=scale, n=n),
              headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(json.loads(r.text))
    df["dt"] = pd.to_datetime(df["day"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    return df.set_index("dt")[["open", "high", "low", "close", "volume"]]


def load_60m(code: str, refresh: bool) -> pd.DataFrame:
    path = CACHE / f"{code.lower()}_60m.csv"
    if refresh or not path.exists():
        new = fetch_sina(code, 60)
        if path.exists():  # 新浪只给最近约 4000 根，和旧缓存合并以便慢慢攒长
            old = pd.read_csv(path, index_col=0, parse_dates=True)
            new = pd.concat([old, new]).loc[lambda d: ~d.index.duplicated(keep="last")].sort_index()
        CACHE.mkdir(parents=True, exist_ok=True)
        new.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _bin(path: Path) -> tuple[int, np.ndarray]:
    raw = path.read_bytes()
    return int(struct.unpack("<f", raw[:4])[0]), np.frombuffer(raw[4:], dtype="<f4")


def load_1d(code: str) -> pd.DataFrame:
    cal = pd.to_datetime((QLIB_DIR / "calendars/day.txt").read_text().split())
    cols = {}
    for f in ("open", "high", "low", "close", "volume"):
        start, v = _bin(QLIB_DIR / f"features/{code.lower()}/{f}.day.bin")
        cols[f] = pd.Series(v, index=cal[start:start + len(v)])
    return pd.DataFrame(cols).dropna()


def prepare(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    d = df.copy()
    c, h, l, o = d["close"], d["high"], d["low"], d["open"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["body"] = (c - o).abs()
    d["wrb"] = (d["body"] > d["body"].shift(1).rolling(LOOKBACK).max()) & (d["body"] > WRB_ATR * d["atr"])
    for n in (20, 60, 200):
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    d["date"] = d.index.normalize()
    daily_close = c.groupby(d["date"]).last()
    d["prev_day_close"] = d["date"].map(daily_close.shift())
    if intraday:
        # 日内量能 U 形：只和同一时段比，否则 10:30 永远"放量"
        slot = d.index.strftime("%H:%M")
        d["vol_ratio"] = d["volume"] / d.groupby(slot)["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    else:
        d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
    return d


# --------------------------------------------------------------------------- 信号


def find_signals(d: pd.DataFrame) -> list[dict]:
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    wrb, atr, body = d["wrb"].to_numpy(), d["atr"].to_numpy(), d["body"].to_numpy()
    n, out = len(d), []
    for j in range(2, n):
        i = j - 1
        if not wrb[i]:
            continue
        if c[i] > o[i] and l[j] > h[i - 1]:
            side, lo, hi = 1, h[i - 1], l[j]
        elif c[i] < o[i] and h[j] < l[i - 1]:
            side, lo, hi = -1, h[j], l[i - 1]
        else:
            continue
        mid = (lo + hi) / 2
        # 首测：HG 确认（j 收盘）之后第一根触及区间的 K 线，只看这一根
        for k in range(j + 1, n):
            touched = l[k] <= hi if side == 1 else h[k] >= lo
            if not touched:
                continue
            ok = (c[k] > mid and c[k] > o[k]) if side == 1 else (c[k] < mid and c[k] < o[k])
            if ok and k + 1 < n:
                out.append(dict(side=side, wrb=i, hg=j, sig=k, lo=lo, hi=hi, mid=mid,
                                wrb_lo=l[i], wrb_hi=h[i], atr=atr[k],
                                age=k - j, size=(hi - lo) / atr[j],
                                strength=body[i] / atr[i], vol_ratio=d["vol_ratio"].iat[i]))
            break
    return out


def simulate(d: pd.DataFrame, s: dict, tp_r: float, stop_mode: str, t1: bool) -> dict | None:
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    dates = d["date"].to_numpy()
    side, e = s["side"], s["sig"] + 1
    entry = o[e]
    if side == 1 and entry >= d["prev_day_close"].iat[e] * (1 + d.attrs.get("limit", 0.1) - 0.003) \
            and d.index[e].hour <= 10:
        return None  # 开盘涨停，买不进
    if stop_mode == "A":
        stop = s["lo"] - 0.1 * s["atr"] if side == 1 else s["hi"] + 0.1 * s["atr"]
    else:
        stop = s["wrb_lo"] if side == 1 else s["wrb_hi"]
    risk = (entry - stop) * side
    if risk < MIN_RISK_ATR * s["atr"]:
        return None  # 开盘已穿止损或贴着止损：R 分母过小，一次跳空就是几十个 R
    tp = entry + side * tp_r * risk
    last = min(e + MAX_HOLD - 1, len(d) - 1)
    exit_px, mfe, mae, b = None, 0.0, 0.0, e
    for b in range(e, len(d)):
        fav = (h[b] - entry) * side if side == 1 else (entry - l[b])
        adv = (entry - l[b]) if side == 1 else (h[b] - entry)
        can_exit = not (t1 and side == 1 and dates[b] == dates[e])
        if can_exit:
            op = o[b]
            if b > e and (op - stop) * side <= 0:
                exit_px = op
            elif b > e and (op - tp) * side >= 0:
                exit_px = op
            elif (l[b] <= stop) if side == 1 else (h[b] >= stop):
                exit_px = stop
            elif (h[b] >= tp) if side == 1 else (l[b] <= tp):
                exit_px = tp
        mfe, mae = max(mfe, fav), max(mae, adv)
        if exit_px is None and b >= last and can_exit:
            exit_px = c[b]
        if exit_px is not None:
            break
    if exit_px is None:
        return None  # 数据末尾未平仓
    r = (exit_px - entry) * side / risk
    return dict(entry_time=d.index[e], R=r, R_net=r - COST * entry / risk,
                mfe=mfe / risk, mae=mae / risk, bars=b - e + 1, risk_pct=risk / entry)


def run(d, sigs, tp_r=2.0, stop_mode="A", side=1, model="A", t1=True) -> pd.DataFrame:
    rows = []
    for s in sigs:
        if s["side"] != side:
            continue
        k = s["sig"]
        cl, e20, e60, e200 = (d[x].iat[k] for x in ("close", "ema20", "ema60", "ema200"))
        if k < 200 and model != "A":
            continue  # EMA200 未热身
        if model == "B" and not ((cl > e200) if side == 1 else (cl < e200)):
            continue
        if model == "C" and not ((cl > e200 and e20 > e60 > e200) if side == 1 else (cl < e200 and e20 < e60 < e200)):
            continue
        t = simulate(d, s, tp_r, stop_mode, t1)
        if t:
            rows.append({**s, **t})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 统计


def stats(t: pd.DataFrame, col: str = "R") -> dict:
    if t.empty:
        return {"Trades": 0}
    r = t.sort_values("entry_time")[col]
    win, loss = r[r > 0], r[r <= 0]
    cum = r.cumsum()
    streak = best = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)
    return {
        "Trades": len(r),
        "Win%": f"{len(win) / len(r):.0%}",
        "AvgWin": round(win.mean(), 2) if len(win) else 0,
        "AvgLoss": round(loss.mean(), 2) if len(loss) else 0,
        "AvgR": round(r.mean(), 3),
        "MedR": round(r.median(), 2),
        "PF": round(win.sum() / -loss.sum(), 2) if loss.sum() < 0 else np.inf,
        "MaxDD_R": round((cum.cummax().clip(lower=0) - cum).max(), 2),
        "Sharpe/trade": round(r.mean() / r.std(), 2) if len(r) > 1 else np.nan,
        "MaxLoseRun": best,
        "MFE_med": round(t["mfe"].median(), 2),
        "MAE_med": round(t["mae"].median(), 2),
        "NetAvgR": round(t["R_net"].mean(), 3),
        "t": round(r.mean() / r.std() * np.sqrt(len(r)), 2) if len(r) > 1 else np.nan,
    }


def buckets(t: pd.DataFrame, col: str, edges: list[float], labels: list[str]) -> pd.DataFrame:
    g = pd.cut(t[col], edges, labels=labels, right=True)
    return t.groupby(g, observed=False)["R"].agg(n="count", AvgR="mean", Win=lambda x: (x > 0).mean()).round(2)


def universe(market: str) -> list[str]:
    rows = [ln.split() for ln in (QLIB_DIR / f"instruments/{market}.txt").read_text().splitlines() if ln.strip()]
    latest = max(r[2] for r in rows)
    return sorted({r[0] for r in rows if r[2] >= latest})


def load_frames(codes: list[str], refresh: bool, sleep: float) -> dict[str, list]:
    """每个周期一组 (code, df, 信号)。信号按股票各自找，交易再合并。"""
    frames: dict[str, list] = {"60m": [], "1D(同窗口)": [], "1D(全史)": []}
    fails = []
    for i, code in enumerate(codes, 1):
        cached = (CACHE / f"{code.lower()}_60m.csv").exists()
        try:
            h60 = load_60m(code, refresh)
            d1 = load_1d(code)
        except Exception as exc:  # 单只失败不中断
            fails.append(f"{code}:{type(exc).__name__}")
            continue
        limit = 0.2 if code[2:5] in ("688", "300", "301") else 0.1
        parts = {
            "60m": prepare(h60, True),
            "1D(同窗口)": prepare(d1, False).loc[lambda x: x.index >= h60.index[0].normalize()],
            "1D(全史)": prepare(d1, False),
        }
        for k, d in parts.items():
            d.attrs["limit"] = limit
            frames[k].append((code, d, find_signals(d)))
        if (refresh or not cached) and len(codes) > 1:
            time.sleep(sleep)
        if i % 50 == 0:
            print(f"  {i}/{len(codes)}", flush=True)
    if fails:
        print(f"失败 {len(fails)} 只: {' '.join(fails[:10])}")
    return frames


def run_all(parts: list, **kw) -> pd.DataFrame:
    ts = [run(d, s, **kw).assign(code=c) for c, d, s in parts]
    ts = [t for t in ts if not t.empty]
    return pd.concat(ts, ignore_index=True) if ts else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?")
    ap.add_argument("--pool", help="qlib 成分表名，如 csi300")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    codes = universe(args.pool) if args.pool else [args.code]
    frames = load_frames(codes, args.refresh, args.sleep)
    for k, parts in frames.items():
        n_bar = sum(len(d) for _, d, _ in parts)
        span = f"{min(d.index[0] for _, d, _ in parts).date()} ~ {max(d.index[-1] for _, d, _ in parts).date()}"
        n_wrb = sum(int(d["wrb"].sum()) for _, d, _ in parts)
        n_l = sum(x["side"] == 1 for _, _, s in parts for x in s)
        n_s = sum(x["side"] == -1 for _, _, s in parts for x in s)
        print(f"{k}: {len(parts)} 只，{n_bar} 根，{span}；WRB {n_wrb}，首测信号 多 {n_l} / 空 {n_s}")

    cols = ["Trades", "Win%", "AvgWin", "AvgLoss", "AvgR", "MedR", "PF", "t", "MaxLoseRun", "MFE_med", "MAE_med", "NetAvgR"]
    base = {k: run_all(frames[k]) for k in frames}
    print("\n== 表1 核心：纯 HG 做多，止损 A，TP 2R，持有 20 根 ==")
    print(pd.DataFrame({k: stats(base[k]) for k in frames}).T[cols].to_string())

    print("\n== 表2 TP 连续性（纯 HG 做多，止损 A）==")
    rows = {}
    for k in frames:
        for tp in (1, 1.5, 2, 3):
            st = stats(run_all(frames[k], tp_r=tp))
            rows[(k, f"{tp}R")] = {x: st.get(x) for x in ("Trades", "Win%", "AvgR", "PF", "t", "NetAvgR")}
    print(pd.DataFrame(rows).T.to_string())

    print("\n== 表3 止损 A(HG-0.1ATR) vs B(WRB low)，TP 2R ==")
    rows = {(k, m): stats(run_all(frames[k], stop_mode=m)) for k in frames for m in ("A", "B")}
    print(pd.DataFrame(rows).T[["Trades", "Win%", "AvgR", "PF", "t", "MFE_med", "MAE_med", "NetAvgR"]].to_string())

    print("\n== 表4 趋势过滤（做多，TP 2R）AvgR (n, t) ==")
    rows = {}
    for k in frames:
        rows[k] = {}
        for m, name in (("A", "Pure HG"), ("B", "+EMA200"), ("C", "+Trend")):
            st = stats(run_all(frames[k], model=m))
            rows[k][name] = f"{st.get('AvgR', float('nan'))} ({st['Trades']}, t={st.get('t')})"
    print(pd.DataFrame(rows).T.to_string())

    print("\n== 表5 空头 HG 预测力（只统计，不受 T+1 约束），止损 A，TP 2R ==")
    print(pd.DataFrame({k: stats(run_all(frames[k], side=-1, t1=False)) for k in frames}).T[cols].to_string())

    print("\n== 表6 按年（纯 HG 做多，TP 2R）AvgR (n) ==")
    rows = {}
    for k in frames:
        t = base[k]
        if t.empty:
            continue
        g = t.groupby(t["entry_time"].dt.year)["R"]
        rows[k] = (g.mean().round(2).astype(str) + " (" + g.count().astype(str) + ")").to_dict()
    print(pd.DataFrame(rows).fillna("").to_string())

    for k in ("60m", "1D(全史)"):
        t = base[k]
        if t.empty:
            continue
        print(f"\n== 分桶 {k}（纯 HG 做多，TP 2R）==")
        print("HG 年龄（根）\n", buckets(t, "age", [0, 1, 2, 5, 10, 20, 50, 1e9], ["1", "2", "3-5", "6-10", "11-20", "21-50", ">50"]).to_string())
        print("HG 大小 / ATR\n", buckets(t, "size", [0, 0.2, 0.5, 1, 1e9], ["<0.2", "0.2-0.5", "0.5-1", ">1"]).to_string())
        print("WRB 强度 body/ATR\n", buckets(t, "strength", [1.2, 1.5, 2, 3, 1e9], ["1.2-1.5", "1.5-2", "2-3", ">3"]).to_string())
        print("WRB 量比\n", buckets(t, "vol_ratio", [0, 1, 1.5, 2, 3, 1e9], ["<1", "1-1.5", "1.5-2", "2-3", ">3"]).to_string())
        good = ((t["mae"] < 0.4) & (t["mfe"] > 1.5)).mean()
        print(f"MAE<0.4R 且 MFE>1.5R 的占比: {good:.0%}")
        print("MFE 分位 (R):", t["mfe"].quantile([.25, .5, .75]).round(2).tolist(),
              " MAE 分位 (R):", t["mae"].quantile([.25, .5, .75]).round(2).tolist())
        top = t.groupby("code")["R"].sum().sort_values()
        print(f"R 贡献集中度: 最好 5 只合计 {top.tail(5).sum():.1f}R，最差 5 只 {top.head(5).sum():.1f}R，总计 {t['R'].sum():.1f}R")
    if args.pool:
        CACHE.mkdir(parents=True, exist_ok=True)
        for k, t in base.items():
            t.to_csv(CACHE / f"trades_{args.pool}_{k}.csv", index=False)


if __name__ == "__main__":
    main()

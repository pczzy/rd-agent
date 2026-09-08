"""按笔记录实际成交，维护持仓与现金。

`run_daily.py --confirm` 假设整张清单原样成交 —— 它把当日**目标组合**直接写成持仓。
实盘常常只成交一部分（挂限价单、分批下手、手动挑着买），这时必须按笔记账，否则
系统里的持仓是一个从未存在过的组合。

    # 记一天的成交（卖出用负股数）
    python production/record_trades.py --date 2026-09-04 SZ002594:100@87.31 SZ000333:100@87.25

    # 股数/价格照抄某天的订单清单（只在确实按清单价成交时才对）
    python production/record_trades.py --date 2026-09-04 --from-order 2026-09-03 SZ002594 SZ000333

    python production/record_trades.py --show      # 看当前持仓和现金
    python production/record_trades.py --rebuild   # 改完流水后重放

`state/trades.csv` 是唯一事实来源，追加式流水；`positions.json` 和 `cash.json` 都由它
重放得出。填错了改流水再 `--rebuild` 即可，不必手工去改派生状态 —— 手改派生状态的话，
下一次重放又会把它盖掉，错误会以为已经修好了。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from production.pipeline import load_config  # noqa: E402

LEDGER = ROOT / "state/trades.csv"
POSITIONS = ROOT / "state/positions.json"
CASH = ROOT / "state/cash.json"
NAMES = ROOT / "state/names.json"
COLUMNS = ["date", "code", "name", "shares", "price", "fee"]
CODE_RE = re.compile(r"^S[HZ]\d{6}$")
SPEC_RE = re.compile(r"^(?P<code>[A-Za-z]{2}\d{6})(?::(?P<shares>-?\d+))?(?:@(?P<price>[\d.]+))?$")


def names() -> dict[str, str]:
    return json.loads(NAMES.read_text()) if NAMES.exists() else {}


def fee_of(value: float, sell: bool, cfg: dict) -> float:
    """实际扣款口径：佣金 + 过户费（双边）+ 印花税（仅卖出）。

    这里**不能**用 config 的 cost_one_side —— 那个 0.191% 是事前估算用的混合数，
    含半价差和冲击成本，它们体现在成交价里而不是券商扣款里。用它算现金会重复扣两遍。
    """
    trd = cfg["trading"]
    commission = max(trd["min_commission"], value * trd["commission_rate"])
    transfer = value * trd["transfer_rate"]
    stamp = value * trd["stamp_duty"] if sell else 0.0
    return round(commission + transfer + stamp, 2)


def parse_spec(spec: str, order: pd.DataFrame | None) -> tuple[str, int, float]:
    m = SPEC_RE.match(spec.strip())
    if not m:
        raise SystemExit(f"看不懂 {spec!r}，格式是 CODE:股数@价格，例如 SZ002594:100@87.31")
    code = m["code"].upper()
    if not CODE_RE.match(code):
        raise SystemExit(f"代码 {code} 不合法，应形如 SH600000 / SZ000001")

    shares, price = m["shares"], m["price"]
    if shares is None or price is None:
        if order is None:
            raise SystemExit(f"{code} 没给全股数和价格，要么补全，要么加 --from-order 某日")
        row = order[order["code"] == code]
        if row.empty:
            raise SystemExit(f"{code} 不在该日订单清单里，请显式写出股数和价格")
        row = row.iloc[0]
        if shares is None:
            shares = int(row["shares"]) * (-1 if str(row["side"]).upper() == "SELL" else 1)
        if price is None:
            price = float(row["price"])
    return code, int(shares), float(price)


def replay(cfg: dict) -> tuple[dict[str, dict], float, str | None]:
    """重放流水，得出持仓（含加权平均成本）和现金。

    加仓成本按股数加权平均，减仓成本不变 —— 与 pipeline.save_positions 同一口径，
    两条写状态的路径若口径不同，浮亏告警会随记账方式漂移。
    """
    cash = float(cfg["account"]["capital"])
    book: dict[str, dict] = {}
    if not LEDGER.exists():
        return book, cash, None

    led = pd.read_csv(LEDGER)
    led = led.sort_values("date", kind="stable")
    for _, t in led.iterrows():
        code, shares, price, fee = t["code"], int(t["shares"]), float(t["price"]), float(t["fee"])
        cash -= shares * price + fee            # 卖出时 shares 为负，等于收回现金
        held = book.get(code, {"shares": 0, "cost": 0.0})
        if shares > 0:
            total = held["shares"] + shares
            held["cost"] = (held["shares"] * held["cost"] + shares * price) / total
            held["shares"] = total
        else:
            held["shares"] += shares            # 减仓不动成本价
            if held["shares"] < 0:
                raise SystemExit(f"{code} 卖出 {-shares} 股超过持有量，检查 {LEDGER}")
        book[code] = held
    book = {c: h for c, h in book.items() if h["shares"] > 0}
    return book, round(cash, 2), str(led["date"].max())


def write_state(book: dict[str, dict], cash: float, as_of: str | None) -> None:
    POSITIONS.write_text(
        json.dumps(
            {c: {"shares": int(h["shares"]), "cost": round(h["cost"], 4)} for c, h in sorted(book.items())},
            indent=1,
        )
    )
    CASH.write_text(json.dumps({"cash": cash, "as_of": as_of}, indent=1))


def show(book: dict[str, dict], cash: float, as_of: str | None) -> None:
    nm = names()
    if book:
        rows = [
            {
                "code": c,
                "name": nm.get(c, ""),
                "shares": h["shares"],
                "cost": round(h["cost"], 3),
                "成本市值": round(h["shares"] * h["cost"], 0),
            }
            for c, h in sorted(book.items(), key=lambda kv: -kv[1]["shares"] * kv[1]["cost"])
        ]
        print(pd.DataFrame(rows).to_string(index=False))
        invested = sum(r["成本市值"] for r in rows)
    else:
        print("（空仓）")
        invested = 0.0
    print(f"\n持仓 {len(book)} 只 | 成本市值 {invested:,.0f} 元 | 现金 {cash:,.0f} 元 "
          f"| 合计 {invested + cash:,.0f} 元（截至 {as_of or '无记录'}）")
    print("注：成本市值按买入价，不是当前市值。当前市值和回撤看 run_daily.py --risk-only。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trades", nargs="*", metavar="CODE:股数@价格", help="卖出用负股数")
    ap.add_argument("--date", help="成交日期 YYYY-MM-DD")
    ap.add_argument("--from-order", metavar="DATE", help="缺省的股数/价格照抄该日订单清单")
    ap.add_argument("--show", action="store_true", help="只看当前状态")
    ap.add_argument("--rebuild", action="store_true", help="按流水重放派生状态")
    ap.add_argument("--force", action="store_true", help="允许写入与流水完全重复的一笔")
    args = ap.parse_args()

    cfg = load_config()

    if args.show or args.rebuild:
        book, cash, as_of = replay(cfg)
        if args.rebuild:
            write_state(book, cash, as_of)
            print(f"已按 {LEDGER} 重放并写回 positions.json / cash.json\n")
        show(book, cash, as_of)
        return 0

    if not args.trades or not args.date:
        ap.print_help()
        return 1
    try:
        date = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"日期 {args.date!r} 解析不了")

    order = None
    if args.from_order:
        path = ROOT / f"orders/{args.from_order}.csv"
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        order = pd.read_csv(path)

    nm = names()
    new = []
    for spec in args.trades:
        code, shares, price = parse_spec(spec, order)
        if shares == 0:
            raise SystemExit(f"{code} 股数为 0")
        if shares > 0 and shares % cfg["trading"]["lot_size"]:
            print(f"  [注意] {code} 买入 {shares} 股不是整手，按实际成交记，不拦")
        value = abs(shares) * price
        new.append(
            {
                "date": date,
                "code": code,
                "name": nm.get(code, ""),
                "shares": shares,
                "price": price,
                "fee": fee_of(value, shares < 0, cfg),
            }
        )

    led = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=COLUMNS)
    if not args.force and not led.empty:
        for t in new:
            dup = led[(led["date"] == t["date"]) & (led["code"] == t["code"])
                      & (led["shares"] == t["shares"]) & (led["price"] == t["price"])]
            if not dup.empty:
                raise SystemExit(
                    f"{t['date']} {t['code']} {t['shares']}股@{t['price']} 流水里已有同样一笔。"
                    "确实分两笔成交就加 --force"
                )
    fresh = pd.DataFrame(new)
    # 空表参与 concat 会丢列的 dtype（pandas FutureWarning），首次记账直接用新表
    led = fresh if led.empty else pd.concat([led, fresh], ignore_index=True)
    led = led.sort_values("date", kind="stable")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    led.to_csv(LEDGER, index=False)

    for t in new:
        side = "买入" if t["shares"] > 0 else "卖出"
        print(f"  记入 {t['date']} {side} {t['code']} {t['name']} "
              f"{abs(t['shares'])}股 @{t['price']} 费用 {t['fee']:.2f} 元")

    book, cash, as_of = replay(cfg)
    write_state(book, cash, as_of)
    print()
    show(book, cash, as_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

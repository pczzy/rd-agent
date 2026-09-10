"""按笔记录实际成交（命令行）。看板 http://<本机>:8501 的「持仓管理」页做同样的事。

`run_daily.py --confirm` 假设整张清单原样成交 —— 它把当日**目标组合**直接写成持仓。
实盘常常只成交一部分（挂限价单、分批下手、手动挑着买），这时必须按笔记账，否则系统里
的持仓是一个从未存在过的组合。

    # 卖出用负股数；--time 可选，但填了以后成本价就能回答"这是什么时候的价"
    python production/record_trades.py --date 2026-09-04 --time 10:31 SZ002594:100@87.31

    # 股数/价格照抄某天的订单清单（只在确实按清单价成交时才对）
    python production/record_trades.py --date 2026-09-04 --from-order 2026-09-03 SZ002594

    python production/record_trades.py --show      # 看持仓和现金
    python production/record_trades.py --changes   # 看持仓变动明细
    python production/record_trades.py --rebuild   # 改完流水后重放

流水在 `state/trades.csv`，是唯一事实来源；positions.json 和 cash.json 都由它重放得出。
填错了改流水再 `--rebuild`，别手工改派生状态 —— 下次重放会把手改盖掉。
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

from production.ledger import (  # noqa: E402
    LEDGER,
    add_trades,
    changes,
    duplicate_of,
    fee_of,
    read_ledger,
    rebuild,
    replay,
)
from production.pipeline import load_config  # noqa: E402

CODE_RE = re.compile(r"^S[HZ]\d{6}$")
SPEC_RE = re.compile(r"^(?P<code>[A-Za-z]{2}\d{6})(?::(?P<shares>-?\d+))?(?:@(?P<price>[\d.]+))?$")


def names() -> dict[str, str]:
    p = ROOT / "state/names.json"
    return json.loads(p.read_text()) if p.exists() else {}


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


def show(book: dict[str, dict], cash: float, as_of: str | None) -> None:
    nm = names()
    if book:
        rows = [{
            "code": c, "name": nm.get(c, ""), "shares": h["shares"],
            "cost": round(h["cost"], 3),
            "成本形成于": h["first"] if h["lots"] == 1 else f"{h['first']} ~ {h['last']}（{h['lots']} 笔加权）",
            "成本市值": round(h["shares"] * h["cost"], 0),
        } for c, h in sorted(book.items(), key=lambda kv: -kv[1]["shares"] * kv[1]["cost"])]
        print(pd.DataFrame(rows).to_string(index=False))
        invested = sum(r["成本市值"] for r in rows)
    else:
        print("（空仓）")
        invested = 0.0
    print(f"\n持仓 {len(book)} 只 | 成本市值 {invested:,.0f} 元 | 现金 {cash:,.0f} 元 "
          f"| 合计 {invested + cash:,.0f} 元（末笔成交 {as_of or '无记录'}）")
    print("注：成本市值按买入价，不是当前市值。当前市值和回撤看 run_daily.py --risk-only 或看板。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trades", nargs="*", metavar="CODE:股数@价格", help="卖出用负股数")
    ap.add_argument("--date", help="成交日期 YYYY-MM-DD")
    ap.add_argument("--time", default="", help="成交时间 HH:MM，可选但建议填")
    ap.add_argument("--note", default="", help="备注，例如 限价单/分批第2笔")
    ap.add_argument("--from-order", metavar="DATE", help="缺省的股数/价格照抄该日订单清单")
    ap.add_argument("--show", action="store_true", help="只看当前状态")
    ap.add_argument("--changes", action="store_true", help="看持仓变动明细")
    ap.add_argument("--rebuild", action="store_true", help="按流水重放派生状态")
    ap.add_argument("--force", action="store_true", help="允许写入与流水完全重复的一笔")
    args = ap.parse_args()

    cfg = load_config()

    if args.changes:
        ch = changes(cfg)
        print(ch.to_string(index=False) if not ch.empty else "流水为空")
        return 0
    if args.show or args.rebuild:
        book, cash, as_of = rebuild(cfg) if args.rebuild else replay(cfg)
        if args.rebuild:
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
    tm = args.time.strip()
    if tm and not re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", tm):
        raise SystemExit(f"时间 {tm!r} 解析不了，格式是 HH:MM")

    order = None
    if args.from_order:
        path = ROOT / f"orders/{args.from_order}.csv"
        if not path.exists():
            raise SystemExit(f"找不到 {path}")
        order = pd.read_csv(path)

    nm, led, new = names(), read_ledger(), []
    for spec in args.trades:
        code, shares, price = parse_spec(spec, order)
        if shares == 0:
            raise SystemExit(f"{code} 股数为 0")
        if shares > 0 and shares % cfg["trading"]["lot_size"]:
            print(f"  [注意] {code} 买入 {shares} 股不是整手，按实际成交记，不拦")
        row = {"date": date, "time": tm, "code": code, "name": nm.get(code, ""),
               "shares": shares, "price": price,
               "fee": fee_of(abs(shares) * price, shares < 0, cfg), "note": args.note}
        if not args.force and duplicate_of(led, row):
            raise SystemExit(f"{date} {code} {shares}股@{price} 流水里已有同样一笔。"
                             "确实分两笔成交就加 --force")
        new.append(row)

    book, cash, as_of = add_trades(new, cfg)
    for t in new:
        side = "买入" if t["shares"] > 0 else "卖出"
        when = f"{t['date']} {t['time']}".strip()
        print(f"  记入 {when} {side} {t['code']} {t['name']} "
              f"{abs(t['shares'])}股 @{t['price']} 费用 {t['fee']:.2f} 元")
    print()
    show(book, cash, as_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

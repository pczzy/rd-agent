"""成交流水：持仓与现金的唯一事实来源。

`state/trades.csv` 是追加式流水，`positions.json` 和 `cash.json` 都由它重放得出。
命令行（`record_trades.py`）和看板（`dashboard.py`）共用这里的实现 —— 两份实现迟早会
在成本价口径上漂移，而那会让浮亏告警随记账方式变化。

**每个价格都带时间。** 成交价记 `date` + `time`（时间可留空），持仓成本价记它由哪几笔、
从什么时候加权而来。看板上任何一处价格都要能回答"这是什么时候的价"。
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "state/trades.csv"
POSITIONS = ROOT / "state/positions.json"
CASH = ROOT / "state/cash.json"
LOCK = ROOT / "state/.ledger.lock"
COLUMNS = ["date", "time", "code", "name", "shares", "price", "fee", "note"]


@contextmanager
def _locked():
    """网页和命令行可能同时写，加把锁。"""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def stamp(row) -> str:
    """一笔成交的时间戳。时间没填就只给日期 —— 不编一个不知道的钟点出来。"""
    t = str(row.get("time") or "").strip()
    return f"{row['date']} {t}" if t and t.lower() != "nan" else str(row["date"])


def read_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=COLUMNS)
    led = pd.read_csv(LEDGER, dtype={"time": str, "note": str})
    for c in COLUMNS:                       # 老流水没有 time/note 两列
        if c not in led.columns:
            led[c] = ""
    led["time"] = led["time"].fillna("")
    led["note"] = led["note"].fillna("")
    return led[COLUMNS].sort_values(["date", "time"], kind="stable").reset_index(drop=True)


def _cent(x: float) -> float:
    """四舍五入到分。

    不能用内置 round()：它是银行家进位（round(2.675, 2) == 2.67、
    round(17.505, 2) == 17.5），而回单上的进位是四舍五入。
    """
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def fee_parts(value: float, sell: bool, cfg: dict, code: str = "SH") -> dict[str, float]:
    """按回单的科目拆分：佣金 / 过户费 / 印花税，**各自进位到分**。

    回单是分项列示、分项进位的，所以这里也分项进位再相加 —— 先加总再进位会差 1 分
    （两项各 1.004 时，分项进位得 2.00，合计进位得 2.01）。

    过户费按市场区分。券商《佣金标准》写明：沪市佣金含经手费、结算费、证管费，
    **过户费单独向客户收取**；深市佣金里**已经包含过户费**，再加一遍就是重复计费。
    北交所/股转同深市。

    不能用 config 的 cost_one_side —— 那是含半价差和冲击的事前估算，这两项体现在
    成交价里而不是扣款里，拿它算现金会重复扣两遍。
    """
    trd = cfg["trading"]
    is_sh = str(code).upper().startswith("SH")
    return {
        "佣金": _cent(max(trd["min_commission"], value * trd["commission_rate"])),
        "过户费": _cent(value * trd["transfer_rate"]) if is_sh else 0.0,
        "印花税": _cent(value * trd["stamp_duty"]) if sell else 0.0,
    }


def fee_of(value: float, sell: bool, cfg: dict, code: str = "SH") -> float:
    """券商实际扣款合计。明细见 fee_parts()。"""
    return _cent(sum(fee_parts(value, sell, cfg, code).values()))


def replay(cfg: dict) -> tuple[dict[str, dict], float, str | None]:
    """重放流水，得出持仓（含加权平均成本及其形成区间）和现金余额。

    加仓成本按股数加权平均、减仓成本不变 —— 与 pipeline.save_positions 同一口径。
    成本价是多笔加权的结果，所以它的"时间"是一个区间：first/last 记第一笔和最后一笔加仓。
    """
    cash = float(cfg["account"]["capital"])
    book: dict[str, dict] = {}
    led = read_ledger()
    if led.empty:
        return book, cash, None

    for _, t in led.iterrows():
        code, shares, price, fee = t["code"], int(t["shares"]), float(t["price"]), float(t["fee"])
        cash -= shares * price + fee                  # 卖出 shares 为负，等于收回现金
        held = book.get(code, {"shares": 0, "cost": 0.0, "lots": 0, "first": None, "last": None})
        if shares > 0:
            total = held["shares"] + shares
            held["cost"] = (held["shares"] * held["cost"] + shares * price) / total
            held["shares"] = total
            held["lots"] += 1
            held["first"] = held["first"] or stamp(t)
            held["last"] = stamp(t)
        else:
            held["shares"] += shares
            if held["shares"] < 0:
                raise ValueError(f"{code} 卖出 {-shares} 股超过持有量，检查 {LEDGER}")
        book[code] = held
    book = {c: h for c, h in book.items() if h["shares"] > 0}
    return book, round(cash, 2), stamp(led.iloc[-1])


def changes(cfg: dict) -> pd.DataFrame:
    """持仓变动明细：每一笔成交前后的股数、成本价、现金。

    流水记的是"做了什么"，这里给的是"变成了什么" —— 对账时要看的是后者。
    """
    cash = float(cfg["account"]["capital"])
    book: dict[str, dict] = {}
    rows = []
    for _, t in read_ledger().iterrows():
        code, shares, price, fee = t["code"], int(t["shares"]), float(t["price"]), float(t["fee"])
        held = book.get(code, {"shares": 0, "cost": 0.0})
        s0, c0, cash0 = held["shares"], held["cost"], cash
        cash -= shares * price + fee
        if shares > 0:
            total = s0 + shares
            held = {"shares": total, "cost": (s0 * c0 + shares * price) / total}
        else:
            held = {"shares": s0 + shares, "cost": c0}
        book[code] = held
        rows.append({
            "成交时间": stamp(t), "代码": code, "名称": t["name"],
            "方向": "买入" if shares > 0 else "卖出", "股数": abs(shares),
            "成交价": price, "费用": fee,
            "持股": f"{s0} → {held['shares']}",
            "成本价": f"{c0:.3f} → {held['cost']:.3f}" if s0 or held["shares"] else f"{held['cost']:.3f}",
            "现金": f"{cash0:,.0f} → {cash:,.0f}",
            "备注": t["note"],
        })
    return pd.DataFrame(rows)


def write_state(book: dict[str, dict], cash: float, as_of: str | None) -> None:
    POSITIONS.write_text(json.dumps(
        {c: {"shares": int(h["shares"]), "cost": round(h["cost"], 4),
             "cost_lots": h.get("lots", 1),
             "cost_from": h.get("first"), "cost_to": h.get("last")}
         for c, h in sorted(book.items())}, indent=1, ensure_ascii=False))
    CASH.write_text(json.dumps({"cash": cash, "as_of": as_of}, indent=1, ensure_ascii=False))


def rebuild(cfg: dict) -> tuple[dict[str, dict], float, str | None]:
    book, cash, as_of = replay(cfg)
    write_state(book, cash, as_of)
    return book, cash, as_of


def add_trades(new: list[dict], cfg: dict) -> tuple[dict[str, dict], float, str | None]:
    """追加成交并重放。整个过程在锁里，网页和命令行同时写也不会互相覆盖。"""
    with _locked():
        led = read_ledger()
        fresh = pd.DataFrame(new)
        # 空表参与 concat 会丢列的 dtype（pandas FutureWarning），首次记账直接用新表
        led = fresh if led.empty else pd.concat([led, fresh], ignore_index=True)
        led = led[COLUMNS].sort_values(["date", "time"], kind="stable").reset_index(drop=True)
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        led.to_csv(LEDGER, index=False)
        return rebuild(cfg)


def drop_trade(index: int, cfg: dict) -> tuple[dict[str, dict], float, str | None]:
    """删掉流水里的一行并重放。填错了就删了重记，不要去手改派生状态。"""
    with _locked():
        led = read_ledger()
        if index not in led.index:
            raise ValueError(f"流水里没有第 {index} 行")
        led.drop(index=index).to_csv(LEDGER, index=False)
        return rebuild(cfg)


def duplicate_of(led: pd.DataFrame, row: dict) -> bool:
    """同日同股同量同价，多半是重复提交而不是真的分两笔。"""
    if led.empty:
        return False
    same = led[(led["date"] == row["date"]) & (led["code"] == row["code"])
               & (led["shares"] == row["shares"]) & (led["price"] == row["price"])]
    return not same.empty

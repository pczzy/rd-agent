"""每日信号 -> 目标组合 -> 订单清单。

刻意不连券商：本模块只产出 orders/*.csv，下单由人确认后执行。

流程与 docs/worklog/WORKLOG.md 里通过 holdout 验证的那套一致 —— 因子在 csi300 上计算
（截面缩尾必须用实际交易的股票池，否则那步是空转），20 日前瞻标签，信号每 10 个交易日
刷新，topk=30，只在流动性前 50% 里选。偏离这些会让回测数字失效。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config.yaml").read_text())


# --------------------------------------------------------------------------- 因子


def compute_factors(h5_path: Path, out_path: Path) -> pd.DataFrame:
    """在最新行情上跑 factors/ 下的每个因子，横向拼成一张表。

    每个因子在独立子进程里跑：它们是 LLM 生成的代码，彼此的全局状态不该互相影响，
    并且单个因子失败不应拖垮整批。
    """
    sys.path.insert(0, str(REPO))
    from rdagent.scenarios.qlib.developer.utils import _combine_factor_frames, feature_columns

    frames, failed = [], []
    for src in sorted((ROOT / "factors").glob("[0-9][0-9]_*.py")):
        work = Path(tempfile.mkdtemp())
        (work / "factor.py").write_text(src.read_text())
        (work / "daily_pv.h5").symlink_to(h5_path)
        proc = subprocess.run([sys.executable, "factor.py"], cwd=work, capture_output=True, text=True, timeout=3600)
        result = work / "result.h5"
        if proc.returncode != 0 or not result.exists():
            failed.append((src.name, (proc.stderr or "")[-200:]))
            continue
        frame = pd.read_hdf(result, key="data")
        frames.append(frame.to_frame() if isinstance(frame, pd.Series) else frame)

    if failed:
        for name, err in failed:
            print(f"  [因子失败] {name}: {err}", file=sys.stderr)
    if not frames:
        raise RuntimeError("没有任何因子计算成功")

    combined = _combine_factor_frames(frames)
    combined.columns = feature_columns(combined.columns)
    combined = combined.sort_index()
    combined.to_parquet(out_path, engine="pyarrow")
    return combined


# --------------------------------------------------------------------------- 组合


@dataclass
class Position:
    code: str
    shares: int
    price: float

    @property
    def value(self) -> float:
        return self.shares * self.price


def liquid_universe(adv: pd.Series, pct: float) -> pd.Index:
    """当日成交额排名前 pct 的股票。

    小盘/低流动性标的的 alpha 在实盘中兑现不了：holdout 上不做这层过滤，同一信号
    年化从 +10.23% 掉到 -1.44%，回撤从 -9.6% 扩到 -23.9%。
    """
    if adv.empty:
        return adv.index
    return adv[adv >= adv.quantile(1.0 - pct)].index


def build_target(
    scores: pd.Series,
    prices: pd.Series,
    adv: pd.Series,
    cfg: dict,
) -> dict[str, int]:
    """把当日打分转成目标持仓（股数），已做整手和单只上限约束。"""
    acct, uni, trd = cfg["account"], cfg["universe"], cfg["trading"]

    eligible = scores.index.intersection(liquid_universe(adv, uni["liquidity_pct"]))
    eligible = eligible.intersection(prices.dropna().index)
    if len(eligible) == 0:
        return {}

    lot = trd["lot_size"]
    cap_per_stock = acct["capital"] * acct["max_weight_per_stock"]

    # 先按分数排序，再顺着取买得起的，直到凑满 max_positions。
    # 只从前 N 名里筛会漏掉资金：csi300 里一手上万的名字不少（SH688256 一手 10.35 万），
    # 20 万的账户买不起，若不顺延就会有两成多资金闲置。
    ranked = scores.loc[eligible].sort_values(ascending=False).index
    affordable = [c for c in ranked if float(prices[c]) * lot <= cap_per_stock]
    affordable = affordable[: acct["max_positions"]]
    if not affordable:
        return {}

    target: dict[str, int] = {}
    remaining, slots = acct["capital"], len(affordable)
    for code in sorted(affordable, key=lambda c: float(prices[c]) * lot, reverse=True):
        budget = min(remaining / slots, cap_per_stock)
        px = float(prices[code])
        lots = int(budget // (px * lot))
        if lots >= 1:
            shares = lots * lot
            target[code] = shares
            remaining -= shares * px
        slots -= 1
    return target


def generate_orders(
    current: dict[str, int],
    target: dict[str, int],
    prices: pd.Series,
    cfg: dict,
) -> pd.DataFrame:
    """目标 vs 现有持仓求差，产出订单；按风控规则过滤。"""
    risk, trd, acct = cfg["risk"], cfg["trading"], cfg["account"]
    rows: list[dict] = []
    stuck: list[str] = []
    for code in sorted(set(current) | set(target)):
        delta = target.get(code, 0) - current.get(code, 0)
        if delta == 0:
            continue
        px = float(prices.get(code, np.nan))
        if not np.isfinite(px):
            # 手上有仓却拿不到价（退市/长期停牌）必须报出来。静默跳过的话这笔仓位
            # 会一直挂在账上却从不出现在任何订单里，直到有人手工发现。
            if current.get(code, 0):
                stuck.append(code)
            continue
        value = abs(delta) * px
        # 清仓单必须放行：跌破 min_order_value 也要能出掉
        if value < risk["min_order_value"] and target.get(code, 0) != 0:
            continue
        rows.append(
            dict(
                code=code,
                side="BUY" if delta > 0 else "SELL",
                shares=abs(delta),
                price=round(px, 3),
                value=round(value, 2),
            )
        )

    if stuck:
        print(f"  [警告] 持仓无法定价，需人工处理: {', '.join(stuck)}")

    orders = pd.DataFrame(rows, columns=["code", "side", "shares", "price", "value"])
    if orders.empty:
        return orders

    # 换手上限只在已有持仓时生效：从空仓建仓的换手必然等于目标仓位，
    # 拿它去撞上限会把首日订单砍掉大半，永远建不起仓。
    turnover = orders["value"].sum() / acct["capital"]
    if current and turnover > risk["max_daily_turnover"]:
        cap = acct["capital"] * risk["max_daily_turnover"]
        orders = orders.sort_values("value", ascending=False)
        orders = orders[orders["value"].cumsum() <= cap]
        print(f"  [风控] 换手 {turnover:.1%} 超上限，订单截断至 {len(orders)} 笔")

    return orders.sort_values(["side", "value"], ascending=[True, False]).reset_index(drop=True)


def estimate_cost(orders: pd.DataFrame, cfg: dict) -> float:
    """订单的预估交易成本（元）。事前估算，不参与撮合。"""
    if orders.empty:
        return 0.0
    trd = cfg["trading"]
    per_side = orders["value"] * trd["cost_one_side"]
    commission_floor = np.maximum(per_side, trd["min_commission"])
    stamp = orders.loc[orders["side"] == "SELL", "value"].sum() * trd["stamp_duty"]
    return float(commission_floor.sum() + stamp)


# --------------------------------------------------------------------------- 状态


def load_positions(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return {k: int(v) for k, v in json.loads(path.read_text()).items()}


def save_positions(path: Path, positions: dict[str, int]) -> None:
    path.write_text(json.dumps({k: v for k, v in sorted(positions.items()) if v}, indent=1))

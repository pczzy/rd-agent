"""每日信号 -> 目标组合 -> 订单清单。

刻意不连券商：本模块只产出 orders/*.csv，下单由人确认后执行。

流程与 docs/worklog/WORKLOG.md 里通过 holdout 验证的那套一致 —— 因子在 csi300 上计算
（截面缩尾必须用实际交易的股票池，否则那步是空转），20 日前瞻标签，信号每 10 个交易日
刷新，topk=30，只在流动性前 50% 里选。偏离这些会让回测数字失效。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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


def screen(
    scores: pd.Series,
    prices: pd.Series,
    adv: pd.Series,
    cfg: dict,
) -> tuple[pd.Index, list[str]]:
    """三层筛选，返回 (合格池, 最终候选)。

    build_target 和 rank_table 共用这一份：一个决定买什么，一个向人解释为什么没买，
    两处各写一份筛选逻辑迟早会漂移，那时报告里的"未入选原因"就会开始骗人。
    """
    acct, uni, trd = cfg["account"], cfg["universe"], cfg["trading"]

    eligible = scores.index.intersection(liquid_universe(adv, uni["liquidity_pct"]))
    eligible = eligible.intersection(prices.dropna().index)
    if len(eligible) == 0:
        return eligible, []

    cap_per_stock = acct["capital"] * acct["max_weight_per_stock"]
    # 先按分数排序，再顺着取买得起的，直到凑满 max_positions。
    # 只从前 N 名里筛会漏掉资金：csi300 里一手上万的名字不少（SH688256 一手 10.35 万），
    # 20 万的账户买不起，若不顺延就会有两成多资金闲置。
    ranked = scores.loc[eligible].sort_values(ascending=False).index
    affordable = [c for c in ranked if float(prices[c]) * trd["lot_size"] <= cap_per_stock]
    return eligible, affordable[: acct["max_positions"]]


def build_target(
    scores: pd.Series,
    prices: pd.Series,
    adv: pd.Series,
    cfg: dict,
) -> dict[str, int]:
    """把当日打分转成目标持仓（股数），已做整手和单只上限约束。"""
    acct, trd = cfg["account"], cfg["trading"]

    _, affordable = screen(scores, prices, adv, cfg)
    if not affordable:
        return {}

    lot = trd["lot_size"]
    cap_per_stock = acct["capital"] * acct["max_weight_per_stock"]

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
    halted: bool = False,
) -> pd.DataFrame:
    """目标 vs 现有持仓求差，产出订单；按风控规则过滤。

    halted=True 时只放行卖单。回撤触发熔断说明信号可能整体失效，此时继续按信号
    加仓是在往失效的方向追加暴露。
    """
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

    if halted:
        dropped = (orders["side"] == "BUY").sum()
        orders = orders[orders["side"] == "SELL"]
        if dropped:
            print(f"  [熔断] 回撤超限，已丢弃 {dropped} 笔买单，仅保留卖出")

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


# --------------------------------------------------------------------------- 名称


_NAMES_CACHE = ROOT / "state/names.json"
_SINA_HQ = "http://hq.sinajs.cn/list={codes}"


def stock_names(codes, cache_path: Path | None = None) -> dict[str, str]:
    """代码 -> 中文简称。订单清单是给人看的，光有代码核对起来太容易看错行。

    名称几乎不变，所以命中缓存的不再请求；ST 加帽摘帽这类改名确实会发生，
    删掉 state/names.json 即可重建。名称只是显示用，取不到不该拖垮出单，
    因此网络失败只警告并留空。
    """
    path = cache_path or _NAMES_CACHE
    cache: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
    missing = sorted({c for c in codes if c not in cache})

    for i in range(0, len(missing), 50):
        batch = missing[i : i + 50]
        try:
            # Referer 必须带：新浪对该接口做了防盗链，缺了就返回空串
            resp = requests.get(
                _SINA_HQ.format(codes=",".join(c.lower() for c in batch)),
                headers={"User-Agent": "Mozilla/5.0", "Referer": "http://finance.sina.com.cn"},
                timeout=20,
            )
            resp.encoding = "gbk"
            for line in resp.text.splitlines():
                m = re.match(r'var hq_str_(\w+)="([^",]*)', line.strip())
                if m and m.group(2):
                    cache[m.group(1).upper()] = m.group(2)
        except Exception as exc:  # noqa: BLE001
            print(f"  [警告] 股票名称获取失败，将留空: {exc}", file=sys.stderr)
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True))
    return {c: cache.get(c, "") for c in codes}


def with_names(orders: pd.DataFrame) -> pd.DataFrame:
    """在 code 右边插入 name 列。"""
    if orders.empty:
        return orders.assign(name="")[["code", "name", "side", "shares", "price", "value"]]
    out = orders.copy()
    out.insert(1, "name", out["code"].map(stock_names(list(out["code"]))))
    return out


def rank_table(
    scores: pd.Series,
    prices: pd.Series,
    adv: pd.Series,
    target: dict[str, int],
    cfg: dict,
) -> pd.DataFrame:
    """全池按模型打分倒排，并标注每只为何进/没进目标组合。

    订单清单是按金额排的，看不出模型的信心次序 —— 排第一行的往往只是股价低、
    一手便宜、凑得出最大单笔金额的那只。真正的高分票常因为一手买不起被跳过，
    这在只看订单时是隐形的，必须单独出一张表才看得见。
    """
    acct, trd = cfg["account"], cfg["trading"]
    lot, cap_per_stock = trd["lot_size"], acct["capital"] * acct["max_weight_per_stock"]
    eligible, affordable = screen(scores, prices, adv, cfg)
    eligible, affordable = set(eligible), list(affordable)
    ranked = scores.sort_values(ascending=False)
    names = stock_names(list(ranked.index))

    rows = []
    for i, (code, score) in enumerate(ranked.items(), 1):
        px = float(prices.get(code, np.nan))
        lot_cost = px * lot if np.isfinite(px) else np.nan
        shares = target.get(code, 0)
        if shares:
            status = "入选"
        elif not np.isfinite(px):
            status = "无报价"
        elif code not in eligible:
            status = "流动性未达标"
        elif lot_cost > cap_per_stock:
            # 一手就超过单只上限：不是模型不看好，是这个账户规模买不起
            status = "一手超上限"
        elif code in affordable:
            status = "资金不足"
        else:
            status = "名额已满"
        rows.append(
            dict(
                rank=i,
                code=code,
                name=names.get(code, ""),
                score=round(float(score), 6),
                price=round(px, 3) if np.isfinite(px) else np.nan,
                lot_cost=round(lot_cost, 1) if np.isfinite(lot_cost) else np.nan,
                shares=shares,
                status=status,
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 状态


def load_positions(path: Path) -> dict[str, int]:
    """只要股数。旧格式是 {code: shares}，新格式带成本价，两者都读得了。"""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: int(v["shares"] if isinstance(v, dict) else v) for k, v in raw.items()}


def load_cost_basis(path: Path) -> dict[str, float]:
    """建仓成本价。旧格式没有这个信息，返回空表示无法计算浮亏。"""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: float(v["cost"]) for k, v in raw.items() if isinstance(v, dict) and v.get("cost")}


def save_positions(
    path: Path,
    positions: dict[str, int],
    prices: pd.Series | None = None,
    previous: dict[str, int] | None = None,
    prev_cost: dict[str, float] | None = None,
) -> None:
    """写回持仓并维护成本价。

    加仓时成本按股数加权平均，减仓时成本不变（卖出不改变剩余股份的持有成本）。
    没有成本价就算不出单只浮亏，也就没法在报告里提示需要人工看基本面的票。
    """
    previous, prev_cost = previous or {}, prev_cost or {}
    out: dict[str, dict] = {}
    for code, shares in sorted(positions.items()):
        if not shares:
            continue
        entry = {"shares": int(shares)}
        old_shares, old_cost = previous.get(code, 0), prev_cost.get(code)
        px = float(prices[code]) if prices is not None and code in prices.index else None
        if shares > old_shares and px is not None:
            added = shares - old_shares
            entry["cost"] = round((old_shares * old_cost + added * px) / shares if old_cost else px, 4)
        elif old_cost:
            entry["cost"] = round(old_cost, 4)
        elif px is not None:
            entry["cost"] = round(px, 4)
        out[code] = entry
    path.write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- 回撤


def record_equity(path: Path, date: str, equity: float) -> tuple[float, float]:
    """追加净值并返回 (峰值, 当前回撤)。

    回撤要算在组合层面而不是个股：这是截面策略，个股跌 20% 而指数跌 25% 其实是赢的，
    真正该防的是信号整体失效，那表现为组合净值持续走低。
    """
    rows = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["date", "equity"])
    rows = rows[rows["date"] != date]
    rows = pd.concat([rows, pd.DataFrame([{"date": date, "equity": equity}])], ignore_index=True)
    rows = rows.sort_values("date").reset_index(drop=True)
    rows.to_csv(path, index=False)
    peak = float(rows["equity"].cummax().iloc[-1])
    return peak, 0.0 if peak <= 0 else 1.0 - equity / peak


def losing_positions(
    positions: dict[str, int], cost: dict[str, float], prices: pd.Series, threshold: float
) -> list[tuple[str, float]]:
    """浮亏超过阈值的持仓。只提示不自动卖 —— 退市风险、财务暴雷这类事件模型看不到，
    人能看到，该由人判断。"""
    out = []
    for code in positions:
        basis, px = cost.get(code), prices.get(code)
        if basis and px and np.isfinite(px):
            ret = px / basis - 1.0
            if ret <= -threshold:
                out.append((code, ret))
    return sorted(out, key=lambda x: x[1])

"""每日运行入口：刷新因子 -> 出预测 -> 生成订单清单。

    python production/run_daily.py                 # 全流程
    python production/run_daily.py --skip-signal   # 复用上次预测，只重算组合和订单
    python production/run_daily.py --risk-only     # 只看净值/回撤，不出订单
    python production/run_daily.py --date 2026-08-21

产出三份：orders/YYYY-MM-DD.csv（按订单金额）、signals/YYYY-MM-DD.csv（全池按模型打分
倒排，含未入选原因），以及对应的 reports/YYYY-MM-DD.md 与 reports/YYYY-MM-DD-signal.md。
不下单。
确认订单后自行执行，再用 --confirm 把持仓状态写回。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from production.pipeline import (  # noqa: E402
    build_target,
    compute_factors,
    estimate_cost,
    generate_orders,
    load_config,
    load_cost_basis,
    load_positions,
    rank_table,
    losing_positions,
    record_equity,
    save_positions,
    with_names,
)


def refresh_signal(cfg: dict, factors_path: Path, as_of: str) -> Path:
    """跑 qlib 训练+预测，返回 pred.pkl 路径。

    直接复用 qrun 与回测同一条代码路径，而不是另写一份推理代码 —— 两份实现迟早会漂移，
    而这里任何偏差都会让 holdout 验证出来的数字不再适用。
    """
    from rdagent.utils.env import QTDockerEnv
    from rdagent.utils.qlib import ALPHA20

    sig, uni = cfg["signal"], cfg["universe"]
    src = REPO / "rdagent/scenarios/qlib/experiment/factor_template"
    work = Path(tempfile.mkdtemp())
    for f in src.glob("*"):
        if f.is_file():
            shutil.copy(f, work)
    shutil.copy(factors_path, work / "combined_factors_df.parquet")
    shutil.copy(ROOT / "models/model.py", work / "model.py")

    env = {
        "PYTHONPATH": "./",
        "train_start": sig["train_start"],
        "train_end": sig["train_end"],
        "valid_start": sig["valid_start"],
        "valid_end": sig["valid_end"],
        "test_start": sig["valid_end"],
        # 用数据实际末端而非 2099：qlib 回测循环会走到日历尽头再取下一步，
        # 越界抛的 IndexError 出现在预测产出之后（不影响 pred.pkl），但它会在日志里
        # 留下一条假故障，掩盖真正需要注意的报错。
        "test_end": as_of,
        "label_horizon": str(sig["label_horizon"]),
        "feature_names": str(list(ALPHA20.keys())),
        "feature_expressions": str(list(ALPHA20.values())),
        "num_features": "31",
        "num_timesteps": "20",
        "step_len": "20",
        "dataset_cls": "TSDatasetH",
        # 固定种子：不设的话每次刷新都是一套新的随机初始化，两次信号的差异里
        # 混着重训噪声，分不清哪部分是新数据带来的。
        "seed": str(sig.get("seed", 42)),
        "n_epochs": "50",
        "lr": "0.0005",
        "early_stop": "10",
        "batch_size": "1024",
        "weight_decay": "1e-4",
        "topk": str(cfg["account"]["max_positions"]),
        "n_drop": "1",
        "MLFLOW_ALLOW_FILE_STORE": "true",
    }
    qt = QTDockerEnv()
    qt.prepare()
    qt.check_output(local_path=str(work), entry="qrun conf_combined_factors_sota_model.yaml", env=env)
    preds = sorted(work.glob("mlruns/*/*/artifacts/pred.pkl"), key=lambda p: p.stat().st_mtime)
    if not preds:
        raise RuntimeError("qrun 未产出 pred.pkl，检查容器日志")
    return preds[-1]


_SNAPSHOT_SRC = """
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, qlib
from qlib.constant import REG_CN
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN,
          expression_cache=None, dataset_cache=None)
from qlib.data import D
market, as_of, win, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
start = (pd.Timestamp(as_of) - pd.Timedelta(days=win * 3)).strftime("%Y-%m-%d")
df = D.features(D.instruments(market),
                ["$close", "$factor", "Mean($close*$volume, %d)" % win],
                start_time=start, end_time=as_of, freq="day")
df.columns = ["close", "factor", "adv"]
last = df.groupby(level="instrument").last()
pd.DataFrame({"price": last["close"] / last["factor"], "adv": last["adv"]}).dropna().to_pickle(out)
"""


def market_snapshot(cfg: dict, as_of: str) -> tuple[pd.Series, pd.Series]:
    """返回 (真实价, 滚动成交额)。

    $close 是复权价，下单要用真实价 $close/$factor —— 用复权价会把一手成本算错数倍。

    qlib 只装在 rdagent4qlib 里，所以这段在子进程跑，避免整个流程被绑死在某个
    conda 环境下。QLIB_PYTHON 可覆盖解释器路径。
    """
    win = cfg["universe"]["liquidity_window"]
    work = Path(tempfile.mkdtemp())
    script, out = work / "snap.py", work / "snap.pkl"
    script.write_text(_SNAPSHOT_SRC)
    python = os.environ.get("QLIB_PYTHON", "/root/miniconda3/envs/rdagent4qlib/bin/python")
    proc = subprocess.run(
        [python, str(script), cfg["universe"]["market"], as_of, str(win), str(out)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if not out.exists():
        raise RuntimeError(f"行情快照失败（{python}）:\n{(proc.stderr or '')[-500:]}")
    snap = pd.read_pickle(out)
    return snap["price"], snap["adv"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="截止日期，默认用数据最新日")
    ap.add_argument("--skip-signal", action="store_true", help="复用上次预测")
    ap.add_argument("--confirm", action="store_true", help="把今日订单计入持仓状态")
    ap.add_argument(
        "--risk-only",
        action="store_true",
        help="只看净值/回撤/浮亏，不出订单（信号未到刷新日时用）",
    )
    args = ap.parse_args()
    if args.risk_only and args.confirm:
        print("--risk-only 不产生订单，不能和 --confirm 同用", file=sys.stderr)
        return 1

    cfg = load_config()
    h5 = REPO / "rdagent/scenarios/qlib/experiment/factor_data_template/daily_pv_all.h5"
    if not h5.exists():
        print("缺少行情数据，先跑 generate_data_folder_from_qlib()", file=sys.stderr)
        return 1

    as_of = args.date or pd.read_hdf(h5, key="data").index.get_level_values("datetime").max().strftime("%Y-%m-%d")
    print(f"截止日期 {as_of}")

    pred_cache = ROOT / "state/pred.pkl"
    if not args.risk_only:
        if args.skip_signal and pred_cache.exists():
            print("复用已有预测")
        else:
            print("计算因子…")
            factors = ROOT / "state/combined_factors_df.parquet"
            compute_factors(h5, factors)
            print("训练并预测…（数分钟）")
            shutil.copy(refresh_signal(cfg, factors, as_of), pred_cache)

        pred = pd.read_pickle(pred_cache)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        latest = pred.index.get_level_values("datetime").max()
        scores = pred.xs(latest, level="datetime")
        print(f"信号日期 {latest.date()}，覆盖 {len(scores)} 只")

    prices, adv = market_snapshot(cfg, as_of)
    pos_path = ROOT / "state/positions.json"
    current = load_positions(pos_path)
    cost_basis = load_cost_basis(pos_path)

    # 组合回撤：持仓市值 + 现金。没有持仓时按满仓资金计，避免空仓被当成回撤。
    held_value = sum(float(prices.get(c, 0)) * s for c, s in current.items())
    equity = held_value + max(0.0, cfg["account"]["capital"] - held_value) if current else cfg["account"]["capital"]
    peak, drawdown = record_equity(ROOT / "state/equity.csv", as_of, equity)
    halted = drawdown >= cfg["risk"]["halt_on_drawdown"]
    if halted:
        print(f"  [熔断] 回撤 {drawdown:.1%} >= {cfg['risk']['halt_on_drawdown']:.0%}，本日仅允许卖出")

    losers = losing_positions(current, cost_basis, prices, 0.30)
    for code, ret in losers:
        print(f"  [关注] {code} 浮亏 {ret:.1%}，建议人工核查基本面")

    if args.risk_only:
        # 信号未到刷新日时的每日一看：净值和回撤只需要持仓和当日价格，不需要预测。
        # 刻意不出订单 —— 同一份打分下，仅价格漂移就能让整手数和筛选结果变化，
        # 实测这种"零新信息"的日间换手可达 15%，执行它只是在付手续费。
        print(f"\n持仓 {len(current)} 只 | 市值 {held_value:,.0f} 元")
        print(f"净值 {equity:,.0f} 元 | 峰值 {peak:,.0f} | 回撤 {drawdown:.1%}"
              + (f"（已超 {cfg['risk']['halt_on_drawdown']:.0%} 熔断线）" if halted else ""))
        print("未出订单（--risk-only）。到刷新日再跑完整流程。")
        return 0

    target = build_target(scores, prices, adv, cfg)
    orders = with_names(generate_orders(current, target, prices, cfg, halted=halted))
    cost = estimate_cost(orders, cfg)

    stamp = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    if not orders.empty:
        orders.to_csv(ROOT / f"orders/{stamp}.csv", index=False)

    held = sum(prices.get(c, 0) * s for c, s in target.items())
    lines = [
        f"# 交易计划 {stamp}",
        "",
        f"- 信号日期：{latest.date()}",
        f"- 目标持仓：{len(target)} 只，市值 {held:,.0f} 元" f"（占资金 {held / cfg['account']['capital']:.1%}）",
        f"- 订单：{len(orders)} 笔"
        f"（买 {(orders['side'] == 'BUY').sum() if not orders.empty else 0}，"
        f"卖 {(orders['side'] == 'SELL').sum() if not orders.empty else 0}）",
        f"- 成交额：{orders['value'].sum() if not orders.empty else 0:,.0f} 元",
        f"- 预估成本：{cost:,.0f} 元" f"（{cost / cfg['account']['capital']:.3%}）",
        f"- 组合净值：{equity:,.0f} 元 | 峰值 {peak:,.0f} | 回撤 {drawdown:.1%}"
        + ("  **已熔断，仅卖出**" if halted else ""),
        "",
        *(["## 需人工核查（浮亏 >30%）", ""] + [f"- {c} {r:.1%}" for c, r in losers] + [""] if losers else []),
        "## 订单",
        "",
        orders.to_markdown(index=False) if not orders.empty else "无",
        "",
        "> 本清单不自动下单。人工确认执行后，用 --confirm 更新持仓状态。",
    ]
    (ROOT / f"reports/{stamp}.md").write_text("\n".join(lines))

    # 第二张表：按模型打分倒排。订单表按金额排，第一行往往只是"一手最便宜、
    # 凑得出最大单笔"的那只，跟模型的信心次序无关，两者必须分开看。
    ranks = rank_table(scores, prices, adv, target, cfg)
    ranks.to_csv(ROOT / f"signals/{stamp}.csv", index=False)

    top = ranks.head(cfg["account"]["max_positions"] + 10)
    skipped = int((ranks.head(10)["status"] == "一手超上限").sum())
    picked = ranks[ranks["shares"] > 0]
    sig_lines = [
        f"# 信号排名 {stamp}",
        "",
        f"- 信号日期：{latest.date()}，全池 {len(ranks)} 只",
        f"- 流动性达标 {(ranks['status'] != '流动性未达标').sum()} 只"
        f"（前 {cfg['universe']['liquidity_pct']:.0%} 成交额）",
        f"- 入选 {len(picked)} 只，打分排名中位 {picked['rank'].median():.0f}"
        if len(picked)
        else "- 入选 0 只",
        f"- 前 10 名中有 {skipped} 只因一手超过单只上限（"
        f"{cfg['account']['capital'] * cfg['account']['max_weight_per_stock']:,.0f} 元）买不起",
        "",
        f"## 前 {len(top)} 名",
        "",
        top.to_markdown(index=False),
        "",
        "> 完整 300 只见 signals/" + stamp + ".csv。status 含义：入选 / 一手超上限 /",
        "> 流动性未达标 / 名额已满 / 资金不足 / 无报价。",
    ]
    (ROOT / f"reports/{stamp}-signal.md").write_text("\n".join(sig_lines))

    print(f"\n目标 {len(target)} 只 | 订单 {len(orders)} 笔 | 预估成本 {cost:,.0f} 元")
    print(f"报告 production/reports/{stamp}.md（按金额）")
    print(f"     production/reports/{stamp}-signal.md（按模型打分）")

    if args.confirm:
        # 熔断时目标不等于实际结果：买单没下，持仓只减不增
        settled = {c: s for c, s in target.items() if not halted or s <= current.get(c, 0)}
        if halted:
            settled = {c: min(s, current.get(c, 0)) for c, s in target.items() if current.get(c, 0)}
        save_positions(pos_path, settled, prices, current, cost_basis)
        print("持仓状态已更新")
    elif not orders.empty:
        print("确认执行后加 --confirm 更新持仓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

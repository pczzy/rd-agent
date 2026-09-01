"""记录限价单实际成交情况，校准成本参数。

限价单的显性成本（佣金+过户费）只有约 0.076%，但它有两项回测没建模的隐性成本：

  成交率      挂单不保证成交，未成交的部分让实际组合偏离目标
  逆向选择    你想买的票涨上去了买不到，砸下来的才成交 —— 成交的那批被市场
              选择过，系统性偏向短期走弱的股票。这部分不出现在任何手续费里。

用法：

    # 1) 下单后生成待填模板（已预填挂单信息）
    python production/record_fills.py --template 2026-09-01

    # 2) 收盘后把 filled_shares / fill_price 填进 fills/2026-09-01.csv

    # 3) 累计够几天后出报告
    python production/record_fills.py --analyze

跑两周（约 10 个交易日）就够把 config.yaml 的 cost_one_side 校准到位。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from production.pipeline import load_config  # noqa: E402

FILL_COLUMNS = [
    "code",
    "side",
    "order_shares",
    "order_price",  # 由 orders/ 预填
    "filled_shares",
    "fill_price",  # 人工填写
]

# 后续收益要在装了 qlib 的环境里取
_RETURNS_SRC = """
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd, qlib
from qlib.constant import REG_CN
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN,
          expression_cache=None, dataset_cache=None)
from qlib.data import D
market, start, end, horizon, out = sys.argv[1:6]
df = D.features(D.instruments(market),
                ["Ref($close, -%s)/$close - 1" % horizon],
                start_time=start, end_time=end, freq="day")
df.columns = ["fwd_ret"]
df.swaplevel().sort_index().to_pickle(out)
"""


def make_template(date: str) -> Path:
    orders_path = ROOT / f"orders/{date}.csv"
    if not orders_path.exists():
        raise SystemExit(f"找不到 {orders_path}，先跑 run_daily.py")

    orders = pd.read_csv(orders_path)
    tmpl = pd.DataFrame(
        {
            "code": orders["code"],
            "side": orders["side"],
            "order_shares": orders["shares"],
            "order_price": orders["price"],
            "filled_shares": "",  # 全未成交填 0
            "fill_price": "",  # 未成交留空
        }
    )
    out = ROOT / "fills" / f"{date}.csv"
    out.parent.mkdir(exist_ok=True)
    tmpl.to_csv(out, index=False)
    return out


def _forward_returns(market: str, start: str, end: str, horizon: int) -> pd.Series:
    work = Path(tempfile.mkdtemp())
    script, out = work / "r.py", work / "r.pkl"
    script.write_text(_RETURNS_SRC)
    python = os.environ.get("QLIB_PYTHON", "/root/miniconda3/envs/rdagent4qlib/bin/python")
    proc = subprocess.run(
        [python, str(script), market, start, end, str(horizon), str(out)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if not out.exists():
        raise RuntimeError(f"取后续收益失败:\n{(proc.stderr or '')[-400:]}")
    return pd.read_pickle(out)["fwd_ret"]


def analyse(horizon: int) -> None:
    cfg = load_config()
    files = sorted((ROOT / "fills").glob("*.csv"))
    if not files:
        raise SystemExit("fills/ 下没有记录，先用 --template 生成并填写")

    frames = []
    for path in files:
        day = pd.read_csv(path)
        missing = [c for c in FILL_COLUMNS if c not in day.columns]
        if missing:
            print(f"  [跳过] {path.name} 缺列: {missing}")
            continue
        day = day[day["filled_shares"].notna()]
        if day.empty:
            print(f"  [跳过] {path.name} 尚未填写")
            continue
        day["date"] = pd.Timestamp(path.stem)
        frames.append(day)

    if not frames:
        raise SystemExit("没有已填写的记录")

    df = pd.concat(frames, ignore_index=True)
    df["filled_shares"] = pd.to_numeric(df["filled_shares"], errors="coerce").fillna(0)
    df["fill_price"] = pd.to_numeric(df["fill_price"], errors="coerce")
    df["filled"] = df["filled_shares"] > 0

    print(f"记录 {len(files)} 天，{len(df)} 笔订单\n")

    # --- 成交率 -----------------------------------------------------------
    by_count = df["filled"].mean()
    by_value = (df.loc[df["filled"], "filled_shares"] * df.loc[df["filled"], "order_price"]).sum() / (
        df["order_shares"] * df["order_price"]
    ).sum()
    print("=== 成交率 ===")
    print(f"  按笔数 {by_count:.1%} | 按金额 {by_value:.1%}")

    # --- 实际滑点 ---------------------------------------------------------
    hit = df[df["filled"] & df["fill_price"].notna()].copy()
    if not hit.empty:
        # 买入成交价高于挂单价为不利，卖出反之
        sign = np.where(hit["side"] == "BUY", 1.0, -1.0)
        hit["slippage"] = sign * (hit["fill_price"] - hit["order_price"]) / hit["order_price"]
        print("\n=== 实际滑点（相对挂单价，正=不利）===")
        print(f"  均值 {hit['slippage'].mean():+.4%} | 中位 {hit['slippage'].median():+.4%}")

    # --- 逆向选择 ---------------------------------------------------------
    start = df["date"].min().strftime("%Y-%m-%d")
    end = (df["date"].max() + pd.Timedelta(days=horizon * 3)).strftime("%Y-%m-%d")
    fwd = _forward_returns(cfg["universe"]["market"], start, end, horizon)
    df["fwd_ret"] = [fwd.get((d, c), np.nan) for d, c in zip(df["date"], df["code"])]

    all_buys = df[df["side"] == "BUY"]
    buys = all_buys[all_buys["fwd_ret"].notna()]
    print(f"\n=== 逆向选择（买单，{horizon} 日后收益）===")
    if len(all_buys) and buys.empty:
        # 最近 horizon 天的订单还没有未来收益可算，等数据补齐再看
        print(
            f"  {len(all_buys)} 笔买单都还取不到 {horizon} 日后收益"
            f"（下单日距数据末端不足 {horizon} 个交易日），等行情更新后重跑"
        )
    elif buys["filled"].nunique() < 2:
        print(f"  可比样本 {len(buys)} 笔，但买单全部成交或全部未成交，无法比较")
    else:
        got = buys.loc[buys["filled"], "fwd_ret"].mean()
        miss = buys.loc[~buys["filled"], "fwd_ret"].mean()
        print(f"  成交的   {got:+.3%}  (n={buys['filled'].sum()})")
        print(f"  未成交的 {miss:+.3%}  (n={(~buys['filled']).sum()})")
        print(f"  差额     {got - miss:+.3%}  ← 为负说明买到的是走弱的那批")

    # --- 成本建议 ---------------------------------------------------------
    trd = cfg["trading"]
    explicit = 0.00075 + 0.00001  # 佣金 + 过户费
    slip = hit["slippage"].mean() if not hit.empty else 0.0
    adverse = 0.0
    if not buys.empty and buys["filled"].nunique() == 2:
        adverse = max(0.0, -(got - miss)) / horizon * 10  # 折算到 10 日调仓周期
    suggested = explicit + max(slip, 0.0) + adverse
    print("\n=== 成本参数建议 ===")
    print(f"  显性(佣金+过户)  {explicit:.4%}")
    print(f"  实际滑点         {max(slip, 0.0):.4%}")
    print(f"  逆向选择折算     {adverse:.4%}")
    print(f"  → cost_one_side  {suggested:.5f}   (当前配置 {trd['cost_one_side']})")
    if len(files) < 10:
        print(f"\n  注意：仅 {len(files)} 天样本，建议累计 10 天以上再回填配置")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", metavar="DATE", help="按该日订单生成待填模板")
    ap.add_argument("--analyze", action="store_true", help="汇总已填记录")
    ap.add_argument("--horizon", type=int, default=5, help="逆向选择的观察天数")
    args = ap.parse_args()

    if args.template:
        print(f"已生成 {make_template(args.template)}")
        print("填写 filled_shares 和 fill_price 后运行 --analyze（未成交填 0）")
    elif args.analyze:
        analyse(args.horizon)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

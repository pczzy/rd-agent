#!/usr/bin/env bash
# 手工重训：补数据 -> 算因子 -> 训练并预测 -> 出订单清单。不会动持仓。
#
#   production/train.sh                 截止到数据最新日
#   production/train.sh --date 2026-09-24
#
# 训练窗口按截止日自动后延（config.yaml 的 train_start / valid_years，见 pipeline.rolling_segments）。
# 其余参数原样传给 run_daily.py；--skip-signal 与本脚本的目的相反，拒掉。
# 订单确认后再单独跑 run_daily.py --skip-signal --confirm 计入持仓。

set -uo pipefail
export http_proxy=''
export https_proxy=''

REPO=/root/RD-Agent
PY=/root/miniconda3/envs/rdagent/bin/python
LOGDIR="$REPO/production/logs"
LOG="$LOGDIR/train-$(date +%F-%H%M%S).log"
# 和 auto_update.sh 的锁分开：这里第一步就会调它，同一把锁会自己把自己挡在门外。
LOCK=/run/lock/rdagent-train.lock

for a in "$@"; do
    if [ "$a" = "--skip-signal" ] || [ "$a" = "--risk-only" ]; then
        echo "train.sh 就是用来重训的，不接受 $a；要复用旧预测直接跑 run_daily.py" >&2
        exit 1
    fi
done

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1   # rdagent 的配置从 cwd 的 .env 读

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "已有一次训练在运行" >&2
    exit 1
fi

exec > >(tee -a "$LOG") 2>&1
echo "===== $(date '+%F %T') 开始，日志 $LOG ====="

# 先把行情和 h5 补齐。已是最新时它几秒内就 SKIP 退出；它的细节写在自己的日志里。
echo "[1/2] 补数据"
if ! "$REPO/production/auto_update.sh"; then
    echo "补数据失败，见 $LOGDIR/update.log" >&2
    exit 1
fi
tail -n1 "$LOGDIR/update.log"

echo "[2/2] 训练并预测"
if ! "$PY" production/run_daily.py "$@"; then
    echo "===== $(date '+%F %T') 失败 =====" >&2
    exit 1
fi
echo "===== $(date '+%F %T') 完成 ====="

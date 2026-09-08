#!/usr/bin/env bash
# 每个交易日收盘后自动补数据。由 crontab 在周一至周五 18:00 触发。
#
# 做两件事，顺序不能反：
#   1) update_data.py                    补日历、顺延成分表、写各股 .bin
#   2) generate_data_folder_from_qlib()  重生成 daily_pv_all.h5（因子源数据，跑在 docker 里）
# 只做第一步的话，run_daily.py 读的 h5 还停在旧日期上 —— 症状是"行情更新了但信号不动"。
#
# 节假日不必特判：休市日新浪返回不出新的交易日，update_data.py 打印"数据已是最新"后
# 0 退出。周末由 crontab 的 1-5 挡掉（A 股调休也不在周末开市）。

set -uo pipefail

export HOME=/root
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

REPO=/root/RD-Agent
# rdagent 环境：两步都能跑。rdagent4qlib 有 qlib 却缺 fuzzywuzzy，import rdagent 就炸；
# 而第二步的 qlib 跑在 docker 里，本地根本不需要 qlib。
PY=/root/miniconda3/envs/rdagent/bin/python
CAL=$HOME/.qlib/qlib_data/cn_data/calendars/day.txt
LOGDIR="$REPO/production/logs"
SUMMARY="$LOGDIR/update.log"
LOG="$LOGDIR/$(date +%F).log"
# 上一次成功重生成 h5 时的日历末日。用它而不是"本次日历有没有变长"来决定要不要重生成：
# 若某天第一步成功、第二步失败，只看增量的话第二天会以为无事可做，h5 就永远停在旧日期。
SYNCED="$LOGDIR/.h5_synced_until"
LOCK=/run/lock/rdagent-update.lock

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1   # rdagent 的配置从 cwd 的 .env 读

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$SUMMARY"; }

# 自锁：手动执行和 cron 撞上时，后来者直接退出而不是并发写同一批 .bin
exec 9>"$LOCK"
if ! flock -n 9; then
    say "SKIP  上一次更新仍在运行"
    exit 0
fi

exec >>"$LOG" 2>&1
echo "===== $(date '+%F %T') 开始 ====="

before=$(tail -n1 "$CAL")

if ! timeout 1800 "$PY" production/update_data.py; then
    say "FAIL  update_data.py 失败，见 $LOG"
    exit 1
fi

after=$(tail -n1 "$CAL")
synced=$(cat "$SYNCED" 2>/dev/null || echo "-")

if [ "$synced" = "$after" ]; then
    echo "h5 已对齐 $after，跳过重生成"
    say "SKIP  无新交易日，数据仍为 $after"
    exit 0
fi

if ! timeout 3600 "$PY" -c \
    "from rdagent.scenarios.qlib.experiment.utils import generate_data_folder_from_qlib as g; g()"; then
    say "FAIL  h5 重生成失败（行情已补至 $after，h5 仍停在 $synced），见 $LOG"
    exit 1
fi

echo "$after" > "$SYNCED"
say "OK    $before -> $after，h5 已刷新"

find "$LOGDIR" -name '20*-*-*.log' -mtime +60 -delete

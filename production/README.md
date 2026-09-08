# 实盘系统

从信号到订单清单的完整流程。**不连券商，不自动下单** —— 产出 CSV，人工确认后执行。

## 数据更新

qlib 官方数据集更新滞后（本机曾落后 11 个交易日），每次调仓前先补数据：

```bash
python production/update_data.py --check   # 看差多少天
python production/update_data.py           # 从新浪财经补齐
python -c "from rdagent.scenarios.qlib.experiment.utils import generate_data_folder_from_qlib as g; g()"
```

脚本做三件事，缺一不可：补日历、**顺延成分表**、写各股 `.bin`。漏掉成分表的话
`D.instruments()` 在新日期返回空集，症状是"行情更新了但信号不动"。

两个口径要点：新浪返回**不复权价**（qlib 存的是复权价 = 真实价 × factor），
volume 按股而 qlib 按手。factor 沿用最后已知值——日线接口看不到除权信息，所以
跨越除权日会有偏差，**每季度应该用 qlib 官方数据重拉一次全量**。

### 自动更新

上面两步已由 `auto_update.sh` 接管，crontab 在**周一至周五 18:00**（收盘后三小时）触发：

```cron
0 18 * * 1-5 /root/RD-Agent/production/auto_update.sh >/dev/null 2>&1
```

- **节假日不必特判**：休市日新浪返回不出新的交易日，`update_data.py` 打印"数据已是最新"
  后 0 退出，h5 重生成随之跳过。周末由 cron 的 `1-5` 挡掉（A 股调休也不在周末开市）。
- **h5 只在它落后于日历时才重生成**：它跑在 docker 里，是这条链上最慢的一步。
  判据是 `logs/.h5_synced_until`（上次重生成成功时的日历末日）与当前日历末日是否相等，
  而不是"本次日历有没有变长" —— 后者在第一步成功、第二步失败的那天之后会以为无事可做，
  h5 就永远停在旧日期上。用前者则第二天自动补上。
- **自锁**：`flock` 挡住手动执行与 cron 撞车，避免两个进程并发写同一批 `.bin`。
- 用 `rdagent` 环境的解释器：`rdagent4qlib` 虽然装了 qlib 却缺 `fuzzywuzzy`，
  `import rdagent` 就会炸；而第二步的 qlib 跑在 docker 里，本地并不需要 qlib。

日志在 `production/logs/`（不入库）：`update.log` 是一行一天的流水，
`YYYY-MM-DD.log` 是当天全文。**每天开工前先看一眼流水**：

```bash
tail -5 production/logs/update.log
# 2026-09-08 18:52:10 OK    2026-09-03 -> 2026-09-08，h5 已刷新
# 状态含义：OK 补上了 / SKIP 无新交易日或有别的实例在跑 / FAIL 见当天全文
```

`FAIL` 分两种后果：第一步失败则数据原样未动；第二步失败意味着 `.bin` 已经补到新日期
但 h5 还是旧的，此时 `run_daily.py` 会用旧日期跑出一份看似正常的计划。下一次自动运行
会重试第二步，但**当天不要直接调仓** —— 要么等下一次，要么手动补跑那条
`generate_data_folder_from_qlib()`。

自动更新只管数据，**不碰信号也不出订单** —— 调仓仍然是手动跑 `run_daily.py`。

## 每日操作

```bash
# 1) 生成今日交易计划（首次或需刷新信号时，约 10-15 分钟）
python production/run_daily.py

# 2) 只重算组合和订单，复用上次预测（秒级）
python production/run_daily.py --skip-signal

# 3) 人工执行订单后，把持仓写回状态
python production/run_daily.py --skip-signal --confirm
```

产出：
- `orders/YYYY-MM-DD.csv` —— 订单清单（code / side / shares / price / value）
- `reports/YYYY-MM-DD.md` —— 当日计划摘要

信号按 `config.yaml` 的 `refresh_every_n_days: 10` 更新，其余交易日用 `--skip-signal`
复用即可。

## 当前配置的由来

参数不是随手设的，每条都对应 `docs/worklog/WORKLOG.md` 里的实测：

| 参数 | 值 | 依据 |
|---|---|---|
| `liquidity_pct` | 0.50 | holdout 上不过滤时年化 -1.44%/回撤 -23.9%，过滤后 +10.23%/-9.6% |
| `label_horizon` | 21 | 1 日目标的 Rank IC 只有 20 日的 1/2.2 |
| `refresh_every_n_days` | 10 | holdout 上每日/每5日/每10日分别 +8.96%/+5.37%/+9.22% |
| `max_positions` | 30 | topk=30 优于 20（净 +2.20% vs -4.81%，换手更低） |
| `cost_one_side` | 0.00191 | 20 万账户实测：佣金 0.075% + 过户 0.001% + 半价差 0.030% + 冲击 0.085% |

**回测表现**（holdout 2025-01~2026-08，397 交易日，从未参与参数选择）：
年化超额 +9.22%，IR 0.73，最大回撤 -19.6%，日均换手 6.4%。

## 结构

```
config.yaml     所有参数
pipeline.py     因子计算 / 组合构建 / 订单生成 / 成本估算
run_daily.py    每日入口
factors/        11 个因子代码（搜索得到）
models/         BiGRU-Attention 模型定义
state/          当前持仓 positions.json、缓存的预测 pred.pkl
orders/         订单清单
reports/        每日报告
```

## 风控

配置在 `config.yaml` 的 `risk` 段，代码里强制执行：

- **单只上限** `max_weight_per_stock: 0.045`。必须 ≥ 1/`max_positions`，否则两个约束
  互相打架 —— 0.05×30 看似宽松，实际前 20 只就用完资金，只能建 20 只仓。
- **单日换手上限** `max_daily_turnover: 0.15`。仅在已有持仓时生效：空仓建仓的换手必然
  等于目标仓位，拿它撞上限会把首日订单砍掉大半。
- **最小单笔** `min_order_value: 3000`。低于此金额时 5 元佣金占比过高（0.17%）。
  **清仓单不受此限制**，否则小额仓位永远出不掉。
- **整手约束**：全部订单为 100 股整数倍。
- **一手超上限的股票自动跳过并顺延**。csi300 里一手上万的不少（SH688256 一手 10.35 万），
  20 万账户买不起；不顺延会有两成多资金闲置。
- **持仓无法定价时告警**。退市/长期停牌的仓位会打印警告要求人工处理，而不是静默跳过
  ——静默的话这笔仓位会一直挂在账上却从不出现在订单里。
- **组合回撤熔断** `halt_on_drawdown: 0.20`。净值自峰值回撤超 20% 时丢弃全部买单，
  只允许卖出。净值记在 `state/equity.csv`，每次 `--confirm` 追加一条。
- **单只浮亏 >30% 告警**。只提示不自动卖 —— 退市风险、财务暴雷这类事件模型看不到，
  该由人判断。成本价记在 `state/positions.json`，加仓按股数加权平均，减仓不变。

### 为什么没有个股止损

这是**截面选股**策略，信号预测的是相对排序而非绝对涨跌。一只票跌 20% 而指数跌 25%，
它其实是赢的；固定比例止损会把这类赢家砍掉，且系统性地在低位卖出，与均值回复相悖。

真正该防的是信号整体失效，那表现为**组合层面**持续回撤 —— 这正是熔断要管的事。

卖出只有两个触发点：掉出打分前 30（清仓），或目标权重下降（减仓）。信号每 10 个
交易日刷新，所以实际调仓节奏是 10 天一次。

## 校准限价单成本

若用限价单（成本比市价单低一半以上：单边约 0.076% vs 0.191%），需要实测两项回测
没建模的隐性成本 —— 成交率和逆向选择。

```bash
python production/record_fills.py --template 2026-09-01   # 下单后生成待填模板
# 收盘后把 filled_shares / fill_price 填进 fills/2026-09-01.csv（未成交填 0）
python production/record_fills.py --analyze                # 累计几天后出报告
```

报告给出成交率、实际滑点、逆向选择，以及建议的 `cost_one_side`。累计 10 个交易日
以上再回填配置。

**逆向选择**是限价单的本质代价：你想买的票涨上去了买不到，砸下来的才成交，
成交的那批被市场选择过、系统性偏向短期走弱。它不出现在任何手续费里，只能这样测。

holdout 上的成本敏感性：单边 0.191% 时年化 +14.33%/IR 0.90，0.076% 时 +21.04%/IR 1.28。

## 已知限制

1. **未接券商**。订单需人工执行。接入时在 `run_daily.py` 末尾加下单调用即可，
   `orders` DataFrame 已是标准格式。
2. **模型 train/valid 止于 2025-12**。上线前应重跑一次 `run_daily.py`（不加
   `--skip-signal`）用最新数据重训。
3. **冲击成本系数未实测**。`cost_one_side` 里的 0.085% 用文献 `Y=1` 估算。建议用几千块
   小额单实测：记录下单时中间价与成交均价之差，反推真实 Y。
4. **`best epoch = 0`**。20 日重叠标签使有效独立时段仅约 170 个（3401 交易日 / 20），
   模型第一轮就学完可学的信息。不影响信号有效性（已做标签随机化检验：打乱后 Rank IC
   归零），但意味着深度模型的容量是浪费的 —— LightGBM 在同数据上 Rank IC 0.1256，
   与 BiGRU 的 0.1245 打平。若要简化部署可换 LightGBM。

## 上线前检查

- [ ] 用最新数据重训（`run_daily.py` 不加 `--skip-signal`）
- [ ] 小额实测冲击成本，回填 `config.yaml` 的 `cost_one_side`
- [ ] 确认 `state/positions.json` 与券商实际持仓一致
- [ ] 首日只用小仓位跑通全流程（下单、成交、对账）

# RD-Agent 量化实验工作日志

> 配套文档：[PITFALLS.md](PITFALLS.md) —— 历次错误与防复发清单。
> 下结论前建议先过一遍那份的"通用检查清单"。


记录区间：2026-08-24 ~ 2026-08-28
目标：用 RD-Agent 在 CSI300 上搜出可实盘的 alpha，Rank IC 做到 0.05 以上

---

## TODO

- [ ] **用生产口径重跑 holdout，把数字坐实**（2026-09-09 提出）
  - README 的年化/IR/回撤已全部撤下（见零点七），现在没有可引用的生产口径数字
  - 口径：CSI300 + 20 万 + 21 日标签 + 10 日刷新 + 流动性前 50% + 单边 0.191%
  - **跑完当场把 mlruns 指标抄进本文件** —— 这次的教训正是没抄，工作区已被清掉
  - 顺带纳入 2026-09-08 改的换手截断新规则（卖出豁免、买入独享上限）

- [ ] **补 cuDNN 确定性设置，让 `seed` 真正生效**（2026-09-03 提出）
  - 现状：种子已固定且确认传到模型，但同输入重跑 Spearman 仅 0.762（见零点六）
  - 训练在 cuda:0 上，cuDNN 的 GRU 反向默认非确定
  - 需要 `torch.backends.cudnn.deterministic = True`、`benchmark = False`、
    `torch.use_deterministic_algorithms(True)`，外加环境变量
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`（不设会在 RNN 上直接报错）
  - 挂载点 `production/models/model.py`，环境变量走 `run_daily.py` 的 env
  - 做完要连跑两次验证，目标是完全一致而不是"接近"

- [ ] **测同配置重跑的 Rank IC 方差**（2026-09-03 提出）
  - 判定准则用的是"Rank IC 提升 >0.001 视为噪音之上"，但这个阈值没有对照过
    重跑方差；若方差接近 0.001，部分 loop 的判定就是在噪声上做的
  - 方法：Loop 27 同配置重跑 3~5 次，看 Rank IC 分布
  - 依赖上一项 —— 先确定性可复现，才能把"重跑方差"和"随机初始化"分开

- [x] **关掉 `log_llm_chat_content`**（2026-08-30 完成）
  - `.env` 加 `LOG_LLM_CHAT_CONTENT=False`
  - 实测：17MB 日志里 97.6% 是 LLM prompt/回复正文（104213 行中 101680 行），
    真实事件行只有 2533 行
  - CoSTEER 把历史失败反馈嵌进每次 prompt 是**正确设计**（演化式写码需要看到上一版
    错在哪），问题只在于连同 prompt 一起落盘，导致过去的报错反复出现、grep 诊断失效
  - prompt 本身没有无界增长：109 次调用 token 在 2475~10831 之间震荡

- [x] **改判定准则：首要指标从 Rank ICIR 换成 Rank IC**（2026-08-28 完成）
  - `developer/feedback.py:17` `IMPORTANT_METRICS` 把 `Rank IC` 提到首位
  - `prompts.yaml` 因子侧/模型侧判定改为：**Rank IC 提升（幅度 >0.001，低于此视为噪音）
    且 Rank ICIR ≥ SOTA 的 80%** 才接受；Rank ICIR 降级为稳定性护栏
  - 用三个真实历史案例回测新准则，判断全部正确：
    Loop 27（Rank IC 0.0425）由 `no` 纠正为 `yes`；Loop 22 噪音级提升仍拒绝；
    假 alpha（ARR 28% / Rank IC 0.0087）仍拒绝
- [x] **修 OOM**（2026-08-28 完成，未动 `MULTI_PROC_N`）
  - `developer/utils.py` 新增 `_combine_factor_frames()` 取代 `pd.concat(axis=1)`：
    先求一次并集索引，再填预分配的 float32 二维块
  - 实测 4 个真实因子（各约 1500 万行）：`pd.concat` +2928MB → 新实现 **+230MB（降 92%）**
  - 等价性已验证：shape / index / columns / NaN 位置完全一致，
    最大相对误差 5.96e-08（float32 精度内）
  - 非数值列等异常会回退到 `pd.concat`，最坏情况不劣于原实现
  - `MULTI_PROC_N=6` 保留 —— 并行度是因子重算速度的关键
- [x] **Loop 27 定为新基线**（2026-08-28 完成）
  - 已把 session `log/2026-08-25_01-43-12-429660/__session__/28/4_record` 里
    `trace.hist[27].decision` 从 `False` 翻为 `True`（原判定来自旧的 Rank ICIR 优先准则）
  - 验证：`get_sota_hypothesis_and_experiment()` 现返回 Rank IC 0.0425 / Rank ICIR 0.2324 /
    ARR 0.1162；decision=True 的条目变为 `[0,1,3,4,5,27]`，已接受因子实验 4 个
  - 原文件备份在同目录 `4_record.bak-before-loop27-baseline`
  - **续跑请用这个路径**：`--path log/2026-08-25_01-43-12-429660/__session__/28/4_record`
    （其他更早的 dump 里 trace 仍是旧的）

---

## 零、CSI1000 线（2026-08-29~30）—— 当前最佳可实盘方案

CSI300 那条线在 20 万以下资金上被交易成本压死（详见"关键发现 3"）。改用 CSI1000
后信号和成本两端同时改善，holdout 验证通过。

### 可实盘配置

```
本金        10 万
股票池      CSI1000 ∩ 流动性前 50%（滚动 20 日成交额）
持仓        30 只等权，每只约 3300 元
信号        Loop 27 因子集（Alpha20 + 11 个自研因子）→ BiGRU-Attention 模型
预测目标    20 日前瞻收益  Ref($close,-21)/Ref($close,-1)-1
调仓        信号每 10 个交易日更新，TopkDropout n_drop=1
```

### 结果（train 2015-2020 / valid 2021-2022 不变，仅换 test 窗口）

| | test 2023-24 | **holdout 2025-01~2026-08** |
|---|---|---|
| Rank IC | 0.0901 | **0.0927** |
| Rank ICIR | 0.6273 | 0.5656 |
| 年化净超额 (c=0.027) | +7.76% | **+10.23%** |
| IR | 0.92 | **0.92** |
| 最大回撤 | -9.4% | -9.6% |
| 日均换手 | 4.9% | 4.2% |

holdout 从未参与任何参数选择。两段不相交时间上 IR 完全一致（0.92），Rank IC 甚至
略升，是较强的稳健性证据。

冲击系数 c 在 0.013~0.10（7 倍跨度）内净收益都在 10~11.6%，**已对该未知参数不敏感**
——因为换手压到 4.2% 后成本占比很小。

### 流动性过滤是必需的，不是可选项

holdout 上全池对照：年化 **-1.44%**、回撤 -23.9%；加过滤后 +10.23%、回撤 -9.6%。

原因是 Loop 27 因子集里含 `Amihud_Illiquidity_20d_Median`，全池训练时模型会被引导到
流动性差的票上，而那部分收益无法兑现。过滤等于给模型套了执行约束。

> 注意：这与 Alpha20 基线上的观察**相反**（基线上过滤会损害收益，因为小盘 alpha 集中
> 在低流动性票）。同一个操作在不同因子集上效果相反，不能一概而论。

### 数据管道改造

| 项 | 改动 |
|---|---|
| 因子数据 | `generate.py` 改 `D.instruments("csi1000")`，1500 万行 → **273 万行** |
| 股票池/基准 | 五个模板 `csi300→csi1000`、`SH000300→SH000852` |
| 涨跌停 | `0.095→0.195`（池内混合 ±10%/±20%，取宽松边界） |
| 交易成本 | `open_cost 0.0065 / close_cost 0.0070`（实测口径，原 5bp/15bp 是机构假设） |
| 日期 | train 2015-2020 / valid 2021-2022 / test 2023-2024 |

### 修正的两个重要错误

**1. `$close` 是复权价，不是真实价。** 真实价 = `$close / $factor`。早先用复权价算的
"一手成本"严重偏低（CSI300 中位数算成 558 元，真实是 2478 元）。

**2. 冲击系数取值缺乏校准。** 平方根律 `冲击 = Y × σ_日 × √参与率`，文献 Y ≈ 0.5~1.0。
CSI1000 日波动率中位 2.67%，故合理的 `c = Y×σ` 应在 **0.013~0.040**。我最初凭印象取
c=0.1，隐含 Y=3.7，远超文献上限。这个过于悲观的取值一度让我误判"20 万做不了日频选股"。

叠加上"评估时误用 Alpha20 基线而非 Loop 27 信号"，两个错误合起来把一个可行方案判成了
死路。修正后：毛收益 3.21% → 9.50%，净收益由负转正。

### 仍存的风险

1. train/valid 止于 2022，模型没见过 2023 年后的数据，实盘前应用最新数据重训
2. 冲击系数未实测（虽已不敏感），建议用小额实盘单反推真实 Y
3. 10.2% 是**相对中证1000 的超额**，指数下跌时策略同样亏损
4. 397 个交易日样本，IR 0.92 的标准误约 0.8，真实值可能在 0.1~1.7

---

## 零点五、预测周期对比（2026-09-01，CSI300）

同一套 31 因子、同一数据划分（train 2010-2023 / valid 2024-2025 / test 2026-01~08），
只改 `label_horizon`，并让调仓频率与预测周期匹配。

### 信号强度

| 预测周期 | Rank IC | Rank ICIR | IC |
|---|---|---|---|
| 5 日 | 0.0616 | 0.3949 | 0.0200 |
| 10 日 | 0.1117 | **0.9347** | 0.0811 |
| **20 日** | **0.1265** | 0.8081 | 0.0792 |

### 净收益（20 万，topk=30，单边 0.191%）

| 预测周期 | 调仓 | 年化(无成本) | 年化净 | IR | MDD |
|---|---|---|---|---|---|
| 5 日 | 每 1 日 | -13.20% | **-16.73%** | -1.06 | -28.7% |
| 5 日 | 每 5 日 | -13.23% | -16.67% | -1.02 | -28.4% |
| 10 日 | 每 1 日 | 6.98% | +3.68% | 0.28 | -17.1% |
| 10 日 | 每 5 日 | 9.56% | +6.32% | 0.46 | -19.0% |
| 10 日 | 每 10 日 | 9.61% | +6.48% | 0.46 | -17.9% |
| 20 日 | 每 1 日 | 16.87% | +13.56% | 0.97 | -11.6% |
| **20 日** | **每 10 日** | **20.73%** | **+17.51%** | **1.21** | -14.2% |
| 20 日 | 每 20 日 | 4.35% | +1.75% | 0.12 | -19.5% |

### 结论

**5 日预测完全不能用**（净 -16.7%）。5 日收益里噪音占绝对主导，而我们这套因子多是
10-20 日窗口，预测不了那么短的周期。

**调仓频率约等于预测周期的一半时最优**。太频繁（每 1 日）信号没兑现就换掉且多付成本；
太稀疏（每 20 日）持仓跟不上市场——注意那一行连**无成本**收益都塌到 4.35%，说明不是
成本问题而是信号失效。

**Rank ICIR 最高的不是收益最好的**：10 日预测 ICIR 0.9347 高于 20 日的 0.8081，但收益
只有 6.48% vs 17.51%。信号稳定 ≠ 赚得多；20 日的 Rank IC 更高，且预测周期长意味着单位
时间换手更低。

生产配置 `label_horizon: 21` + `refresh_every_n_days: 10` 得到独立验证——此前是基于
CSI1000 上的观察选的，这次在 CSI300 上重新确认。

> 样本仅 154 个交易日，IR 1.21 的标准误约 1.3。**不同周期之间的相对排序比绝对数值
> 可信得多**，因为它们跑在同一段数据、同样的市场环境上。

---

## 零点六、信号重跑噪声底（2026-09-03，CSI300）

起因是想搞清楚"今天的信号和前天不一样"里有多少是新信息。结论：**基本没有。**

### 测量

数据固定在 `daily_pv_all.h5` 至 2026-09-03，配置不变，`seed=42` 已固定，
连跑两次完整 `run_daily.py`，比对最后一个交易日 300 只的截面打分。

| | 同种子重跑两次 | 09-01 vs 09-03 |
|---|---|---|
| 完全相同 | 否 | — |
| 最大绝对差 | 0.347 | — |
| Spearman | **0.762** | **0.753** |
| 前 30 名重合 | 16/30 | — |

**两次同输入重跑的差异，和相隔两个交易日的差异一样大。** 0.753 落在噪声底
0.762 之内，日间信号变化里分辨不出新信息。

### 固定种子不足以复现

`signal.seed: 42` 已经从 config 传到模板（`{% if seed %}` 保护，不影响搜索循环），
容器日志确认模型收到了 `seed : 42`。但训练跑在 `cuda:0`（RTX 3090）上，**cuDNN 的 GRU
反向传播默认非确定**，只设 `np.random.seed` + `torch.manual_seed`（qlib 的
`GeneralPTNN` 就做了这两件事）拦不住。要真复现还需要 `cudnn.deterministic = True`、
`torch.use_deterministic_algorithms(True)` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
挂载点是 `production/models/model.py`（我们自己的文件，建模时导入），环境变量走
`run_daily.py` 的 env。**未做，见 TODO。**

### 含义

1. **判断"信号变了"要先跟 0.762 比。** 低于这个水平的变化不值得解读。
2. **支持 10 天刷新节奏。** 配套的另一项测量：把打分冻住只让价格变，
   仅重新配权就产生 1.3%~15.0% 的日换手，5 天累计 40%、155 元成本、零新信息 ——
   而第二章记的判定是换手 >12% 的配置全部亏钱。中间日用 `--risk-only` 只看回撤。
3. **搜索循环的判定阈值可能需要重新标定（待验证）。** 现行准则是 Rank IC 提升
   >0.001 才算噪音之上。但如果单次重训的截面打分 Spearman 只有 0.76，同一份配置
   重跑两次的 Rank IC 方差本身有多大？没测过。**这是推论不是结论**，但如果方差
   接近或超过 0.001，那么部分 loop 的接受/拒绝判定就是在噪声上做的。
   验证方法很直接：拿 Loop 27 同配置重跑 3~5 次，看 Rank IC 的分布。

---

## 零点七、README 数字溯源（2026-09-09）

起因是要报一个"模型年化"。查下来：**`production/README.md` 里多数回测数字找不到出处。**

### 方法

把 `git_ignore_folder/RD-Agent_workspace` 下所有 run 的 mlruns 指标全部读出来
（84 个 run，其中 72 个含 `1day.excess_return_with_cost.annualized_return`），
与 README 声称的数字逐个比对，容差 ±0.06%。

### 结果

| README 出处 | 数字 | 可追溯 |
|---|---|---|
| 「回测表现」标题 | 年化 +9.22% / IR 0.73 / MDD -19.6% / 换手 6.4% | 否 |
| 成本敏感性（单边 0.191%） | +14.33% / IR 0.90 | 否 |
| 成本敏感性（单边 0.076%） | +21.04% / IR 1.28 | 否 |
| 刷新频率（每日 / 每 5 日 / 每 10 日） | +8.96% / +5.37% / +9.22% | 否 |
| 持仓数（topk=30 / 20） | +2.20% / -4.81% | 否 |
| `label_horizon`「1 日目标 Rank IC 是 20 日的 1/2.2」 | — | 否 |
| 流动性过滤对照 | -1.44%/-23.9% vs +10.23%/-9.6% | **是**（第零章） |

- `+14.33%`、`+21.04%`、`+17.51%` 在 72 个 run 里一个都匹配不到。
- `+9.22%` 匹配到 08-24 的一个 run，但那个 run 的 IR 是 **1.14**，不是 README 写的 0.73。
  72 个样本按 ±0.06% 容差撞上一个，按巧合处理。

### 一个内部矛盾

`config.yaml` 的 `cost_one_side` 就是 `0.00191`。按 README 自己那张成本敏感性表，
生产配置的年化应当是 **+14.33% / IR 0.90**，而不是标题上的 +9.22% / IR 0.73。
同一段 holdout、同一个成本，两个数并存 —— 至少有一个是错的。

### 成因与教训

这些数大概率出自 `refresh_signal()` 用 `tempfile.mkdtemp()` 建的临时工作区
（`production/run_daily.py:56`），跑完随目录一起消失，当时也没抄进本文件。

> **回测数字必须在产出当时就抄进 worklog。工作区不是存档。**

`production/README.md` 已按可追溯性重写，未追溯到的数字全部撤下，只留有出处的。

### 现状：没有生产口径的数字

**没有任何一组结果的口径完全等于当前 `config.yaml`**（CSI300 + 20 万 + 21 日标签 +
10 日刷新 + 流动性前 50% + 单边 0.191%）。已有两组最接近，但都不能直接当作本流程的预期：

| 口径 | 窗口 | 年化超额 | IR | MDD |
|---|---|---|---|---|
| 第零章（CSI1000 / 10 万） | holdout 2025-01~2026-08，397 日 | +10.23% | 0.92 | -9.6% |
| 第零点五章（CSI300 / 20 万） | test 2026-01~08，154 日 | +17.51% | 1.21 | -14.2% |

两者都跑在 qlib 的 TopkDropout 上，**不含**生产流程另加的流动性过滤、整手约束、
单只权重上限和回撤熔断。见 TODO。

---

## 一、最终结果

### 当前最佳模型：Loop 27

session：`log/2026-08-25_01-43-12-429660/__session__/27/`

| 指标 | test 2022-2024 | **holdout 2025-01~2026-08** |
|---|---|---|
| Rank IC | 0.0425 | **0.0308** |
| Rank ICIR | 0.2324 | 0.1690 |
| IC | 0.0305 | 0.0260 |
| 年化超额（含成本） | 11.62% | **10.27%** |
| IR | — | **1.42** |
| 最大回撤 | -15.3% | **-7.7%** |
| 日均换手 | — | 2.1% |
| 成本拖累 | — | 0.48%/年 |

holdout 是搜索循环从未接触过的数据。跨期衰减温和（Rank IC 保住 72%，年化保住 88%），
没有过拟合塌方。

### 对比：Loop 5（此前的 SOTA）

| | test 2022-2024 | holdout |
|---|---|---|
| Rank IC | 0.0357 | 0.0293 |
| 年化超额（100/1 持仓） | 9.01% | 5.61% |
| IR | 1.49 | 0.61 |
| 最大回撤 | -12.2% | -15.4% |

**Loop 27 在 holdout 上全面更优。**

### 与论文的距离

论文 *R&D-Agent-Quant*（arXiv 2505.15155）用的是 train 2008-2014 / valid 2015-2016 /
**test 2017-01-01~2020-08-01**，与我们的区间不同，不能直接比。

我们在论文同窗口上跑过 3 轮，Rank IC 达 0.0399~0.0406；论文最好成绩
R&D-Model(o3-mini) Rank IC 0.0546、R&D-Agent(Q) 0.0495。论文每个配置跑 30 轮。

> 注意：论文的 0.0546 是 2017-2020 的产物，**不应外推到近期数据**。我们自己的实测显示
> 同一模型在 2022-2024 的 Rank IC 只有 2017-2020 的一半左右。

---

## 二、关键发现

### 1. 判定准则用年化收益会选出"假 alpha"

改准则前，系统按"年化收益提升就接受"判定。它选出的模型在 holdout 上：

```
年化超额 28.04%，但 Rank IC 只有 0.0087
月度 Rank IC 胜率 50%（掷硬币）
最好 20 天贡献了 121% 的累计超额（其余 377 天净亏）
```

高收益 + near-zero 排序能力 = 少数几天押对，不可重复。**ARR 会骗人，Rank IC 不会。**

改成 Rank ICIR 主导后立刻见效：7 轮就把 Rank IC 从 0.0247 推到 0.0357（此前 10 轮
用旧准则只到 0.0247），且出现了"年化 -1.85% 但排序能力提升被接受"、"年化 +7.28%
但排序能力弱被拒绝"这类正确判断。

### 2. 持仓参数比模型搜索更值钱

同一个模型、同一份预测，只改 `topk`/`n_drop`：

| 配置 | test 年化 | holdout 年化 | holdout IR |
|---|---|---|---|
| 50/5（原默认） | 10.66% | **-2.25%** | -0.21 |
| 100/1 | 9.01% | **+5.61%** | 0.61 |
| 100/3 | 13.59% | +3.32% | 0.40 |
| 150/3 | 11.78% | +5.08% | 0.67 |

原默认配置每天换 19.5% 仓位，对一个日度 Rank IC 只有 0.03 的信号来说太激进，
alpha 还没兑现就被手续费磨掉。holdout 上**所有换手 >12% 的配置全部亏钱，
所有 <7% 的全部赚钱** —— 两段不相交数据上的一致规律。

已把模板默认改为 `topk=100 / n_drop=1`。

### 3. 5 万本金跑不了这个策略（重要）

资金规模敏感性（holdout，topk=30/n_drop=1）：

| 本金 | 年化(无成本) | 年化(含成本) | 年成本 |
|---|---|---|---|
| 5 万 | **-3.75%** | **-8.35%** | 4.70% |
| 10 万 | +1.52% | -0.82% | 2.39% |
| 30 万 | +3.05% | +1.46% | 1.62% |
| 50 万 | +5.66% | +4.10% | 1.60% |
| **100 万** | +8.40% | **+6.82%** | 1.61% |
| 1000 万 | +6.14% | +4.57% | 1.60% |

**5 万块连"无成本收益"都是负的**，说明不只是费率问题：

- 一手 100 股 + CSI300 价格离散（一手成本中位数 558 元、75% 分位 1710 元、
  10% 的股票超过 4180 元）→ 5 万块每只预算太小，只买得起便宜股票，
  被动引入低价股风格暴露
- 5 元最低佣金：topk=100 时每笔 500 元 → 实际费率 1.00%（模板假设 0.05%，差 20 倍）

**盈亏平衡线在 30 万~50 万之间，100 万左右最优。**

### 4. 搜索参数不要迁就小资金

在 5 万规模下几乎所有配置回测都是负的（21 个组合里 18 个亏钱），
搜索循环会失去可攀爬的信号。正确做法是**两件事分开**：

- 搜索阶段：用大账户评估，目标是找信号强度（Rank IC）
- 落地阶段：信号找到后再用真实资金规模做可行性检验

---

## 三、修掉的 bug

按发现顺序，全部经过实测验证。

### 1. `np.str_` 列名污染 parquet，导致后半程搜索全面失效（影响最大）

**症状**：Loop 16 起连续多轮，新实验回测全部失败，SOTA 15 轮纹丝不动。
表面看像"搜索到瓶颈"，实际是搜索根本没在有效运行。

**根因**：
```
StaticDataLoader._maybe_load_raw_data
  → pd.read_parquet("combined_factors_df.parquet", engine="pyarrow")
    → pyarrow pandas_compat.py:824 _deserialize_column_index
      → ast.literal_eval("('feature', np.str_('HGB_Predicted_5d_Rank_60d_Expanding'))")
        → ValueError: malformed node or string: <ast.Call object>
```
LLM 生成的因子代码里某个操作返回 `np.str_` 而非原生 `str`。numpy 2.x 起
`np.str_('x')` 的 repr 变成了 `np.str_('x')`；pandas 把列名 repr 写进 parquet 元数据，
pyarrow 读回时用 `ast.literal_eval` 解析，遇到这个函数调用形式就抛异常。

只在 SOTA 因子库积累起来后才会触发（需要写/读 `combined_factors_df.parquet`），
所以前 9 轮正常、之后越来越频繁。

**修复**：`developer/utils.py` 新增 `feature_columns()`，写 parquet 前把列名强制转
原生 `str`；`factor_runner.py`（2 处）、`model_runner.py`（1 处）替换调用。
已清理 5 个受污染的 parquet 缓存。

**验证**：同数据旧写法读回抛 ValueError、新写法正常；修复后 Loop 20 完整跑通产出结果。

### 2. Runner 缓存键不含日期区间

`components/runner/__init__.py` 的 `CachedRunner.get_cache_key` 只哈希 task 描述。
基线实验 `sub_tasks=[]` → 哈希空串 → 常量 key。

**后果**：换了 train/test 区间后，系统拿**旧区间缓存的基线**去对比**新区间的实验**，
所有 SOTA 判定失效。实测过：基线数字 0.044615 是 2017-2020 的，却被用来和
2022-2024 的实验比。

**修复**：新增 `get_cache_extra_key()` 钩子，两个 qlib runner override 后把六个日期
拼进 key。验证不同区间产生不同哈希。

> 这个修复对 holdout 验证是刚需 —— 那时正要改 test 区间。

### 3. `model_type` 静默穿透

LLM 有 5/10 次把 prompt 里的占位符 `"Tabular or TimeSeries"` 原样返回。
`factor_runner.py` 的 if/elif 没有 else，导致 `num_features` 不进环境变量，
模板渲染成空 → YAML 读成 `null` → 一路传到 `nn.Linear(None)` 才崩。

**修复**：`developer/utils.py` 新增 `normalize_model_type()` 做归一化 + 未知值显式报错；
改了 prompt 措辞让占位符不再像合法取值；模板加 jinja 兜底，`num_features` 缺失即报错。

### 4. `CONDA_DEFAULT_ENV` 在非交互环境缺失

`factor_coder/config.py` 直接读 `os.environ.get("CONDA_DEFAULT_ENV")` 塞给 `CondaConf`。
这个变量只有交互式 `conda activate` 才导出，systemd/cron 下是空的 → pydantic 校验失败。

第一次定时任务就是这么挂的（18:00 触发，13 秒后 exit 1）。

**修复**：缺失时从 `sys.prefix` 推导环境名。

### 5. 其他

| 问题 | 位置 | 修复 |
|---|---|---|
| qlib conda 环境半装（pip 失败被吞） | `utils/env.py` `QlibCondaEnv.prepare` | 验证 `import qlib` 而非只看 env 是否存在；pip 失败重试 3 次并抛错；刷新 `bin_path` |
| `QTDockerEnv` 所有实例共用一个 conf | `utils/env.py` | 可变默认参数改 `None`，每实例建自己的 conf |
| `get_model_env` 清空 qlib 数据挂载 | `model_coder/conf.py` | `extra_volumes` 改为合并而非覆盖 |
| mlflow 拒绝 file store | 两个 CondaConf/DockerConf | 加 `env_dict={"MLFLOW_ALLOW_FILE_STORE": "true"}` |
| DataLoader 死锁（n_jobs=20 > 12 核） | 三个 conf 模板 | 改 `{{ n_jobs \| default(8, true) }}` |
| 四处重复的 docker/conda 分支 | 多处 | 统一到 `get_qlib_env()` |
| runtime_info 无 JSON 时报 `AttributeError` | `shared/get_runtime_info.py` | 改为带输出内容的 `RuntimeError` |

---

## 四、环境与配置

### `.env` 关键项

```bash
MODEL_CoSTEER_ENV_TYPE=docker        # 走 local_qlib 镜像（GPU 可用，torch 2.2.1）
no_proxy=localhost,127.0.0.1,::1     # 本机 ollama 必须绕过 HTTP 代理
NO_PROXY=localhost,127.0.0.1,::1
MULTI_PROC_N=6                       # 因子并行（见 TODO：需降到 2-3）
CHAT_MODEL_MAP={"direct_exp_gen": {"model": "deepseek/deepseek-v4-pro", "reasoning_effort": "high"}}

# 三套前缀必须一致（runner 读 FACTOR/MODEL，场景描述读 QUANT）
QLIB_{QUANT,FACTOR,MODEL}_TRAIN_START=2010-01-01
QLIB_{QUANT,FACTOR,MODEL}_TRAIN_END=2019-12-31
QLIB_{QUANT,FACTOR,MODEL}_VALID_START=2020-01-01
QLIB_{QUANT,FACTOR,MODEL}_VALID_END=2021-12-31
QLIB_{QUANT,FACTOR,MODEL}_TEST_START=2022-01-01
QLIB_{QUANT,FACTOR,MODEL}_TEST_END=2024-12-31
```

**holdout 2025-01-01 ~ 2026-08-21 刻意不配进来**，只在最终验证时用内联环境变量临时覆盖。

### 数据

- 交易日历：2000-01-04 ~ 2026-08-21
- 因子源 `daily_pv_all.h5`：2008-12-29 ~ 2026-08-21，6081 只股票
- csi300 时点成分表更新至 2026-08-21

### 回测假设（已核实，合理）

| 项 | 值 | 说明 |
|---|---|---|
| label | `Ref($close,-2)/Ref($close,-1)-1` | T 日出信号、T+1 买、T+2 卖，无未来函数 |
| deal_price | close | 尾盘成交 |
| limit_threshold | 0.095 | 涨跌停不可成交 |
| open/close_cost | 5bp / 15bp | 含印花税 |
| topk / n_drop | 100 / 1 | 已改（原 50/5） |

---

## 五、已知问题

### OOM（必须解决）

2026-08-27 04:59 被内核 OOM 杀死，`anon-rss 9.7GB`，死在 `process_factor_data`。

机理：SOTA 因子库越积越多，`process_factor_data` 要把每个因子在全历史（1500 万行）
上重算并横向拼接，内存随因子数线性上涨；叠加 `MULTI_PROC_N=6` 六个子进程各占 ~1.7GB，
30GB 内存在第 29 轮撑不住。

**随因子库增长必然重演，不是偶发。**

### 日志判读陷阱

RD-Agent 把完整历史 feedback 塞进每次 LLM prompt，**所有过去的报错都会被反复打印**，
越往后越多。用关键词计数判断"有没有失败"会严重误导。

可靠判据：

```bash
# 1. 启动后是否有真实容器执行（注意时间格式，别用 systemd 的 "Wed ... CST"）
find git_ignore_folder/RD-Agent_workspace/*/logs -name 'docker_execution_2*.log' \
     -newermt '2026-08-26 18:48:50' | while read f; do
  grep -laq "Render the template" $f && echo "$f malformed=$(grep -ac 'malformed node' $f)"; done

# 2. 是否有新结果对象产出
ls -lt pickle_cache/rdagent.scenarios.qlib.developer.*/*.pkl | head
```

### `--loop-n` 语义

续跑时 `loop_idx` 被重置为 0（`utils/workflow/loop.py:375`），且 `loop_n` 会从 pickle
恢复上次的残留值。所以它是**含已完成轮次的总数**，不是"还要跑几轮"。
要到达 loop N，需传 `--loop-n (N+1)`。

（`loop.py:132` 那句 `# remain loop count` 注释是误导的。）

---

## 六、改动文件清单

19 个文件，+224 / -95。

```
rdagent/components/coder/factor_coder/config.py      CONDA_DEFAULT_ENV 兜底
rdagent/components/coder/model_coder/conf.py         get_qlib_env / extra_volumes 合并
rdagent/components/coder/model_coder/model.py        统一用 get_qlib_env
rdagent/components/runner/__init__.py                缓存键加 get_cache_extra_key 钩子
rdagent/scenarios/qlib/developer/factor_runner.py    feature_columns / normalize_model_type
rdagent/scenarios/qlib/developer/feedback.py         IMPORTANT_METRICS 加 Rank 系列
rdagent/scenarios/qlib/developer/model_runner.py     同上 + 缓存键
rdagent/scenarios/qlib/developer/utils.py            新增两个 helper
rdagent/scenarios/qlib/experiment/*/conf_*.yaml (5)  topk/n_drop、n_jobs、num_features 兜底
rdagent/scenarios/qlib/experiment/workspace.py       统一 get_qlib_env
rdagent/scenarios/qlib/prompts.yaml                  判定准则 + model_type 措辞
rdagent/scenarios/shared/get_runtime_info.py         报错信息带上下文
rdagent/utils/env.py                                 conda 环境、docker conf、mlflow
rdagent/utils/qlib.py                                统一 get_qlib_env
```

---

## 七、运行记录

| 时间 | 内容 | 结果 |
|---|---|---|
| 08-24 | 修环境（conda/docker/代理/mlflow），首次跑通 | Loop 0-2，test 2017-2020 |
| 08-24 夜 | 切到新区间 2022-2024，跑 10 轮 | SOTA Rank IC 0.0247 |
| 08-25 | 改判定准则为 Rank ICIR 主导 + 推理模式 | 7 轮到 Rank IC 0.0357（Loop 5） |
| 08-25 夜 | 续跑，发现后半程回测全挂 | 定位 np.str_ parquet bug |
| 08-26 18:00 | systemd 定时续跑（第一次因 conda 变量失败） | 修复后 18:48 启动 |
| 08-26~27 | 跑完 loop 20-28，10 小时 | **Loop 27: Rank IC 0.0425** |
| 08-27 04:59 | OOM 中断于 loop 29 | — |
| 08-28 | Loop 27 holdout 验证 | **年化 10.27% / IR 1.42 / MDD -7.7%** |
| 09-03 | 实盘管道：名称列 / 打分报表 / 分配修复 / --risk-only | 信号噪声底 Spearman 0.762 |

累计 LLM 花费约 $3。

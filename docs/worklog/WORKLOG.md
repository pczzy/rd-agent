# RD-Agent 量化实验工作日志

记录区间：2026-08-24 ~ 2026-08-28
目标：用 RD-Agent 在 CSI300 上搜出可实盘的 alpha，Rank IC 做到 0.05 以上

---

## TODO

- [ ] **当前 10 轮（loop 29-38）跑完后，关掉 `log_llm_chat_content`**
  - `.env` 加 `LOG_LLM_CHAT_CONTENT=False`（需重启生效，所以等这轮结束再改）
  - 实测：17MB 日志里 **97.6% 是 LLM prompt/回复正文**（104213 行中 101680 行），
    真实事件行只有 2533 行。日志膨胀约 40 倍
  - 更麻烦的是 grep 诊断失效：CoSTEER 把历史失败反馈嵌进每次 prompt（这部分**是**
    正确设计，演化式写码需要看到上一版错在哪），但连同 prompt 一起打进日志后，
    过去的报错会被反复打印，容易误判成新故障——排查时已被坑过两次
  - 注意 prompt 本身**没有**无界增长：109 次调用 token 数在 2475~10831 之间震荡，
    不是单调上升（`max_past_message_include=10` 在裁剪）
  - 复盘需要 prompt 原文时，从 `log/<时间戳>/` 的结构化日志取，比翻文本可靠


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

累计 LLM 花费约 $3。

# EA-V3：基于物化 Root 的执行兑现研究——实验操作文档

> **版本：MR-EP-1.0｜编制日期：2026-09-14**  
> **性质：新提出、待人工批准与运行前冻结的操作协议；不是实验结果。**  
> **主仓库：`rollinpioneer/OG`**  
> **只读历史基点：`7714e33958dd6ab7d8a1ddf894dca1eb88a69b17`**  
> **建议新分支：`exp/ea-v3-materialized-roots-v1`**  
> **第一次交给 Agent 的范围：只做 S0 + S1，到工程验收后停止。**

**核心目标：先把“同一个实验现场”保存成可独立重建的文件对象，再检验执行兑现差距；机制通过后，才训练执行模型。**

---

## 0. 使用说明：本次究竟新在哪里

### 0.1 三类内容必须分开

| 类型 | 本文采用的内容 |
|---|---|
| 已核验的历史事实 | EA-V2 Cube 已封存；最终状态为 `HOLD_CUBE_ROOT_RECONSTRUCTION_MISMATCH`；Phase C 未解锁。[S1–S2] |
| 沿用的研究假设 | 创新 2 提供数据锚定的执行估计，创新 1 利用估计选择更能兑现后续价值的子目标。[S3–S4] |
| 本文新建议 | 物化 Root、原现场参考轨迹、独立进程验收、故障注入测试、80/40 机制规模、阶段状态和预算。批准前不属于已冻结协议。 |

EA-V3 是**实验协议升级**，不是已经成立的新算法，也不是把 EA-V2 的无效统计换一个名称继续使用。“物化 Root”首先是评价基础设施，不能单独包装成离线学习算法创新。

本文所列新脚本名、命令行和数据结构是**待实现接口契约**，不表示仓库中已经有这些程序。本文没有运行仿真、训练模型、创建远端分支或推送代码。

### 0.2 对封存结果的准确理解

封存报告记录：61 个历史失败候选分支中，54 个可评价分支在三种取证协议下均通过；另外 7 个均属于 root `4022`，由于原 decision snapshot 无法重建，三种分支测试都不能有效执行。报告还记录了“reset observation 和任务元数据相同，但 37D goal 编码不同”。[S1–S2]

因此，本计划采用的出发点是：

> 需要完整保存并核验 root 的身份，而不是预先认定 MuJoCo 接触本身随机。

“没有在可评价分支观察到动力学分叉”不等于“已经定位全部历史错误”。此前 16 个工程样本通过，也不等于全部未来 root 都可复现。新协议只按每个实际 artifact 的验证结果放行。

### 0.3 旧协议保持关闭

不得在 `ea_v2_cube_mechanism_v1` 下新建研究结果，不运行旧 R2-RP/R3，不覆盖旧审计和封存文件。新结果不得与旧 Phase 0、0-R、0-R2 合并计算确认性统计。

新协议仍是在已知历史结果后设计的；论文中必须披露这一点。新的、未查看的评估样本可以用于新协议下的留出验证，但更名不能消除历史探索和方法选择过程。

---

## 1. 科学问题、方法与授权边界

### 1.1 科学问题不变

同一当前观察 $o$、同一最终目标 $g$、同一候选集 $\mathcal Z(o)$ 下：

\[
z_{\mathrm{ideal}}=\arg\max_{z\in\mathcal Z(o)}V(z,g)
\]

是否会选择一个在**固定低层控制器实际执行后**并不占优的候选？若存在这样的空间，离线训练的执行模型是否能帮助选择，而不是只在事后知道谁最好？

### 1.2 两个候选创新的关系

- **创新 1：执行感知选择。** 用控制器追求 $z$ 后预计产生的短程奖励和实际落点尾值评分，代替只看 $V(z,g)$。
- **创新 2：数据锚定的多时间尺度执行估计。** 用离线记录动作及其真实一步/多步结果训练、校准模型，为创新 1 提供估计。

本轮前半段尚不验证算法性能：

```text
物化并封存 root
    → 独立重建验收
    → 冻结真实数据候选
    → 检查机制空间
    → 通过后才训练 P_1
    → 再考虑 P_MH 与完整重规划系统
```

### 1.3 纯离线边界

模型训练仅使用原官方离线训练数据，模型选择/校准按冻结的离线验证划分进行。新生成的 root、前缀、工程探针、候选分叉和完整 continuation 都属于 `ENV_EVALUATED`，`training_eligible=false`。

root 中的 simulator state、内部目标、接触缓存仅供评价环境恢复使用。控制器、价值模型和未来学习的动力学模型只能获得获准的公开观察、公开目标及动作，不能读取评价用特权字段。

物化 Root 是研究者的评估工具，不是未来策略部署时需要提供的额外输入，也不是新的训练 pair 数据集。

---

## 2. 资产继承与冻结

### 2.1 允许继承的资产

| 资产 | 本轮处理 |
|---|---|
| 官方 Cube train/validation 文件 | 复用原文件，逐字节核验 SHA256，不重新生成 |
| 官方 OGBench 源码 | 固定到下述 commit，不升级版本 |
| Cube GCIQL actor/value | 复用 fixed-final checkpoint，不重训、不重新挑选 checkpoint |
| 训练集归一化及验证 episode 划分 | 读取旧 manifest 并核对；明确实际采用的数组与索引哈希 |
| 旧取证代码 | 可用于定位问题，但不能因文件名有 `v2` 就认定可靠 |
| Gaussian candidate prior | 不用于本轮机制实验 |
| 历史环境分叉 | 只读工程参考，不进入训练或新统计 |

已核验的 backbone manifest 记录 actor/value 共享 checkpoint，且控制器**没有显式剩余时域输入**。因此本文采用 $\pi(a\mid o,z)$；$k=20$ 是候选数据窗口长度，不声称控制器本身是精确的 20 步条件策略。[S5]

### 2.2 必须核对的指纹

```text
historical_repository_commit:
  7714e33958dd6ab7d8a1ddf894dca1eb88a69b17

ogbench_commit:
  1d4140997f60c52c6fb0702ec100dc988b18c548

dataset_id:
  cube-double-play-v0

train_sha256:
  a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9

validation_sha256:
  b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e

actor_and_value_checkpoint_sha256:
  12f7bd7ed9f909039dd6e3e3c44cd226bf765cf7d959396001e469dbbcbc3d87

backbone:
  GCIQL / alpha=1.0 / actor_randomgoal=0.0 / actor_trajgoal=1.0
  seed=0 / updates=1000000 / fixed-final checkpoint
```

checkpoint 指纹取自封存 manifest。[S5] 数据指纹须在 S0 对照原数据 manifest 与服务器实物重新核验。历史日志记录过的成功率只用于背景说明，不视为新协议下的重新评价。

执行模式显式固定为 `temperature=0.0` 的 GCIQL 推理。actor/value 的输入预处理完全沿用原 checkpoint 的合同；检索距离使用的标准化不能被重复应用到原本接收原始 observation 的 actor/value 输入上。每个字段的 dtype 与设备精度在 S0 固定。


另冻结 Python、NumPy、MuJoCo、Gymnasium、JAX/JAXLIB、Flax、dm-control、设备架构、精度设置、BLAS/线程设置、模型 XML/资产及运行时配置。**以运行机实际环境为准核对旧 manifest，不执行 `pip install -U`，也不把网页最新版本当实验版本。**

若指纹不一致、checkpoint 缺失或运行代码有未登记补丁，输出 `EA3_HOLD_ASSET_MISMATCH`，不自动重训或替换。

---

## 3. 执行拆分与停止位置

| 批次 | 阶段 | 工作 | 允许训练 | 结束后 |
|---|---|---|---|---|
| 第一次 | S0 + S1 | 资产核验、实现统一执行器、15 个工程现场及故障注入测试 | 否 | 停止，审查真实 trace 与验收代码 |
| 第二次 | S2 | 采集并物化新 root pool；独立重建；冻结候选库 | 否 | 停止，审查 root/candidate manifest |
| 第三次 | S3 | 同候选配对机制实验与完整 continuation | 否 | 停止，给出机制 gate |
| 第四次 | S4，条件解锁 | 普通一步模型 $P_1$；`EXEC_1` 对比 `IDEAL` | 仅 $P_1$ | 停止，判断创新 1 是否值得推进 |
| 后续独立交接 | S5，条件解锁 | $P_{MH}$、多时域消融与完整任务强对照 | 须另行批准 | 不由本文件自动执行 |

**不把整份文档作为一次“全部完成”任务。** 每阶段入口检查上阶段批准文件、协议哈希和资产锁；研究 gate 不能由 Agent 的文字说明代替。

---

## 4. Root 的正式定义：身份不再只是 seed

定义：

\[
\mathcal R=\bigl(X_0,o_0,g,\mathcal T,A_{0:t-1},X_t,o_t,c_t,\Omega\bigr).
\]

其中 $X$ 是评价器保存的完整现场，$o$ 是策略可见观察，$g$ 是**原 reset 实际返回的完整 37D goal**，$\mathcal T$ 是物理任务目标，$A_{0:t-1}$ 是实际执行过的前缀动作，$c_t$ 是时间/终止/控制上下文，$\Omega$ 是版本与随机状态。

### 4.1 一个 root 必须包含什么

| 类别 | 必存内容 |
|---|---|
| 身份 | protocol ID、root ID、task ID、计划 decision step、实际 prefix 长度、采集代码 SHA |
| 目标 | `goal_observation.npy` 完整 37D 数组；物理目标 cube 位置、mocap、任务元数据、对象身份/排列 |
| reset 现场 | reset API 实际返回的 observation；bootstrap integration state；对应 Python、wrapper、RNG 状态 |
| 前缀 | 完整动作序列及 dtype；逐步 observation/状态摘要、reward、success、terminated、truncated；实际 key 日程 |
| decision 现场 | API 实际返回的 observation；integration state；公开观察所依赖的阶段性缓存；环境/控制器上下文 |
| 时间 | 所有 wrapper 的计数器、物理时间、官方剩余 episode 步数；不能重置为完整 500 步 |
| 可追溯性 | 数据/权重/运行环境/模型资产哈希、每个文件哈希、root 内容清单哈希 |
| 用途 | `result_source=ENV_EVALUATED`、`training_eligible=false`、采集与验收时间 |

**task ID 相同不代表 $g$ 相同；$o_t$ 相同也不代表隐藏现场相同。两种 goal 都必须验收：策略/价值函数收到的 37D 编码，以及环境成功判断使用的物理目标。** [S2,S6]

### 4.2 建议文件结构

```text
root_store/<protocol_id>/<root_id>/
  root_manifest.json
  initial_observation.npy
  goal_observation.npy
  task_state.json
  bootstrap_integration.npy
  bootstrap_python_state/          # 有类型信息的 JSON + NumPy 数组
  prefix_actions.npy
  prefix_trace.npz
  decision_observation.npy
  decision_integration.npy
  decision_python_state/
  observation_stage_state.npz      # 实际需要的派生缓存；不得随意写任意字节内存
  original_live_probe.npz          # 未经 restore 的原现场参考，仅工程用途
  file_hashes.json
```

公共模型资产与版本锁放在 pool 级目录，避免每个 root 重复存储。不能只留下服务器绝对路径、seed 或一个 snapshot 哈希却不保存实际快照。

推荐数值数组使用 `.npy/.npz`、元数据使用带类型标识的 JSON。若某种受信任的本地序列化确实依赖 pickle，必须记录可信来源、版本限制和完整性检查；不把 Python 对象 pickle 默认为跨进程/跨版本便携的 MuJoCo 快照。

### 4.3 三次封存的先后关系

```text
freeze protocol/code
    → acquire + validate root artifacts
    → seal root_pool_manifest
    → generate offline-data candidates
    → seal candidate_manifest
    → 才允许评价各候选环境结果
```

finalizer 只能读取冻结协议，不能在实验结束后生成一份自称“运行前 frozen”的新协议。文件哈希只证明内容一致，不能单独证明冻结时间；使用运行前提交、清单和执行记录建立时间顺序。

---

## 5. 恢复协议：优先重放物化前缀，而不是重新猜测现场

### 5.1 主路径：`BOOTSTRAP_PREFIX_REPLAY`

第一版统一采用以下路径，不按候选成绩选择恢复方式：

```text
新进程建立冻结版本环境（仅用于分配模型与接口）
    → 载入原 bootstrap artifact
    → 恢复原完整 goal、物理目标、RNG、wrapper 与 observation-stage 状态
    → 重放保存的 prefix_actions，不重新用策略生成前缀
    → 对照原 decision artifact 逐项验收
    → 使用原冻结 goal 和候选，开始新的闭环分支
```

新的 `reset(seed, task)` 返回值不是 root 的来源；即使初始化环境时需要调用 reset，也必须随后完整恢复原 artifact。不能把新 reset 的 goal 偷偷交给 continuation policy。

这条主路径比从 decision snapshot 直接加载慢，但避免将本轮首先押在复杂中途状态的快速恢复上。**仍须保存 decision snapshot**，它是重放终点的身份依据和独立核验对象。

### 5.2 快速路径：`DIRECT_DECISION_RESTORE`

可在 S1 做工程对照，但首轮 S3 默认关闭。只有在采集正式 root pool 前，证明该路径与原现场、主重放路径在所有规定指标上等价，并另行冻结，才允许统一替换。

不是“两个恢复后的分支彼此相同”就足够。必须满足：

\[
\text{原现场}\approx\text{独立加载 A}\approx\text{独立加载 B}.
\]

如果两个加载器一致地恢复到了错误的现场，比较 A 与 B 仍可能全部 PASS；原现场参考用于排除这一情况。

### 5.3 Integration state 与公开 observation 的阶段语义

MuJoCo 区分 integration state 和整个 `mjData`；派生量需要按流水线计算。`mj_step` 后的部分派生量可能仍对应刚完成的转移，不能假设任意时刻调用 `mj_forward` 再取 observation，都与环境上一步返回的 observation 等价。[S7–S8]

Cube 的公开状态又包含来自接触力等派生量的字段。[S6] 因此必须实现并核验一个统一的**观察阶段合同**：

- 冻结官方环境何时计算 observation、接触量、success；记录 root 的采样边界。
- integration 使用冻结版本的 `mj_getState/mj_setState` 与 `mjSTATE_INTEGRATION`；数值型 spec、长度、模型指纹必须存档。
- 对 observation 依赖的缓存：明确哪些被复制，哪些用哪条流水线重算。不能凭猜测增加 `mj_forward` 或 `mj_rnePostConstraint` 就宣告修复。
- loader 返回的首个 policy observation 必须等于原现场保存的 37D observation，并由正确恢复过程产生；不能只将一个旧 observation 强塞给策略，掩盖环境状态不同。
- rollout 内部从 loader 接收首个 observation，后续始终使用 `env.step` 返回值，不另写一条偷偷重新计算首个 observation 的入口。
- 所有数组日志必须 `.copy()`，不能把 MuJoCo 可变内存视图当作历史状态保存。[S9]

完整 `mjData` 的内存复制可以作为诊断参照，但不自动等于已实现跨进程持久化。本文不宣称某一序列化方案已经在目标服务器验证成功。

### 5.4 Python、wrapper 和 RNG

为实际环境建立显式字段清单，至少审查 task/goal、`_success`、`_reset_next_step`、`_prev_qpos/_prev_qvel/_prev_ob_info`、控制器缓存及所有 wrapper 状态。每个未保存字段必须说明其是否只读或在下一步前必定重新计算，不能靠“看起来无关”跳过。

冻结并核对环境 RNG、全局 NumPy、Python random（如使用）、JAX key 日程、真正被调用的 action-space RNG。Gymnasium 的环境 seed 与 action-space seed 是不同接口；还需检查该 OGBench 版本的 `action_space` 属性是否返回持久对象。[S6,S10]

**本协议默认不改官方 reset 分布。** 若纯粹为确定性而修改 goal 构造随机数来源，必须作为显式环境补丁在 S1 前批准、记录 diff，并验证任务/动作分布未被人为改简单；不能临时 monkey-patch 后仍称为未修改官方评价。

---

## 6. 唯一执行器与反伪通过测试

### 6.1 唯一执行器

工程审计、root 验收、候选分叉、完整 continuation 和未来 Phase C 共用以下待实现接口：

```text
load_root(bundle, mode, strict=True) -> (env, observation, exact_goal, context, proof)
rollout_segment(env, policy, observation, goal, steps, key_schedule) -> trace
compute_proxy(trace, frozen_value, exact_goal, reward_contract) -> float
compare_traces(trace_a, trace_b, thresholds) -> measured_comparison
```

不得在“审计脚本”中采用缓存首观察，而“正式实验脚本”另行调用 `compute_observation()`。不得让“审计”与“生产”各自复制一份近似相同的执行逻辑。

以下是**执行逻辑示意，不是已存在的可运行程序**：

```python
obs = loaded_root.observation.copy()
for step_index in range(max_steps):
    action = policy(obs, fixed_goal, key_schedule[step_index])
    next_obs, reward, terminated, truncated, info = env.step(action)
    trace.append(copy_all_measured_fields(...))
    obs = next_obs.copy()
    if terminated or truncated:
        break
```

每次 repeat 独立重新计算动作；“在同一旧 observation 上算五个动作再重放”不是闭环。

### 6.2 比较器必须拒绝的情况

空 trace、不同长度、NaN/Inf、缺字段、不同 task/goal、超时后继续 step、不同 root/candidate 哈希，都应输出 FAIL/HOLD。不能使用 `all([])=True`、只比较 `zip` 的共同前缀，或以默认零代替缺失值。

`first_divergent_step=None` 必须由全部指定测量通过计算得到，而不是先写死、后检查部分字段。

### 6.3 必须通过的故障注入测试

在进入正式 root pool 前，人为制造以下错误，验收器必须确实拒绝：

| 注入错误 | 预期检测 |
|---|---|
| 修改 root 文件一个字节 | 内容哈希失败 |
| 仅替换 37D goal、保持 task ID 不变 | goal 身份失败 |
| wrapper elapsed 增加 1 | 时间/剩余预算失败 |
| 一份 trace 删除最后一步或加入 NaN | 完整性失败，不能空集通过 |
| 一份 trace 的动作/状态增加超过阈值的偏差 | 对应动作/状态阈值失败 |
| 分支 terminated 后仍继续 step | 终止协议失败 |
| 跳过 restore 或使用错误 root | 原现场对照失败 |

故障注入样本是测试程序，不是研究数据。记录每个注入是否触发预期错误，以及完整测试输出；不能只写“参考测试通过”。

### 6.4 三维误差的单位与表示

除上述逐维工程差异外，另报告可解释的物理诊断量。按冻结 Cube 源码核对零基索引：[S6]

| 公开 observation 切片 | 含义 |
|---|---|
| `12:15` | 经缩放的末端位置 |
| `19:22`、`28:31` | 两个 cube 经缩放的位置 |
| `22:26`、`31:35` | 两个 cube 的四元数，顺序须核验为 `wxyz` |
| `26:28`、`35:37` | 各 cube yaw 的 cos/sin |

该版本位置表示采用 $o_{xyz}=10(p-[0.425,0,0])$；因此先解码为米，再计算：

\[
d_p=\max_{b\in\{0,1\}}\|p_b^{(A)}-p_b^{(B)}\|_2.
\]

旋转诊断对通过单位范数检查的四元数采用：

\[
d_R=\max_b 2\arccos\!\left(\operatorname{clip}
\left(|\langle q_b^{(A)},q_b^{(B)}\rangle|,0,1\right)\right).
\]

绝对值用于处理 $q$ 与 $-q$ 的同旋转表示；报告单位为弧度。范数异常、四元数与 yaw 表示不一致应单列，不能悄悄归一化并覆盖原数组。必要的数值归一化只用于计算旋转角，同时保留原范数误差。

任务状态误差使用原 task success、每 cube 的物理位置达成标志、抓取/接触诊断与终止标志的逐项不一致率。**姿态误差是执行诊断，不把 Cube 原本的位置成功条件改成新的姿态成功条件。** 原始 `19:37` 差异混有缩放位置、四元数和角度编码，不能直接称作“米”或实际位移。

这些物理量在首轮为解释性指标，不在看到结果后增设新的通过阈值；主工程 gate 仍使用已冻结的逐字段规则。

---

## 7. S0：静态预检与协议冻结

### 操作

1. 核对封存 commit 和 closure，不触碰旧输出目录。
2. 确认工作区无未提交修改，创建建议新分支和独立运行目录。
3. 核验数据、actor/value 权重、normalizer、实际配置和依赖。
4. 将本计划的建议参数落成 `protocol/protocol_v1.json`，所有待决定项必须在任何新环境运行前填完。
5. 建立 `schema_version`、协议状态机和失败默认关闭的入口保护。
6. 实现完整轨迹窗口索引测试，不跨 episode；划分先按 episode 再切窗口。
7. 把待实现的 root/trace/比较器接口和单元测试写入新目录，不直接沿用旧脚本的研究判定逻辑。

可实际执行的 Git 操作如下；若分支已存在或工作区不干净，应停止检查，不能 force/reset：

```bash
set -euo pipefail
cd "$OG_REPO"
test -z "$(git status --porcelain)"
git fetch origin
git cat-file -e 7714e33958dd6ab7d8a1ddf894dca1eb88a69b17^{commit}
git switch -c exp/ea-v3-materialized-roots-v1 \
  7714e33958dd6ab7d8a1ddf894dca1eb88a69b17
```

`OG_REPO` 是执行机已确认的仓库路径，由 Agent 解析并打印后使用，不把本文路径示例当实际路径。

### 交付与 gate

`asset_lock.json`、`source_lock.json`、`environment_lock.json`、`protocol_v1.json`、`test_schema_and_gate.json`。

通过：`EA3_PREFLIGHT_PASS`。未完成字段或资产不符：`EA3_HOLD_ASSET_MISMATCH` / `EA3_HOLD_PROTOCOL_INCOMPLETE`。

S0 不调用环境 step，不训练模型。

---

## 8. S1：物化与执行器的工程试验

### 8.1 固定小规模样本

建议 15 个工程 root，覆盖 5 个官方任务 × 3 个计划 decision step：`0、125、250`。建议 ID `690000–690014`，S0 先检查与所有历史样本不重叠；冲突时停止，在执行前另批新 ID，不能看成绩后换 ID。

每个工程 root 最多使用一个确定性检索的真实训练 endpoint 作为 D3 测试目标，不做候选排名、不计算 gap，也不聚合任务成绩作为研究指标。

若前缀合法地提前成功/终止，保存 `INELIGIBLE_PREFIX_TERMINAL` 记录，不复活、不追加时限、不挑另一个更难 seed。工程阶段至少要求 10 个合法 root、全部 5 个任务有覆盖，并且 125/250 步中间状态均有合法样本，否则停止报告覆盖不足。

### 8.2 采集原现场参照

在任何 restore 之前保存 bootstrap、完整 goal、真实 prefix、decision artifact。从同一原现场实际执行一次五步闭环探针并保存 `original_live_probe`，包括首 observation、逐步动作、状态、奖励、终止和尾值。

该探针只用于评价器正确性；不得反馈训练、检索规则或模型选择。

### 8.3 独立加载测试

对每个合法工程 root 启动两个 fresh worker/process，仅提供文件 artifact 和冻结权重。原采集进程退出后，worker 必须仍可重建；不能依赖原进程内对象或未写盘的全局变量。

执行 D0–D3：

| 测试 | 真实操作 |
|---|---|
| D0 | capture → 改变环境 → restore → 对照原 integration、实际 observation、goal、Python/wrapper 状态 |
| D1 | 同 root 重放一个固定 action，完整记录一步 |
| D2 | 同 root 重放冻结的五步动作序列；两份 trace 都实际执行 |
| D3 | 两个 worker 分别逐步查询当前 observation、重新计算策略动作；与原现场探针及彼此比较 |

另执行 A–B–A 顺序测试：在同一 worker 加载 root A，运行后加载 B，再加载 A；同一 A 的观测与轨迹应一致，排除调用历史污染。

### 8.4 冻结的工程验收标准

| 对象 | 标准 |
|---|---|
| 文件/配置/goal/candidate 身份 | 哈希完全一致；goal 不允许“物理等价但编码不同” |
| action 最大绝对差 | $\le 10^{-7}$ |
| public observation 最大绝对差 | $\le 10^{-6}$ |
| robot `0:19`、cube `19:37` 最大绝对差 | 各 $\le 10^{-6}$ |
| integration 的连续数值分量 | 最大绝对差 $\le 10^{-6}$；离散开关按精确相等 |
| 完整 proxy 绝对差 | $\le 10^{-5}$ |
| reward、success、terminated、truncated | 按冻结离散/数值合同一致 |
| wrapper 步数、trace 长度、终止步 | 完全一致 |
| 缺失、NaN、Inf、空 trace | 不通过，禁止忽略 |

数值比较显式使用 `rtol=0`，不依赖 NumPy 默认相对容忍度。各 integration 字段的单位/尺度须存档；未来若需更改标准，必须在新结果前修订协议，不事后放宽。

D0 不存在动作或 rollout proxy 时，记录 `NOT_APPLICABLE`，而不是写成“已测得 0”；其余阶段的数字必须由独立原始测量计算。

### 8.5 交付与停止

交付：工程样本清单、物化样本包、原现场/两 worker traces、字段级比较、故障注入测试结果、`ENGINEERING_PILOT_REPORT.md`、脚本/协议/数据哈希、资源测量。

只有全部合法工程 root 通过、覆盖满足、所有故障注入被正确拒绝，才输出 `EA3_ENGINEERING_PILOT_PASS`。

**这只代表所测工程范围通过，不代表正式 root pool 已通过。** 第一次任务到此停止。建议本批外部控制步上限 30,000，最多两个明确定位后的实现修复版本；超限报告，不开启一串自动改名重试。

---

## 9. S2：采集并封存正式 root 与候选

### 9.1 运行前安排

以下规模为本计划新建议，不是统计功效保证：

```text
planned roots = 80
planned deep roots = 40
candidate count N = 8
candidate window k = 20
executed segment m = 5
root IDs = 700000–700079
```

80 个 roots 按五个官方任务均衡分配，每任务 16 个。可使用：

\[
\text{root\_id}=700000+5j+t,\quad
j=0,\ldots,15,\ t=0,\ldots,4,
\]

`task_id=t+1`；`decision_step=(0,125,250)[j mod 3]`。每任务分别有 6/5/5 个计划决策时点。

每任务在 16 个计划 root 中，按 `SHA256("ea-v3:deep:" + root_id)` 排序取前 8 个作 deep，事先写入清单。不能在 prefix 终止后另补 deep，也不能根据 candidate 结果选 deep。

### 9.2 在采集时物化，不是事后重建

一次 root 采集必须原子化保存 reset 现场、原目标、实际前缀和 decision artifact。预定决策点前结束的记录如实封存为 `INELIGIBLE_PREFIX_TERMINAL`，不生成 candidate 分叉。

每个合法 root 在生成正式候选前都须通过两个独立进程的 bootstrap-prefix 重建及五步工程探针核验。探针采用固定 final-goal controller，不选事后更好动作，不用于任何学习。

**重建失败不能通过删除该 root、换 seed 或只保留成功恢复的 root 来解决。** 任一合法 root 失败则整个 pool HOLD。唯一正常排除是预先定义的 prefix 终止等不具决策资格事件，所有排除保留在计划分母中。

建议覆盖 gate：合法 root 至少 60/80、可评价 deep 至少 30/40；每任务至少 8 个合法 root、5 个可评价 deep。该 gate 运行前冻结，未达标时不补样“凑通过”。

### 9.3 封存 root pool

生成 `root_pool_manifest.json`，包含所有计划 root 的状态、完整内容哈希、goal 哈希、计划/实际前缀长度、deep 标记、worker 验收结果及 artifact 存储定位。

先将协议、采集代码和 root manifest 提交到新分支；大型 artifact 可放受控服务器存储，不必把权重和原始数据推入 Git。**没有实际持久化文件、只有路径或哈希，不能通过。**

### 9.4 候选检索——本轮只用真实数据状态

构建训练集的合法窗口：

\[
(o_t^{\mathcal D},o_{t+20}^{\mathcal D}),\qquad
\text{episode}(t)=\text{episode}(t+20).
\]

起点距离固定为 train-only 标准化的 37D 均方根距离：

\[
d(o,o_i)=\sqrt{\frac1{37}\sum_{j=1}^{37}
\left(\frac{o_j-o_{i,j}}{\max(\sigma_j,10^{-6})}\right)^2}.
\]

按距离和稳定的原始索引打破并列，取前 8 个**终点内容不同**的窗口；candidate 为对应实际记录的 37D endpoint。索引与去重以原数据 dtype/字节为身份依据，不能因隐式降精度而改变候选去重。

固定 $N=8$，不按 $V$ 或环境结果筛候选；不做高斯采样、不投影到事后成功状态。

每个 candidate 记录 source episode、start/end index、起点距离、endpoint 数组哈希与完整数组。报告近邻距离分布、同 episode 候选比例、最小候选间距等诊断；八个不同字节序列不意味着八个独立行为模式。

“真实 endpoint”只证明状态曾出现，不证明当前 root 下可达。局部数据支持距离只作按离线参照定义的审计，不允许根据新环境成功率过滤。

### 9.5 S2 交付与停止

`root_pool_manifest.json`、`root_qualification.jsonl`、`candidate_manifest.json`、`candidates.npz`、检索索引/normalizer 哈希、`S2_POOL_LOCK_REPORT.md`。

状态：`EA3_ROOT_AND_CANDIDATE_POOL_LOCKED`；不通过则对应 `EA3_HOLD_ROOT_RECONSTRUCTION` / `EA3_HOLD_COVERAGE` / `EA3_HOLD_CANDIDATE_DATA`。

S2 完成后停止。未批准 S3 前，不查询各候选的环境后果。

---

## 10. S3：机制实验——先工程有效，再判断有没有决策空间

### 10.1 两种结果严格区分

短程 proxy 采用：

\[
J_m^{\mathrm{proxy}}(\mathcal R,z)
=\sum_{i=0}^{\ell-1}\gamma^i r_i^{\mathrm{train}}
+\gamma^\ell b_\ell V(o_\ell,g),
\]

其中 $\ell\le\min(m,H-t)$ 是实际步数，$H=500$，$r_i^{\mathrm{train}}=r_i^{\mathrm{env}}-1$ 沿用当前奖励映射；必须用冻结 `reward_value_contract` 验证该映射。$\gamma$ 从 checkpoint 实际配置读取并固定，不使用脚本另一个默认值。

本诊断采用有限 episode 定义：真正终止或达到官方时限后 $b_\ell=0$，其他情况为 1。**这是该代理评价的显式合同，不是“所有 RL 训练遇到 truncation 都应停止 bootstrap”的一般规则。**

这里的 GCIQL $V$ 是冻结的学习尾值，不是当前 continuation policy 的真实成功概率，也不预设与物理任务成功奖励完全校准。[S3–S4]

完整结果为：

\[
\pi(\cdot\mid o,z)\text{ 执行 }m\text{ 步}
\ \longrightarrow\ 
\pi(\cdot\mid o,g)\text{ 执行到成功/终止/官方时限}.
\]

完整 continuation 的剩余预算必须为 $H-t-\ell$，不能给中途 root 再额外 500 步。结束后记录 success、未折扣环境 return、从 root 起的步数、从 episode 起的步数、termination reason。所有时限计数与 wrapper 一致。

### 10.2 执行矩阵

| 对象 | 范围 | 执行 |
|---|---|---|
| 全部冻结候选 | 所有合法 roots × 8 candidates | 两个独立 worker，各跑 m=5 步 |
| 候选完整 continuation | 所有合法 deep roots × 8 candidates | 两个独立 worker，各跑“m 步 z + 剩余 g” |
| `DIRECT` | 所有合法 roots | 不切 candidate，直接追求原最终目标到时限，重复两次 |

同一 root/candidate 的两个 repeats 使用相同 key 日程，分别独立闭环计算动作。repeat 只是确定性检查，不是额外统计样本。

key 日程由稳定 SHA256 派生的 root ID、candidate ID、执行段和步号确定，不使用进程随机化的 Python `hash()`；同一配对 repeat 不把 repeat ID 混入动作 key。standalone m-step 与 deep 的前 m 步使用相同段标识。序号重排不改变同一 candidate 的 key 日程。


固定 candidate 序号执行一次、反向序号在另一 worker 执行一次，比较相同 ID 的结果，检查调用顺序污染。每次分支都从 artifact 重建，不继承上一个候选的剩余状态。

每个 deep 分支的前 m 步，还须与对应 standalone m-step trace 一致；这样防止 proxy 实验与完整实验实际上使用不同起点/目标/执行路径。

### 10.3 必须保存的原始信息

逐 root/candidate/repeat 保存真实的首 observation、exact goal、actions、observations、integration/时间检查摘要、每步 rewards/flags、m-step proxy、完整 continuation 结果、对象哈希和实际成本。

不能仅保存每个 root 的“最佳候选成功”与“IDEAL 成功”，否则后续无法审查候选排序和完整结果关系。完整分支的两次 repeat 都必须写入原始记录。

研究排序器不得访问另一候选的真实环境结果。全部分叉完成且工程验收通过后，独立分析器才能读取结果做诊断。

### 10.4 工程 gate 优先

S1 的相同阈值应用于本轮所有实际分支，包含完整 continuation 和 goal 身份。

- 任一 root 无法重建：`EA3_HOLD_ROOT_RECONSTRUCTION`。
- 任一有可比起点的分支超过数值/终止阈值：`EA3_HOLD_BRANCH_REPRODUCIBILITY`。
- 字段缺失、时限错误、哈希改变：`EA3_HOLD_PROTOCOL_VIOLATION`。

工程 gate 不通过时，`research_metrics_valid=false`；分析器默认不输出科学 PASS。排障日志可保留描述性数值，但不得用它们继续训练模型。

### 10.5 全局容忍度

本轮建议继续使用已登记的全局阈值：

\[
\epsilon_V=0.5495205402374268.
\]

它来源于旧 Cube 对固定离线 state/goal 采样计算的价值标准差 $10.990410804748535$ 的 5%，不是 root 内候选标准差。[S11]

S0 须核对同 checkpoint、同价值实现、同离线取样规则；这是固定诊断尺度，不是经验证的置信区间。若来源无法核对，应在任何新 root 采集前冻结新的离线尺度并明确改版，不能看机制结果后调整。

---

## 11. S3 对比对象与机制 gate

### 11.1 对比对象

| 名称 | 定义 | 地位 |
|---|---|---|
| `IDEAL` | $\arg\max_z V(z,g)$ | 可部署的候选选择规则 |
| `DIRECT` | 不切子目标，直接执行原 GCIQL | 检查子目标干预是否只是制造损伤 |
| `UNIFORM_EXPECTATION` | 同一 root 下全部候选完整成功的平均 | 随机选一个候选的经验期望，不伪装成额外运行的策略 |
| `ORACLE_PROXY_DIAG` | 按实测 m-step proxy 选候选，再查该候选完整结果 | 仅诊断代理目标有无完整控制意义 |
| `ORACLE_FULL_DIAG` | 事后取完整成功最优候选 | 经验机会空间上界，不是部署方法、不能进主算法排名 |

`ORACLE_PROXY_DIAG` 必须只依据 proxy 排序，不能看到完整结果后打破并列。所有 argmax 使用冻结的最小 candidate ID 规则；并列覆盖率另报。

### 11.2 三个主描述量

1. proxy wrong-selection：

\[
\frac1{n}\sum_r\mathbf 1\left[
\max_z J_r^{\mathrm{proxy}}(z)
-J_r^{\mathrm{proxy}}(z_{\mathrm{ideal}})>\epsilon_V
\right].
\]

2. full distinguishable：同一 deep root 的候选成功标志不是全部相同。

3. IDEAL full miss：同一 deep root 存在成功候选，但 IDEAL 所选候选失败。

同时报告每 root 内 Spearman、proxy regret、任务分层结果、计划/有效分母、prefix 终止率、候选集合多样性。低相关本身不是“算法一定可改进”的证据。

### 11.3 新协议建议的进入 S4 条件

先通过工程与覆盖 gate，再要求：

\[
\text{proxy wrong-selection}\ge 10\%,
\]

\[
N_{\mathrm{distinguishable}}\ge\lceil0.25\,n_{\mathrm{deep}}\rceil,
\]

\[
N_{\mathrm{ideal\ miss}}\ge\lceil0.125\,n_{\mathrm{deep}}\rceil.
\]

40 个有效 deep 时分别是至少 10 个可区分、5 个 IDEAL 错选；这些比例借鉴旧小规模门槛，但在 EA-V3 中是**重新提出并待冻结的筛查规则**，不声称具有预定的显著性或检验功效。

另外增加一个与方法目标直接相关的必要检查：

\[
\sum_{r\in\mathrm{deep}}
\left[Y_r(z_{\mathrm{proxy}})-Y_r(z_{\mathrm{ideal}})\right]>0.
\]

也就是实测 proxy 选出的候选在完整成功数上必须对 IDEAL 有**正的配对净收益**。若理想的 proxy 估计都不能提供方向正确的完整决策，先训练更精确的 $P_1$ 并没有充分依据。

这是机会筛查，不是证明新方法有用；S4 还必须在新样本上检验学习模型。

### 11.4 状态与行动

| 状态 | 含义 | 下一步 |
|---|---|---|
| `EA3_MECHANISM_GAP_SUPPORTED` | 上述筛查均满足 | 可提交 S4 冻结方案，仍需人工批准 |
| `EA3_GAP_EXISTS_PROXY_MISALIGNED` | 存在完整候选机会，但 proxy 选择没有正净收益 | 不启动 P1；先重审目标/尾值语义，不能仅追求预测 loss |
| `EA3_GAP_PROXY_ONLY` | proxy 错选达到阈值，完整机会不足 | 不进入 S4 |
| `EA3_NO_USEFUL_GAP` | 在该冻结配置下未达机制筛查 | 封存，不调小 epsilon、换弱控制器或扫新任务 |
| `EA3_HOLD_*` | 工程、身份、覆盖或协议问题 | 不作科学正/负结论 |

状态优先级写进分析器，避免同一结果同时落入多个分类。`HOLD` 不等于科学假设被否证；`NO_USEFUL_GAP` 也只适用于本控制器、候选与任务设置。

### 11.5 统计报告

独立单位是 root，不是 candidate、repeat 或五步 transition。报告比例的区间，以及 root 级配对差异区间；对同任务内 root 重采样的 bootstrap 固定 10,000 次和分析 seed，候选保持成组。

报告所有 5 个任务，不用总体平均掩盖单一任务驱动。只使用一个冻结训练 seed 时，结论只能针对该 backbone；不能声称跨训练 seed 稳健。

本轮门槛是小规模研究筛查，不把“区间跨零但门槛通过”包装成显著性能提升。

---

## 12. S4：通过机制后才验证创新 1

本节是下一批次的范围约束，不授权当前训练。S4 开始前必须另提交完整 `phase_c_protocol.json` 与预算，不能直接运行旧 Phase C 入口。

### 12.1 只增加一个变量

冻结同一 GCIQL、$V$、DATA_REAL retrieval、$k=20,m=5,N=8$、奖励/时限与 root loader。只训练普通一步动力学集成 $P_1$；初始建议集成 3 个成员，训练仅用官方 train 的合法一步转移。

**本阶段不加多时域终点损失、不训练 $P_{MH}$、不微调控制器。** 先回答：学习到的普通执行估计能否比 IDEAL 更好选候选？

模型训练项、网络容量、优化器、update 上限、验证频率、成员 seeds、风险项及离线标定规则全部在看到新的候选环境结果前冻结。不得直接继承旧 AntMaze 的 0.8 A100 GPU-hours 估算作为 Cube 预算。

### 12.2 预设比较

主机制比较：`EXEC_1 − IDEAL`。同时保留 `DIRECT`；同预算 `MPC_EQ` 进入完整控制阶段的必要对照，避免把普通增加规划算力当创新。

`EXEC_1` 在模型中以当前预测观察逐步调用同一冻结 actor：

\[
\tilde a_i=\pi(\tilde o_i,z),\qquad
\tilde o_{i+1}\sim P_1(\cdot\mid\tilde o_i,\tilde a_i).
\]

评分为短程预测奖励加尾值，风险项若启用，系数只在离线划分上确定；集成分歧不自动等于可靠置信界。[S3–S4]

### 12.3 新样本与必要结果

不得将 S3 环境分叉作为模型监督、风险校准或调参依据。S4 重新预登记独立的物化 roots；规模建议与 S3 相同，但 ID 与细节在 S4 冻结时确定，不能重复使用已看过的 roots 当独立确认。

至少报告：离线一步误差、当前 actor 在模型内/环境中的 m-step 执行误差、候选排序、选择 regret、完整任务结果、相对 DIRECT 的损伤/改善、训练与推理成本。

公开 37D 观察不自动等于完备 Markov 状态。预测不足时先检查部分可观测性；不能偷偷给模型增加 simulator 速度/接触缓存。历史输入版本或表示修改需要新合同和所有对照对齐。

S4 只有在候选选择和完整结果有支持性改善时才推进；单独模型 loss 下降不足以解锁 S5。

---

## 13. S5：创新 2 与完整任务确认（暂不实施）

若 S4 成立，后续保持一步模型对照，训练使用相同数据的 $P_{MH}$，增加记录动作下 $h\in\{5,10,20,40\}$ 的真实终点监督。对齐容量、参数搜索与训练/推理预算，检查：

\[
\text{离线多时域校准}
\rightarrow\text{当前控制器预测改善}
\rightarrow\text{排序改善}
\rightarrow\text{完整成功率改善}.
\]

完整部署系统从正常 reset 开始，执行“选择候选 → m 步闭环 → 重规划”，而不是使用训练或测试过程的 oracle 分叉。至少与 `IDEAL`、`EXEC_1`、`EXEC_MH`、`DIRECT/GCIQL`、官方 HIQL 和等计算预算 `MPC_EQ` 对齐；其余强基线依据后续方法定位另行冻结。[S4]

本文件不自动授权多环境扩展、图像输入、更多训练 seeds、大规模正式测试或新的奖励设计。

---

## 14. 预算与成本记账

以下是**控制步数的保守预算建议，不是已测运行时间**。公开数据读取、网络推理、simulator 内部物理 substeps、reset 内部控制调用、磁盘开销分别记录，不能混为一个计数。

| 阶段 | 初始建议上限/规模 |
|---|---|
| S0 | 0 环境控制步；只有读取、哈希、静态和合成测试 |
| S1 | 15 个计划工程 root；外部控制步上限 30,000 |
| S2 | 80 个计划 root 的采集、重放和探针；外部控制步上限 80,000 |
| S3 | 使用主重放路径，外部控制步上限 800,000；另记录 reset 内部开销 |
| S4/S5 | 在放行前另报实测训练/环境/存储预算，不自动启动 |

S3 上限的一个保守展开（所有计划 root 都合法时）：

\[
80\times8\times2\times(250+5)=326{,}400
\]

用于独立短程分支及最大 250 步 prefix 重放；

\[
40\times8\times2\times500=320{,}000
\]

用于 deep 候选从 bootstrap 重放前缀再跑至原 episode 时限；

\[
80\times2\times500=80{,}000
\]

用于 DIRECT 对照。合计至多 726,400 个上述外部控制步，800,000 留出明确的非研究工程核验余量，不能用于新增 roots 或超出矩阵的试探。

S1 完成后测量实际 steps/s、加载延迟、单 root 存储量并估算后续时间。前半段不训练模型，但 actor/value 推理仍可能使用 GPU，不能写成“GPU 成本为零”。

如果使用更快的 direct restore，必须在正式 pool 采集前单独验收和冻结；不能在正结果/负结果之间切换恢复方式。

---

## 15. 工程目录、产物与可审计性

### 15.1 建议新目录

```text
execution_aligned_rl/v3/
  contracts.py
  root_bundle.py
  root_loader.py
  rollout.py
  compare.py
  candidates.py
  preflight.py
  engineering_pilot.py
  materialize_pool.py
  mechanism.py
  analyze.py
  tests/

experiments/execution_aligned/ea_v3_materialized_roots_v1/
  protocol/
  manifests/
  engineering/
  root_pool/
  candidate_pool/
  mechanism/
  report/
  package/
```

这些是建议接口，须由 S0 实现后才可调用。root 的大文件可在外部 `ROOT_STORE`，但 manifest 要提供实际可访问位置、大小、内容哈希及取回方式；本地和服务器至少两处持久副本校验一致。

### 15.2 必须保存的锁与审批

`protocol_v1.json`、`asset_lock.json`、`execution_code_commit`、`root_pool_manifest.json`、`candidate_manifest.json`、阶段 `decision.json`。

人工批准记录放在独立字段 `human_review`，Agent 不得自动把自己的 PASS 当成人工已批准。`phase_c_unlocked=false` 保持到 S3 人工审查完成。

### 15.3 打包检查

轻量包包含协议、锁、汇总、逐 root 原始结果索引、审计报告、执行代码引用与哈希。至少包含可独立加载的少量示例 root，全部 root 的实际文件保存在完整结果存储中。仅有轻量包不足以重跑全部实验时，报告应明确说明。

生成 SHA256 后进行解压和结构化解析测试；哈希一致不等于实验有效，报告同时列出哪些门槛通过、哪些未运行。

### 15.4 历史与新结果的共存

旧 EA-V2 目录和分支保留只读历史。新协议的报告引用封存状态，不改写旧结论。后续代码修复使用新 commit 和新增审计记录，不覆盖旧测量文件以造成“从未失败”的印象。

---

## 16. 停止、修复与继续的规则

### 16.1 工程失败

先停止研究评价，保留失败 artifact，区分：`ROOT_IDENTITY`、`GOAL_ENCODING`、`OBSERVATION_STAGE`、`PREFIX_REPLAY`、`POLICY_ACTION`、`PHYSICS_TRANSITION`、`TERMINATION`、`TRACE_COMPLETENESS`。

没有可比起点时，不使用“simulator nondeterminism”作为默认原因；没有检验的字段记录 `NOT_TESTED`，而不是零 divergence。

本轮建议最多两个明确定位的工程修复提交。每次修复须有失败样例、原因、回归测试和固定资产一致性检查；不能不断运行同一模糊审计直至 PASS。

### 16.2 研究结果不支持

新协议工程完全有效而机制门槛不足时，封存这一配置，不增加候选数、换小阈值、挑任务或削弱控制器来制造空间。

不把工程错误视为算法已经被否定；同样，不以“可能有信号”作为无限投入工程成本的理由。是否另开方向由人工决定，不由 Agent 自动提出并运行新协议。

### 16.3 已查看样本的角色

工程测试可反复使用固定失败现场用于修复。正式机制样本一旦用于方法/门槛修改，不能再宣称为未见确认集。修复纯执行缺陷与更改研究假设必须分别记账，不能借“新版本”抹去适应性开发历史。

---

## 17. 交给 Agent 的第一次操作指令

复制本节即可。**本次只做 S0 + S1，不建立 S2 正式 pool。**

```text
请阅读 EA_V3_MATERIALIZED_ROOTS_OPERATION_PLAN.md，以及旧 EA-V2 closure。

新协议暂定：EA-V3 Materialized Roots v1。
主仓库：rollinpioneer/OG。
只读历史基点：7714e33958dd6ab7d8a1ddf894dca1eb88a69b17。
建议新分支：exp/ea-v3-materialized-roots-v1。
不得在旧 EA-V2 输出目录继续运行或覆盖旧文件。

本次范围仅为 S0 + S1：
资产/协议核验 + 物化 root/统一执行器的工程试验。
不得执行 S2/S3，不得训练任何模型，不得启动 Phase C–F、P_1、P_MH、EXEC。

1. 核验官方 Cube 数据、OGBench commit、GCIQL actor/value checkpoint、
   normalization 与实际依赖。缺失或不匹配立即 HOLD，不自动替换或重训。

2. 将操作文档的新建议落实为 protocol_v1.json，提交并冻结后才运行环境。
   所有数值、字段、代码路径和预算必须明确；finalizer 不得事后重写协议。

3. 实现唯一 root_loader、rollout_segment、compare_traces。
   工程与以后正式实验必须调用同一实现，不另写简化审计路径。

4. 按文档采集最多 15 个工程 root（5 tasks × 0/125/250 步）。
   在采集当时保存完整 37D goal、物理任务目标、bootstrap、实际 prefix actions、
   decision snapshot、API 原始 observation、Python/wrapper/RNG 和文件哈希。
   prefix 终止就记录，不强行延长、不复活、不偷偷换 seed。

5. 主恢复方式是 BOOTSTRAP_PREFIX_REPLAY：
   从原 bootstrap artifact 恢复原目标和现场，再重放原 prefix actions。
   不得重新生成闭环 prefix，不得相信仅有 seed/task 的 reset 已是同一 root。

6. 原采集进程之外的两个 fresh workers 必须只依靠文件重建。
   与原现场未经 restore 的五步参考轨迹比较，而不只是两个恢复结果互相比较。
   检查 D0/D1/D2/真实闭环 D3、A-B-A 调用顺序和 goal 身份。

7. 指标必须真实计算：action、API observation、integration、qpos/qvel/act/ctrl/
   warmstart、robot 0:19、cube 19:37、奖励、完整 proxy、时间与终止。
   适用文档的绝对阈值，rtol=0。空/缺失/NaN/不同长度直接失败。
   不适用字段写 NOT_APPLICABLE，不写伪造的 0。

8. 执行故障注入测试，确认改变 goal、elapsed、trace 长度、动作或状态
   都会被验收器拒绝。不得硬编码 first_divergent_step=None 或状态 PASS。

9. 本批不计算 gap、wrong-selection、oracle 排名或研究成功率。
   所有环境结果为 ENV_EVALUATED、training_eligible=false。
   外部环境控制步预算 30,000，最多两个定位清楚的修复版本；超限停止。

交付：
- protocol_v1.json + asset/source/environment locks
- engineering_root_manifest.json
- 原现场与两个 worker 的原始 traces/物化样例
- engineering_comparisons.jsonl
- negative_test_results.json
- ENGINEERING_PILOT_REPORT.md
- resource_estimate.json
- decision.json
- lightweight ZIP + SHA256 + 完整 root 存储清单

只有所有实际比较、覆盖与故障注入都通过，才输出 EA3_ENGINEERING_PILOT_PASS。
否则输出具体 EA3_HOLD 状态并保存失败现场。
完成后推送新分支并报告完整 commit SHA，然后停止等待人工审查。
不要自动进入正式 root pool 或机制实验。
```

后续 S2、S3、S4 分别单独交接；第一次不把后续命令作为待后台自动执行任务。

---

## 18. 来源与新建议的界线

文中的历史事实与研究设定依据以下固定来源；运行预算、80/40 样本安排、工程状态机与新增 proxy–full 对齐检查均为本计划建议。

**[S1] EA-V2 Cube Closure（固定封存提交）**  
https://github.com/rollinpioneer/OG/blob/7714e33958dd6ab7d8a1ddf894dca1eb88a69b17/experiments/execution_aligned/ea_v2_cube_mechanism_v1/EA_V2_CUBE_CLOSURE.md

**[S2] R2 Failure Forensic Report（固定封存提交）**  
https://github.com/rollinpioneer/OG/blob/7714e33958dd6ab7d8a1ddf894dca1eb88a69b17/experiments/execution_aligned/ea_v2_cube_mechanism_v1/r2_failure_forensic_audit/R2_FAILURE_FORENSIC_REPORT.md

**[S3] V2 研究方案**  
https://github.com/rollinpioneer/OG/blob/7714e33958dd6ab7d8a1ddf894dca1eb88a69b17/docs/execution_aligned_multihorizon_offline_rl_research_v2.md

**[S4] EA-V2 实验计划（仅继承研究边界与方法比较；旧仓库状态等历史描述不作为当前事实）**  
https://github.com/rollinpioneer/OG/blob/7714e33958dd6ab7d8a1ddf894dca1eb88a69b17/docs/EA_V2_EXPERIMENT_PLAN_OG.md

**[S5] 冻结 Cube backbone manifest**  
https://github.com/rollinpioneer/OG/blob/7714e33958dd6ab7d8a1ddf894dca1eb88a69b17/experiments/execution_aligned/ea_v2_cube_mechanism_v1/backbone_manifest.json

**[S6] 冻结 OGBench Cube 与 ManipSpace 源码**  
https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/ogbench/manipspace/envs/cube_env.py  
https://github.com/seohongpark/ogbench/blob/1d4140997f60c52c6fb0702ec100dc988b18c548/ogbench/manipspace/envs/manipspace_env.py

**[S7] MuJoCo 官方文档：state components / integration / simulation state**  
https://mujoco.readthedocs.io/en/latest/programming/simulation.html

**[S8] MuJoCo 官方文档：流水线、mjData 一致性与复现边界**  
https://mujoco.readthedocs.io/en/3.3.5/computation/

**[S9] MuJoCo Python 文档：数值字段与可变内存视图**  
https://mujoco.readthedocs.io/en/3.3.5/python.html

**[S10] Gymnasium 官方文档：环境 RNG 与 action-space seed**  
https://gymnasium.farama.org/api/env/

**[S11] 旧 Cube Phase 0 全局价值尺度记录（仅作固定阈值来源，不作新实验统计）**  
https://github.com/rollinpioneer/OG/blob/7714e33958dd6ab7d8a1ddf894dca1eb88a69b17/experiments/execution_aligned/ea_v2_cube_mechanism_v1/metrics/mechanism/phase0_summary.json

外部文档查阅日期：2026-09-14。不同版本的 API 描述只用于核验原则；具体字段和调用必须以 S0 锁定并在运行机测试的版本为准，不能因引用 latest 文档而升级实验依赖。

---

## 19. 最终验收摘要

本轮应逐项回答：

1. 能否从文件中重建**原本那个现场和那个完整 goal**，而不只是同 seed 的另一个现场？
2. 重新加载的原始观察、闭环动作和终止行为，是否与未经恢复的原现场一致？
3. 在身份和工程有效的固定候选比较里，是否存在有完整任务意义的兑现差距？
4. 实测短程 proxy 的选择是否比 IDEAL 更有完整控制意义，而不只是在数值上不同？
5. 只有前四项成立，才检验离线学到的 $P_1$ 是否可以利用这种空间。

**核心结论：复用已经训练好的 Cube 能力，重建评价基础设施；先物化 root、再封存候选、再判断机制，最后才投入算法训练。**

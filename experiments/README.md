# Scaling experiments

主模型与 30 个实验均已完成。本文件保留配置生成和运行方式。正式报告使用全验证集结果；重新生成报告拟合请运行 `python scripts/analyze_report_complete.py`。下文 `aml.scaling analyze` 默认分析训练期间保存的监控子集指标。

# Part 4：先完成路径一，再扩展路径二

当前仅准备代码和实验设计，没有启动 GPU 实验。所有训练复用 aml.train，主模型架构、训练代码和运行目录不变。

## 设计与预算

| 档位 | 固定训练量组 | 等计算量组 | 独立训练数 | 估算 FLOPs / 主模型 |
|---|---|---|---|---|
| pilot | 每个模型 67,108,864 tokens | C≈1e16、3e16 | 15 | 0.218 |
| full | 每个模型 1,048,576,000 tokens，与主模型一致 | C≈1e16、3e16、1e17 | 20 | 1.496 |

full 可复用 pilot 的 10 次等计算量训练，二者合起来共有 25 个不同 run；不是重新训练全部。pilot 的短固定-D 组仍可作为补充实验，不混入 full 的固定-D 拟合。

以上是 **6ND 估计计算量**，不是墙钟时间或 GPU 小时保证，小模型硬件利用率可能较低。正式运行之前先检查 budget.csv。若资源不足，应先完成 pilot 后缩减/调整正式预算，而不是伪装已经完成主模型外推验证。

5 个规模使用非嵌入参数量约 5M、11M、25M、50M、99M。词表 16k、上下文 1024、位置编码、归一化与前馈类型和主模型一致；同时改变宽度/深度，保持每个注意力头 64 维。精确参数量写在计划中。

每个 (N,D) 都是独立随机初始化并走完自己的 warmup+cosine。不能把长 cosine 训练的中途 checkpoint 当成短预算完整训练结果。

小模型有效 batch=32768 tokens，起始 LR 按主模型 batch 比例平方根缩放，约 1.414e-4。这个值是**待试跑验证的初始配方**，不是已调优结论。小预算不能直接用主模型大 batch，否则更新步数太少。优化器其他配置一致；因此规模曲线应称为“该配方下的经验规律”，不要声称无条件最优规律。建议先挑不同规模各一项做稳定性检查，再决定是否增加小范围学习率试验；若调整配方，重新生成独立计划，不覆盖已有实验。

## 命令

已生成的计划：path1_pilot/plan.json 和 path1_full/plan.json。每个计划附独立配置及 budget.csv；它们是计划，不是实验结果。

重新生成到空目录：

```bash
python -m aml.scaling plan --profile pilot --output experiments/new_pilot
python -m aml.scaling plan --profile full --output experiments/new_full
```

先查看将要运行的 ID（不会分配 GPU）：

```bash
python -m aml.scaling run --plan experiments/path1_pilot/plan.json --data data/tinystories --runs runs/scaling --gpu 1
```

真正执行需显式加 `--execute`。默认使用本组物理 1 号卡，UUID 被硬编码限制为本组 0/1 两张卡；若选择的卡有计算进程，拒绝启动，不会终止任何人的任务。主模型当前使用 0 号卡，不要选择 0。执行器串行运行各实验，不会同时占用两张卡。`--only ID` 可选择一个或几个计划中的 run：

```bash
python -m aml.scaling run --plan experiments/path1_pilot/plan.json --data data/tinystories --runs runs/scaling --gpu 1 --execute
```

在服务器运行应由独立 nohup/tmux 会话管理，并保留日志。执行器会跳过完整 run；有 last.pt 时恢复；有未完成目录但没有恢复点时明确报错，不自行删除。不要将 runs/scaling 指向 runs/main。

完成 pilot 后，用 full 计划和同一个 runs/scaling 目录继续，可复用共有的 run。所有计划共用课程数据预处理产物，避免每个实验重新划分验证集。

## 汇总与拟合

```bash
python -m aml.scaling collect --plan experiments/path1_pilot/plan.json --runs runs/scaling --output runs/scaling/pilot_results.json
python -m aml.scaling analyze --results runs/scaling/pilot_results.json --output runs/scaling/pilot_analysis
```

正式组及主模型均完成后：

```bash
python -m aml.scaling collect --plan experiments/path1_full/plan.json --runs runs/scaling --main runs/main --output runs/scaling/full_results.json
python -m aml.scaling analyze --results runs/scaling/full_results.json --output runs/scaling/full_analysis
```

规则：

- 汇总的是最后计划步的验证 loss，要求存在 last.pt。没有跑完、只有早期 loss 或 best.pt，都不能冒充完成。
- main 只读，不会用于拟合。默认比较的是主模型**最终步**的同口径验证 loss，而不是 Part 3 从 best.pt 得到的全验证集 final_metrics.json。
- 检查数据 metadata、tokenizer、上下文长度、验证文档数量、验证 tokens/bytes 一致。不一致拒绝比较。
- 固定 D 组拟合 L(N)=E+A/N^alpha_N，至少 4 个不同 N；输出系数、RMSE、边界命中和拟合 Jacobian 条件数（大值提示参数难以识别）。
- 5 个规模可做 leave-one-scale-out 外推敏感性检查；这是稳定性诊断，不是统计置信区间。默认只有一个 seed；需要可靠置信区间时，后续增加重复 seed 实验。
- 只有 N 和 D 都匹配，才输出主模型外推绝对/相对误差。pilot 的 fixed-D 预测不等于已预测主模型在 1.049B tokens 下的损失。
- IsoFLOP 训练步数要取整，budget.csv 同时记录目标预算、实际 C 与相对偏差；拟合使用目标预算分组，图中包络使用实际 C。
- compute-optimal 包络只是“本次采样网格中的最好点”。若最优点在最大/最小模型边界，报告明确提示扩展模型搜索范围。
- 至少 3 个 C 预算才拟合 L(C)=A/C^alpha_C，以及 N_opt(C)、D_opt(C) 的分配指数。因此 pilot 的两个预算只画包络点，不提前报告稳定的指数。
- alpha_N、alpha_C 与 N_opt(C) 的指数是三种不同量。与 Kaplan/Chinchilla 对比时必须按相同定义比较，不能把 loss 指数与模型预算分配指数混用。
- 重复采样 tokens 与独立语料量不同。全量固定-D 组与主模型都在相同有限语料上重复抽样，解释指数时需讨论这一限制。

缺少 run 默认拒绝分析；`--allow-partial` 仅用于诊断，报告会标记 partial。不得把它当完整实验结论。

产物：results.csv、analysis.json、各 run 的 loss-vs-tokens 与 loss-vs-compute 曲线、固定-D 模型规模曲线、IsoFLOP 曲线、compute-optimal 包络图。缺少足够的固定-D 点时不会产生该拟合图。

## 路径二的预留

分析命令支持 `--target-n` 指定更大的**非嵌入参数量**，输出同一 D 下的外推与外推跨度。该选项只提供计算接口，**不能据此宣称路径二已完成**。

路径一结果出来后，再决定接近/超过 0.2B 的训练规模、计算预算和外推目标；需要新增对应模型训练和配置，检查固定-D 与等计算量比较，并查阅公开资料讨论外推可信度。当前默认模型都小于主模型，因此尚不满足路径二的实验设计。也未实现完整二维 L(N,D) 参数化曲面；该项是作业选做内容。

## 验证

`tests/test_scaling.py` 检查参数计数、预算取整、两阶段配置复用、拒绝未完成实验、从人工幂律数据恢复指数、主模型保持独立以及图表生成。测试中的 loss 是明确标记的合成数据，临时文件自动清理，不会保存成正式实验结果。


## 已授权的同训练量正式组分配

2026-11-11 用户明确授权使用课程服务器本组两张卡，以及 gpu5 空卡，并同意把代码与编码数据上传到 gpu5。

- 课程服务器 GPU 1：99M，同训练量 1,048,576,000 tokens。
- 课程服务器 GPU 0：主模型及后续评测全部 COMPLETE 后，启动 50M。
- gpu5 GPU 6（A40）：25M。
- gpu5 GPU 7（A40）：先 11M，再 5M。

所有模型分别从头按完整预算训练。gpu5 上的前 10 步是同一完整配置的检查点，随后恢复，不改变学习率总时长。运行器新增 `--host-profile gpu5 --gpu 6/7`；课程服务器默认配置仍仅允许0/1。授权UUID固定，发现卡已占用时拒绝启动。

两台服务器使用相同编码数据，复制后已安排逐文件 SHA-256 验证。gpu5 使用个人目录 `~/aml-hw1`，不占用系统根盘存储数据。该机环境为 PyTorch 2.11.0+cu130/A40，课程服务器为 PyTorch 2.8.0a0 nv25.06/A100；公式、架构、BF16和实验配置相同，但硬件和软件版本差异须在报告中说明，不能声称跨机逐位一致。后续可在同一评测环境重新评测 checkpoint，减少评测数值实现差异；这不能消除训练阶段的数值差异。

分配记录见 `server_records/formal_dispatch.json`。这里只启动同训练量的五个实验，没有启动 full 中新增的第三档 IsoFLOP 或路径二。

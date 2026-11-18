# AML-Homework1

本项目独立实现字节级 BPE、decoder-only GPT、预训练循环、验证评测和文本生成。没有使用现成 GPT/Llama 模型或 Trainer。已完成主模型训练与 30 个 Scaling Law 实验。作者：Xizhi Xiao，College of AI, Tsinghua University。


## 已完成结果与文件

- [完整报告](report/main.pdf)：Parts 1–4、总结及补充材料。
- 主模型：218,974,208 总参数，201,541,632 非嵌入参数；训练 4,000 步、1,048,576,000 tokens。
- 完整验证集：loss 1.11692、PPL 3.05544、BPB 0.42039。
- 30 个 scaling runs；完整验证集拟合对主模型的预测相对误差 2.08%。
- `artifacts/custom_bpe/`：自训练 1,024 词表及验证统计；主模型使用课程提供的 16,000 词表。
- `experiments/path1_pilot`、`path1_full`、`path1_refinement`：实际实验配置。配置在计划间有复用，勿重复运行共享 ID。
- `report/full_validation_all.json`：主模型及 30 个实验的最终完整验证结果。

数据集及 checkpoint 不通过 Git 分发，最终 checkpoint 应按课程要求另行提交。加载模型还需要编码数据目录中的参考 tokenizer 和 metadata。所有 GPU 运行命令仅应在获得授权的设备上执行。

## 复现报告分析

安装依赖后，从仓库根目录运行：

```bash
python scripts/analyze_report_complete.py
```

该命令使用已保存的指标和训练历史重新生成 `report/final_analysis/` 的拟合及图像，不启动训练。
报告源文件可用 LaTeX 编译：

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

### 与报告一致的数据来源

报告使用 `roneneldan/TinyStories` revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` 的 `TinyStories-train.txt`，SHA-256 为 `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`。下载后可使用：

```bash
python -m aml.data --input raw/TinyStories-train.txt --input-format tinystories --tokenizer flexitok/bpe_fw_edu_16000 --output data/tinystories
```

参考 tokenizer 文件的 SHA-256 为 `e9ae6c97a8ae6b49bea5aaa8f7094755e243584a65a1119f88408d76444f2919`。下方也保留了其他数据输入方式；使用不同数据来源或 tokenizer 版本可能改变精确的 token 数。

## 1. 环境

建议在课程 A100 的 Linux 环境使用 Python 3.10+，先安装适配服务器 CUDA 的 PyTorch，再安装依赖：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

所有命令均从本项目根目录执行。正式数据和 checkpoint 不要提交 Git。下载数据需要能访问 Hugging Face；也支持本地 JSONL 和本地 tokenizer.json。

## 2. Part 1：参考分词器编码数据

默认使用课程参考分词器 `flexitok/bpe_fw_edu_16000`。TinyStories 示例：

```bash
python -m aml.data --dataset roneneldan/TinyStories --split train --output data/tinystories --max-train-tokens 1100000000
```

FineWeb-Edu 可替换为：

```bash
python -m aml.data --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --split train --output data/fineweb --max-train-tokens 1100000000
```

首次下载可能很慢。可先加 `--max-documents 10000`，输出到单独的试验目录验证流程。正式预处理运行时保存终端截图。

本地数据：每行一个 JSON 对象，例如 `{"text":"Once upon a time..."}`。

```bash
python -m aml.data --input corpus.jsonl --tokenizer /path/to/tokenizer.json --output data/local
```

数据说明：

- 按文档内容和固定 seed 的 SHA-256 划分 train/val，默认 99%/1%；完全重复的文档必定落入同一侧。不是近似去重，不承诺消除改写或近重复。
- 使用训练来源内部的固定留出集，不额外混用官方 validation。不要把留出集用于训练 BPE。
- 预处理采用 streaming，每次处理一篇文档；不把全数据装入内存。
- 每篇文档末尾添加已有 EOS；不会偷偷增加词表。如果无法识别 EOS，检查词表后传入 `--eos-id`。
- 保存 `train.bin` / `val.bin`、验证文档索引、原始 UTF-8 字节数、词表哈希和来源配置。
- `--max-train-tokens` 在完整文档结束后停止，因此可能略超预算；不指定或取 0 表示处理完整来源。
- 输出目录必须为空，避免误覆盖已有数据。失败会留下不完整目录，没有 metadata.json 的目录不可训练。
- 可用 `--revision` 和 `--tokenizer-revision` 固定下载版本。应保留本地 tokenizer、metadata 和数据文件以复现。
- `metadata.json` 中的 `vocab_size` 必须与模型配置一致；不一致会报错。

### 自己实现的 BPE

预处理自动导出仅来自训练侧的 `bpe_sample.jsonl`。训练自己的分词器：

```bash
python -m aml.bpe --input data/tinystories/bpe_sample.jsonl --output data/custom_bpe/tokenizer.json --vocab-size 1024 --max-documents 1000 --max-bytes 500000
```

算法从 256 个字节开始，统计相邻对，按频率逐步合并；频率相同使用确定性排序；编码按保存的 merge 顺序执行。EOS 额外占一个词表位置。产物包含词表、merges 和往返还原验证报告。若可合并对耗尽，实际词表可能小于目标。

自制分词器用于 Part 1 展示与验证；主模型仍使用课程参考分词器。这里使用清晰的计数/重扫实现，较大样本或 16k 次合并会慢；建议先完成 1k 词表实验，再按需要增加规模。作业没有明确规定自制词表必须 16k。

## 3. Part 2：模型

`aml/model.py` 实现：token embedding、可学习绝对位置 embedding、Pre-LayerNorm、多头因果自注意力、GELU 两层前馈网络、残差连接、最终归一化、共享输入/输出嵌入，以及 temperature/top-k/greedy 自回归生成。

注意力计算、缩放、下三角掩码和多头变换均显式实现，未调用现成 Transformer 层。softmax 使用 FP32 累积。PyTorch 负责自动微分和基础算子。输入/目标错开一位，损失为 next-token 交叉熵。

`configs/main.json` 对应 16 层、1024 隐藏维度、16 头、4096 FFN、1024 上下文、16000 词表。由于包含可学习位置嵌入及 bias，参数量与作业近似数值略有差别，主配置总参数量为 218,974,208，非嵌入参数量为 201,541,632；训练会输出并保存统计。

## 4. Part 3：主模型训练

```bash
python -m aml.train --config configs/main.json --data data/tinystories --output runs/main
```

主配置：AdamW，β=(0.9,0.95)，矩阵参数 weight decay=0.1；归一化和 bias 不衰减；峰值学习率 4e-4，80 步 warmup，cosine 衰减到 10%，梯度裁剪 1.0，CUDA BF16。

默认每个更新步的 token 数：`4 × 64 × 1024 = 262144`（micro batch × 累积次数 × 上下文长度）。4000 步约处理 **1.049B tokens**。报告中的主模型已完成该预算；随机采样不保证所有数据恰好遍历一次。

训练从磁盘映射 token 流随机抽取连续窗口（有放回），允许跨文档窗口，EOS 表示边界。因此累计 tokens 是计算预算，不等于无重复数据量。验证使用固定文档和固定分块，与随机训练窗口的上下文策略有所不同，报告应注明。

### 显存和设备

这是单设备版本。显式注意力需要存储注意力矩阵，先做短跑再确认主配置适合实际显存。OOM 时减少 `micro_batch_size`、相应增加 `gradient_accumulation` 保持有效 batch。两张共享 A100 可以分别运行不同任务；本实现不支持 `torchrun`/DDP，不应启动多个进程写同一输出目录。

CPU 测试用 `configs/smoke.json`；主配置的 BF16 需要 CUDA。`--device cpu` 本身不会改变 precision，CPU 运行时配置必须使用 fp32。

### 短跑与恢复

保留主配置的计划总步数，可先运行 10 步测速和检查：

```bash
python -m aml.train --config configs/main.json --data data/tinystories --output runs/main --stop-after 10
python -m aml.train --config configs/main.json --data data/tinystories --output runs/main --resume runs/main/last.pt
```

`--stop-after` 是绝对步数，不改变学习率衰减总时长。恢复检查配置和数据 metadata 一致，恢复优化器、梯度缩放器、各随机数状态和采样位置状态。不要修改数据文件后沿用旧 metadata。仅加载自己创建或可信来源的 checkpoint。

- `last.pt`：最近一次周期性保存，包含恢复所需状态。
- `best.pt`：固定验证子集上 loss 最低的 checkpoint，也包含恢复状态。
- `metrics.jsonl`：训练/验证指标、累计 tokens、LR、梯度范数、吞吐、近似 FLOPs。
- `config.json`、`model_info.json`、`data_metadata.json`：复现实验与报告所需信息。

默认每 100 步验证，每 100 步保存 last；验证改善会即时保存 best；正常停止总是验证并保存 last。异常中断从最近的 last 恢复，未保存部分需要重跑。吞吐日志是墙钟区间估算，可能包含验证/保存开销，不是纯 GPU kernel 测速。

主配置验证默认取前 128 篇完整文档，使用 `eval_documents` 修改，0 表示全量。比较多次训练时应使用相同 eval_documents。

### 最终评测、样例、曲线

```bash
python -m aml.evaluate --checkpoint runs/main/best.pt --data data/tinystories --output runs/main/final_metrics.json --precision bf16
python -m aml.generate --checkpoint runs/main/best.pt --data data/tinystories --output runs/main/samples.json --prompt "Once upon a time," --prompt "The little girl found" --prompt "One day, a dog"
python -m aml.plot --run runs/main
```

最终评测默认覆盖整个验证集，时间可能较长；`--max-documents` 可以先做检查，正式报告需标明是否全量。生成设置和随机种子保存在样例文件；temperature=0 表示 greedy。超出上下文长度时使用滑动窗口并重新编号位置，不使用 KV cache。

评测定义：每篇文档以 EOS 作为起始上下文，预测所有内容 token 和末尾 EOS。长文档按上下文长度切块，块开始只保留前一 token，统一用于所有验证。累计所有目标 token 的负对数似然 S 后：

- loss = S / 目标 token 数，单位 nats/token；
- PPL = exp(loss)；
- BPB = S / (ln(2) × 原始文本 UTF-8 字节数)。

EOS 的 NLL 计入损失，但 EOS 不贡献原始字节。这是明示的边界编码约定。补齐 batch 的 padding 不计入指标；NLL 与 bytes 使用完全相同的一组完整文档，不用平均 batch loss 代替全局加权均值。

## 5. 提交前检查

- Part 1：自制 BPE JSON、验证报告、参考分词器编码过程截图与 metadata。
- Part 2：解释模型各模块、掩码、参数量和共享权重。
- Part 3：配置、训练截图、curves.png/PDF、final_metrics.json、samples.json、最终 checkpoint。
- 保存实际运行命令、环境版本和异常处理记录；说明 AI 辅助范围。
- 最终报告按作业要求控制在 12 页内；主模型和 Part 4 实验均已完成，具体结果见报告。



## AI assistance

AI tools assisted with implementation, debugging, experiment execution, and report editing. The author directed the study, made decisions on experimental scope, and reviewed and revised the analysis and presentation.

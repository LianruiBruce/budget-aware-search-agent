# 开发记录（DEVLOG）

记录这个项目从搭建到第一次训练失败、排查和修复的全过程，包括走过的弯路。时间均为美东时间（EDT）。

开发方式：在 Claude Code 协助下完成，代码由 Claude 编写，需求、方向和取舍由我决定。面试前需要逐个文件读懂并能复现每个实验。

---

## 1. 项目目标

**问题**：搜索 Agent 每调用一次检索工具都有成本（延迟、API 费用、上下文长度）。现有的 ReAct 式 Agent 不考虑成本：要么搜得过多，要么搜一次就答。

**想法**：把"每次搜索的价格 λ"写进 prompt，用多轮 GRPO 训练一个策略，让它按价格调整搜索次数：便宜时多搜求准，贵时少搜或直接答。最终产出一条"平均搜索次数 vs F1"的 Pareto 曲线，同一个模型只改 prompt 里的 λ，就能在曲线上移动。

**奖励**：`F1(answer, gold) − λ · 搜索次数`；没给出答案记 `−0.1 − λ · 搜索次数`。

---

## 2. 环境

| 项 | 版本 / 配置 |
|---|---|
| GPU | RTX 4080 16GB，驱动 610.60 |
| 系统 | Windows 11，Git Bash |
| Python | 3.11.9（Microsoft Store 版），项目内 `.venv` |
| PyTorch | 2.11.0+cu128 |
| Transformers / PEFT | 5.20.1 / 0.21.2 |
| 检索 | bm25s + PyStemmer |
| 模型 | Qwen3-1.7B（`enable_thinking=False`），LoRA r=32，作用于全部投影层，可训练参数 34.9M（1.99%） |

---

## 3. 时间线

| 时间（10-18） | 事件 |
|---|---|
| 22:04 | 建项目目录、venv，后台安装依赖 |
| 22:08–22:10 | 写 `reward.py` / `data.py` / `retriever.py` / `prompts.py` / `rollout.py` / `train_grpo.py` |
| 22:10 | 构建数据：train 8000、dev 500，BM25 语料 75,194 段 |
| 22:11 | 冒烟测试：检索召回、8 条 greedy rollout |
| 22:12 | 写 `evaluate.py` |
| 22:12–22:15 | 3 步 GRPO 冒烟训练：约 40–55 s/步，显存约 3.3 GB |
| 22:15 | 启动"基线评测 + 400 步训练"，首次提交并推到私有仓库 |
| 22:35 | 发现基线结果异常（问题 4） |
| 22:48 | 发现训练塌缩（问题 5） |
| 22:49 | 停止训练，加 KL、降学习率 |
| 22:50 | 冒烟测试发现：未训练的模型本身采样 16/16 都只搜 1 次 |
| 22:52 | 加入强制分支探索，冒烟通过；重跑基线 + 300 步训练；第二次提交 |

---

## 4. 系统设计（简述）

- **数据**（`basa/data.py`）
  - 用 HotpotQA distractor 设定。
  - 检索语料是所选题目全部 context 段落（含干扰段）的并集，按标题去重。
  - 结果缓存到 `cache/`。
- **工具**（`basa/retriever.py`）：BM25 返回 top-3，每段截断到 400 字符。
- **Rollout**（`basa/rollout.py`）
  - 每个 episode 保存为一条 token 序列加一个 mask：模型生成的 token 的 mask 为 1，参与训练；prompt 和注入的 `<result>` 为 0。
  - 生成时用 `stop_strings=["</search>", "</answer>"]` 截停。
  - 批量生成时左侧 padding。
- **训练**（`basa/train_grpo.py`）
  - 同一道题的 G 条 rollout 共用同一个 λ，组内做奖励标准化得到 advantage。
  - 组内奖励方差为 0 的组直接丢弃。
- **评测**（`basa/evaluate.py`）
  - `no_search`：不允许搜索。
  - `rag1`：先用题目原文搜一次，之后不许再搜。
  - `react`：未训练模型加 Agent prompt。
  - `trained`：GRPO adapter。
  - 每个 λ 报告 EM、F1、平均搜索次数、utility（= F1 − λ·搜索次数）和未作答率。

---

## 5. 问题与修复

### 问题 1：Transformers 5 的 API 变化
- **现象**：`from_pretrained(..., torch_dtype=...)` 在 Transformers 5 中已更名为 `dtype`。
- **修复**：全部改为 `dtype=torch.bfloat16`。

### 问题 2：logits 显存隐患（写代码时预先处理）
- **风险**：Qwen3 词表约 15.1 万。一条 3072 token 的序列，float32 logits 就要约 1.9 GB；再做 `log_softmax`，16 GB 显存很容易爆。
- **修复**：`seq_logprobs` 先取 hidden states，再按 512 个位置一块过 `lm_head`，逐块算 log-prob，完整的 logits 不会同时驻留显存。
- **验证**：冒烟训练显存约 3.3 GB。

### 问题 3：`enable_input_require_grads` 重复注册
- **风险**：最初写在每步训练里调用，会重复注册 forward hook。
- **修复**：在建好 PEFT 模型后只调用一次。

### 问题 4：两个基线几乎全部未作答
- **现象**（`results/baselines_buggy_v0.jsonl`）：

  | 方法 | F1 | 未作答率 |
  |---|---|---|
  | no_search | 0.002 | 99.8% |
  | rag1 | 0.002 | 99.2% |
  | react | 0.292 | 1% |

- **排查**：GPU 被训练占用，于是在 CPU 上用 3 道题复现 rag1 的输出，得到：
  - `'The civil war ... was the Russian Civil War, which ended in 1922.'`
  - `'"Hey Pa! There's a Goat on the Roof" came to market first.'`

  模型答案内容基本对，但完全不写 `<answer>` 标签，所以解析为"未作答"。
- **根因**：
  1. **rag1**：用了另一套简短 system prompt，未训练的 Qwen3-1.7B 不遵守其中的标签要求。
  2. **no_search**：`max_calls=0` 时，模型仍然先尝试搜索，收到"预算用完"，但 rollout 循环只有 `max_calls + 1 = 1` 轮，没有剩余轮次让它作答。
- **修复**：
  1. rag1 改为在 Agent 自己的格式里**预填一次搜索**：以 `<search>题目原文</search><result>…</result>` 开头，计 1 次搜索，之后不许再搜。这样三个基线用同一套 prompt 和格式，对比才公平。
  2. rollout 循环改为 `max_calls + 2` 轮：每次允许的搜索一轮，再加一轮"预算用完后作答"。
- **修复后**（λ=0）：

  | 方法 | F1 | EM | 平均搜索 | 未作答率 |
  |---|---|---|---|---|
  | no_search | 0.133 | 0.088 | 0 | 0% |
  | rag1 | 0.343 | 0.236 | 1 | 0.6% |
  | react | 0.293 | 0.202 | 1.15 | 0.4% |

- **观察**：
  - 未训练的 Agent 自己决定怎么搜，结果反而比"用题目原文搜一次"更差（0.293 vs 0.343）。原因是它常用很泛的查询词，或重复同一个查询。
  - λ 从 0 变到 0.3，它的平均搜索次数只在 1.15–1.29 之间波动，**基本不理会价格**。这正是要用 RL 解决的问题。

### 问题 5：GRPO 训练塌缩到"永远只搜 1 次"
- **现象**：第一次训练（lr 5e-5，无 KL，`runs/collapsed_lr5e-5_nokl/`）中：
  - 前 6 步平均搜索次数在 1.0–1.6 之间波动；
  - 从第 8 步起，**所有 λ 下每一条 rollout 都恰好搜 1 次**，一直到第 28 步停止；
  - 每步 8 组中只有 1–5 组有奖励差异，其余组被丢弃。

  | 步 | 平均搜索 | 未作答率 | 有效组 |
  |---|---|---|---|
  | 1 | 1.03 | 0 | 3 |
  | 2 | 1.23 | 0.05 | 4 |
  | 3 | 1.62 | 0.08 | 5 |
  | 4 | 1.08 | 0.03 | 4 |
  | 8–28 | 1.00 | 0 | 1–5 |

- **根因（三个因素叠加）**：
  1. **未训练模型本身几乎没有多样性**。加了 KL 和低学习率后重新冒烟，第 1 步还没有任何更新，16 条采样（温度 1.0）**全部**恰好搜 1 次。组内搜索次数相同，奖励差异只来自答案对错，"该搜几次"这个决策拿不到任何梯度信号。
  2. **问题 4 的轮次 bug 惩罚了多搜**。上表中未作答只出现在第 2–4 步，正是搜索次数大于 1 的时候。旧循环只有 `max_calls + 1` 轮：搜满 4 次后再想搜，就只收到"预算用完"，没有下一轮作答，记 `−0.1 − 4λ`。所以多搜的轨迹被 bug 打成了最低分。GRPO 学到的"少搜"是 bug 造成的，不是奖励设计的结果。
  3. **lr 5e-5 加上没有 KL 约束**，把上面的偏好迅速固化。约 8 步后策略变成确定性的"搜 1 次"，之后不再探索。
- **修复**：
  1. 轮次 bug 已随问题 4 修复。
  2. **强制分支探索**：每组 8 条 rollout 中：
     - 1 条第一轮强制以 `<answer>` 开头（不搜直接答）；
     - 1 条在第一次结果返回后强制以 `<search>` 开头（再搜一次）；
     - 其余 6 条自由生成。

     强制的开头 token 的 mask 设为 1，作为一个 off-policy 动作参与训练。这样"搜还是答"这个决策本身会拿到 advantage：便宜的题上直接答得分更高，就会提高直接答的概率。
  3. **KL 约束**：对关掉 LoRA 的原模型计算逐 token KL，用 k3 估计 `exp(ref−lp) − (ref−lp) − 1`，系数 0.05。
  4. 学习率降为 1e-5。日志增加每步的搜索次数直方图和 KL。
- **验证**（2 步冒烟，4 组 × 4 条）：
  - 搜索次数直方图从 `{1: 16}` 变为 `{0: 4, 1: 8, 2: 2, 3: 1, 4: 1}`；
  - 有奖励差异的组从 0–1/4 变为 4/4；
  - 第 2 步 KL = 0.026。

### 问题 6：没有验证就启动正式实验（流程错误）
- **经过**：
  - 22:15 第一次正式训练（400 步）之前，只跑了 3 步冒烟测试；
  - 22:52 第二次正式训练（300 步）之前，只跑了 2 步冒烟测试。

  冒烟测试只能证明代码能跑、组内有差异，**不能证明模型学得到想要的行为**。两次都没有事先定成败标准，也没有对照文献的实验设置。
- **第二次训练的结果**：第 5 步时，除强制分支外的 48 条自由 rollout 全部又只搜 1 次；第 4 步 KL 从 0.03 跳到 0.33。强制分支只是让每组表面上有差异，自由生成部分仍然塌缩。训练在第 5 步被叫停。
- **对照文献后发现**（详见 `RELATED_WORK.md`）：
  - 检索语料太小且包含答案段，多搜几乎没有收益；
  - 缺少"模型自己会"的题；
  - 加法奖励容易被钻空子（OTC 报告过）；
  - "恰好 1 轮"是已知的塌缩点（AI-SearchPlanner 报告过）；
  - KL 比文献大 50 倍，每步题数比文献小 30–60 倍；
  - 1.7B 比所有已报告的设置都小。
- **以后的规矩**：
  1. 先对照文献确定设置；
  2. 再做离线诊断，确认有可学的空间；
  3. 然后跑带书面成败标准的短试验；
  4. 最后才跑正式实验。

### 环境问题：Microsoft Store 版 Python 的 AppData 虚拟化
- **现象**：用 `.venv` 的 Python 读写 `C:\Users\...\AppData\Roaming\...` 下的文件时报"路径不存在"，但 Bash 能正常看到这些文件。
- **根因**：venv 是从 Microsoft Store 版 Python 3.11 创建的，Store 应用对 `AppData\Roaming` 做了文件系统虚拟化。
- **影响与规避**：项目本身在 `D:\UserData\Projects` 下，不受影响。需要读 AppData 下的文件时，先复制到别处，或换用非 Store 版 Python。

---

## 6. 已知局限与待验证

- **强制分支有偏**：
  - 强制 token 不是当前策略采样出来的，按 on-policy 处理没有做 importance ratio 修正，所以梯度是有偏的。
  - KL 项也会惩罚这些低概率的强制 token。
  - 要看训练后的模型在**不强制**时，搜索次数是否随 λ 单调变化；评测本来就不使用强制分支。
  - 如果效果不好，备选方案：
    - 只对强制 token 之后的自由生成部分计算梯度；
    - 改用按 λ 分层的 SFT 冷启动。
- **组内奖励方差为 0 的组被直接丢弃**，浪费采样。可以考虑动态加采样，或降低组大小、增加题数。
- **BM25 召回有上限**：用题目原文查询时，top-3 里命中至少一个支撑段的比例是 90.2%，两个支撑段都命中的只有 32.6%。多跳题需要第二次带新实体的查询，这也是"多搜有用"的来源。
- **评测只在 500 道 dev 题上 greedy 解码**，还没做多 seed、置信区间。
- **轮次修复改变了评测设置**：rollout 多了一轮"预算用完后作答"，所以修复后的 react 结果与第一次的数字不完全可比（0.293 vs 0.292，差异很小）。

---

## 7. 当前状态（10-18 23:30 左右）

- 修复后的基线已跑完（`results/baselines.jsonl`）。
- 第二次正式训练在第 5 步叫停（见问题 6）。
- **暂停所有训练**。下一步按 `RELATED_WORK.md` 第 4 节执行：离线诊断 → 改设置 → 带成败标准的试跑 → 正式实验。

## 8. 复现

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m basa.data            # 下载 HotpotQA，构建语料缓存
.venv/Scripts/python scripts/smoke.py        # 检索召回 + 几条 rollout
.venv/Scripts/python -m basa.evaluate --methods no_search,rag1,react --out results/baselines.jsonl
.venv/Scripts/python -m basa.train_grpo --steps 300 --q_per_step 8 --group 8 --out runs/basa_qwen3_1.7b
.venv/Scripts/python -m basa.evaluate --methods trained --adapter runs/basa_qwen3_1.7b/step300 --out results/trained.jsonl
```

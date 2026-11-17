EVAL_DATASET = [
    # ==================== 1. AffectGPT (2501.16566v2.pdf) ====================
    {
        "question": "AffectGPT 这篇论文旨在解决多模态情感识别（MER）领域的什么核心痛点？",
        "ground_truth": "当前社区缺乏具有密集、描述性情感注释的大规模数据集，且现有 MLLM 模型将多模态融合完全交由大语言模型处理，这对于强调多模态特性的情感识别任务来说是不够充分的。",
    },
    {
        "question": "AffectGPT 构建的大规模情感描述数据集叫什么名字？其采用了什么特殊的标注策略？",
        "ground_truth": "数据集名为 MER-Caption（以及经过滤的 MER-Caption+）。其采用了 model-led, human-assisted（模型主导、人类辅助）的标注策略，结合了人类先验来指导描述生成和样本过滤。",
    },
    {
        "question": "AffectGPT 在数据过滤阶段（Low-level Filtering）主要排除了哪两类异常数据？",
        "ground_truth": "第一类是音频和视频不匹配的数据（使用 TalkNet 工具检测并剔除）；第二类是生成描述长度异常（位于高斯分布两端、可能因模型未成功生成而导致过短或过长的描述）的数据。",
    },
    {
        "question": "AffectGPT 模型设计中引入了什么核心操作来增强多模态集成，并将其移到了 LLM 外部？",
        "ground_truth": "引入了 pre-fusion（预融合）操作，将跨模态交互移到了语言模型外部，论文中具体探讨了基于 Q-Former 和基于 Attention 的两种预融合方式。",
    },
    {
        "question": "论文中提出的统一评估基准 MER-UniBench 涵盖了哪三个典型的多模态情感识别任务？",
        "ground_truth": "包含：(1) 细粒度情感识别（Fine-grained Emotion Recognition）；(2) 基本情感识别（Basic Emotion Recognition）；(3) 情感分析（Sentiment Analysis）。",
    },
    {
        "question": "在 AffectGPT 的初步实验中，作者最终选用哪两个模型分别作为音频 LLM（ALLM）和视频 LLM（VLLM）？",
        "ground_truth": "选用 SALMONN 作为音频 LLM 生成音频线索，选用 Chat-UniVi 作为视频 LLM 提取视觉线索，并结合 GPT-3.5 融合多模态线索。",
    },
    {
        "question": "为什么在 MER-Caption 数据集构建时，作者最终选用 GPT-3.5 来合并文本、音频和视频线索，而不是直接替换为开源 LLM？",
        "ground_truth": "因为 MER 中的多模态融合极其复杂，经常遇到模态冲突（不同模态之间出现不一致或矛盾）问题，这对大模型的推理能力提出了极高的要求，使用开源 LLM 会导致性能下降。",
    },
    {
        "question": "AffectGPT 在 MER-UniBench 基准测试中取得了怎样的主要性能提升？",
        "ground_truth": "与现有的 MLLMs 相比，AffectGPT 在各项主要指标上取得了显著提升，相比现有模型实现了超过 9% 的性能提升。",
    },
    {
        "question": "在 AffectGPT 的模型训练中，作者对 LLM 主干网络应用了什么技术来提高训练效率并提升性能？",
        "ground_truth": "仅对 LLM 分支应用了 LoRA（Low-Rank Adaptation）模块进行微调，并将 LoRA 的默认 rank 值设为 16，同时冻结了 LLM 和单模态编码器的权重。",
    },
    {
        "question": "论文中的用户研究（User Study）通过对比哪两个数据集的重叠样本，验证了 MER-Caption+ 标注策略的有效性？",
        "ground_truth": "通过随机选择 20 个重叠样本，将 MER-Caption+ 与基于模型的 MERR-Coarse 以及经人类过滤的 MERR-Fine 进行人工对比，验证了其描述的准确性。",
    },

    # ==================== 2. GPT-1 (original.pdf) ====================
    {
        "question": "这篇论文（Radford et al., OpenAI）提出的通用半监督学习框架包含哪两个主要训练阶段？",
        "ground_truth": "第一阶段是在大规模无标注文本语料库上进行生成式预训练（Generative Pre-training）；第二阶段是在每个具体任务的有标签数据上进行判别式微调（Discriminative Fine-tuning）。",
    },
    {
        "question": "论文中用于无监督预训练的模型架构采用了哪种神经网络结构？",
        "ground_truth": "采用了多层的 Transformer 解码器（Decoder-only Transformer）架构，利用多头自注意力机制和位置前馈层。",
    },
    {
        "question": "为什么论文选择 BooksCorpus 数据集而不是句子级别打乱的语料库来训练语言模型？",
        "ground_truth": "因为 BooksCorpus 包含长段的连续文本，这使得生成式模型能够学习并条件化处理长距离依赖关系（Long-range dependencies）。",
    },
    {
        "question": "在将预训练模型迁移到下游有标签任务时，论文采用了什么方法来处理如文本对或问答等结构化输入，而无需大幅修改模型架构？",
        "ground_truth": "采用了任务感知输入转换（Task-aware input transformations / 遍历式方法），将所有结构化的文本输入转换为模型可以处理的单个连续 Token 序列。",
    },
    {
        "question": "为什么在监督微调阶段加入语言建模（Language Modeling）作为辅助目标（Auxiliary Objective）是有益的？",
        "ground_truth": "因为它通过 (a) 改善监督模型的泛化能力，以及 (b) 加速收敛，从而帮助了模型学习。",
    },
    {
        "question": "在模型的输入转换中，针对文本蕴含（Textual Entailment）任务是如何拼接前提和假设的？",
        "ground_truth": "将前提（Premise）和假设（Hypothesis）Token 序列进行拼接，并在中间插入一个分隔符 Token（$）。",
    },
    {
        "question": "论文中的模型使用了什么特殊的激活函数？",
        "ground_truth": "使用了 GELU（Gaussian Error Linear Unit，高斯误差线性单元）激活函数。",
    },
    {
        "question": "论文在 12 个研究的任务中有多少个实现了对以往 SOTA（State-of-the-Art）的超越？",
        "ground_truth": "在研究的 12 个任务中，有 9 个任务显著超越了先前的最先进水平（State-of-the-art）。",
    },
    {
        "question": "论文消融实验中发现，如果不使用无监督预训练直接在目标任务上训练 Transformer，性能会产生多大幅度的下降？",
        "ground_truth": "缺乏预训练会损害所有任务的性能，平均得分相比完整模型下降了 14.8%。",
    },
    {
        "question": "论文对零样本行为（Zero-shot Behaviors）的分析表明，随着生成式预训练更新次数的增加，模型的零样本性能呈现什么趋势？",
        "ground_truth": "模型的零样本性能保持稳定并随着训练稳步上升，这表明生成式预训练支持学习各种任务相关的功能。",
    },

    # ==================== 3. EmoPrefer (2507.04278v4.pdf) ====================
    {
        "question": "EmoPrefer 这篇论文的核心研究目的是什么？",
        "ground_truth": "探索多模态大语言模型（MLLMs）在解码人类情感偏好（Human Emotion Preferences）方面的潜力，从而为描述性多模态情感识别（DMER）提供更具成本效益的自动化评估策略。",
    },
    {
        "question": "论文构建的第一个以人类情感为中心的偏好数据集叫什么名字？",
        "ground_truth": "EmoPrefer-Data。",
    },
    {
        "question": "EmoPrefer-Data 是通过如何选择样本来确保数据质量的？",
        "ground_truth": "通过选择两个描述性情感数据集（MERR-Fine 和 MER-Caption+）中重叠的视频样本（共 1,368 个），并仅保留三位专家标注者达成完全共识的偏好标注。",
    },
    {
        "question": "在 EmoPrefer-Bench 的评估指标中，“交换一致性（Swap Consistency）”旨在衡量模型什么方面的能力？",
        "ground_truth": "评估模型对输入顺序交换的鲁棒性，确保模型的偏好预测是基于描述内容而不是基于输入顺序（如 Description 1 和 Description 2 互换位置后预测结果应保持一致）。",
    },
    {
        "question": "EmoPrefer-Bench 提出了几种用于 MLLM 偏好预测的提示词策略？其中哪几种策略引入了外部 LLM？",
        "ground_truth": "总共提出了 4 种策略（S1 ~ S4）。其中 S3 和 S4 策略引入了外部 LLM（如 Qwen2.5-7B）来处理纯文本的第二步推理。",
    },
    {
        "question": "实验发现，使用更大的外部 LLM（如 Qwen3-14B）是否总能带来更好的情感偏好对齐效果？",
        "ground_truth": "不一定。统计分析表明，较小的 Qwen2.5-7B 在整体表现上通常优于 Qwen3-14B，这说明更大的 LLM 并不能保证更好地与人类情感偏好对齐。",
    },
    {
        "question": "在所有的评测 MLLM 中，哪款模型在二分类加权 F1 分数（WAF）和交换一致性上取得了最突出的开源模型表现？",
        "ground_truth": "Qwen2.5-Omni（在二分类 WAF 和交换一致性上均表现极为优异）。",
    },
    {
        "question": "论文中的模型基础众包（Model-Based Crowdsourcing）实验表明，通过聚合 Top-k 模型预测结果，对模型范围有什么建议？",
        "ground_truth": "实验表明将模型范围限制在开源模型（如 k=3 或 4）即可获得优异的众包提升，而限制在闭源模型则会导致明显的性能下降。",
    },
    {
        "question": "EmoPrefer 如何将偏好预测应用到实际的模型排名中？",
        "ground_truth": "结合偏好预测结果构建胜负矩阵 W，然后利用 Bradley-Terry 算法估计每个模型的相对优势得分（Advantage Score）并生成模型排名。",
    },
    {
        "question": "EmoPrefer-Data 在版权和许可方面遵循什么规范？",
        "ground_truth": "遵循 CC BY-NC 4.0 非商业许可协议，原始视频取自 MER2024 并获得原作者授权，同时对视频进行了卡通化处理以保护隐私。",
    },

    # ==================== 4. Mamba (2312.00752v2.pdf) ====================
    {
        "question": "传统的结构化状态空间模型（SSMs，如 S4）虽然计算高效且具长程建模能力，但在处理哪类数据时表现不如 Transformer？",
        "ground_truth": "处理离散且信息密集的数据（例如语言文本和 DNA 序列）。",
    },
    {
        "question": "传统 SSM 无法有效处理文本的核心根源是什么？",
        "ground_truth": "它们是线性时不变（LTI, Linear Time Invariant）模型，其参数在时间维度上保持恒定，缺乏根据输入内容进行动态选择（Selective）的能力。",
    },
    {
        "question": "Mamba 引入的选择机制（Selection Mechanism）主要将哪几个参数变成了输入（x）的函数？",
        "ground_truth": "将参数 Δ、B 和 C（以及隐式影响离散化矩阵 A_bar 和 B_bar 的参数）变成了输入依赖的函数。",
    },
    {
        "question": "既然选择机制破坏了时间不变性并使快速卷积计算失效，Mamba 是通过什么算法在硬件上实现高效计算的？",
        "ground_truth": "采用了一种硬件感知（Hardware-aware）的并行扫描（Parallel Scan）算法，通过内核融合（Kernel Fusion）和重计算（Recomputation）将扩展状态保留在快速 SRAM 中，避免了高带宽内存（HBM）的瓶颈。",
    },
    {
        "question": "Mamba 的架构设计（Mamba Block）相比于传统的 H3 或 Transformer 有什么简化创新？",
        "ground_truth": "它将状态空间层（SSM）与类似 Gated MLP（SwiGLU）的组件结合进一个单一且同质化堆叠的模块中，彻底去除了 Attention 模块甚至传统的独立 MLP 模块。",
    },
    {
        "question": "在定理 1 中，当状态维数 N=1、A=-1、B=1 且 Δ 通过 Softplus 激活时，选择性 SSM 的递推公式实际上对应了传统 RNN 的什么机制？",
        "ground_truth": "对应了经典的 RNN 门控机制（Gating Mechanism）。",
    },
    {
        "question": "在归纳头（Induction Heads）实验中，Mamba 在面对比训练长度长数千倍的测试序列时表现如何？",
        "ground_truth": "Mamba 展现出了完美的泛化和外推能力，能够在测试时完美泛化到 2^20（约 100 万）长度的序列（比训练长度长 4000 倍）。",
    },
    {
        "question": "在 Pile 数据集的语言模型预训练（Scaling Laws）中，Mamba 的表现与强 Transformer 基线（Transformer++）相比如何？",
        "ground_truth": "Mamba 是首个在相同参数量和计算预算下完全匹配强 Transformer++ 训练食谱性能的线性时间序列模型。",
    },
    {
        "question": "在人类基因组（HG38）长序列预训练的上下文扩展实验中，随着上下文长度增加至 1M，Mamba 与 HyenaDNA 的困惑度（Perplexity）表现有何不同？",
        "ground_truth": "Mamba 的困惑度随着上下文增加而持续平稳下降（性能变好），而 HyenaDNA 的困惑度反而随着序列长度增加而上升。",
    },
    {
        "question": "相比于同尺寸的 Transformer 模型，Mamba 在自回归生成时的吞吐量（Throughput）表现如何？",
        "ground_truth": "由于不需要 KV 缓存且具备线性时间复杂度，Mamba 的生成吞吐量比同尺寸 Transformer 高出约 5 倍。",
    },

    # ==================== 5. CmIR (2604.18460v1.pdf) ====================
    {
        "question": "论文指出多模态情感计算（MAC）当前面临什么样阻碍模型泛化能力的核心问题？",
        "ground_truth": "模型往往会从训练数据中学习到虚假关联（Spurious Correlations）而不是真正的因果关系，导致在面对分布偏移或有噪模态时性能急剧下降。",
    },
    {
        "question": "CmIR 框架的核心思想是将每个模态解耦成哪两个互补的成分？",
        "ground_truth": "(1) 因果不变表示（Causal Invariant Representation, Z_m^inv）；(2) 环境特定虚假表示（Environment-specific Spurious Representation, Z_m^spu）。",
    },
    {
        "question": "CmIR 的整体优化目标函数由哪三个核心约束组成？",
        "ground_truth": "(1) 不变性约束（Invariance Constraint, R_inv）；(2) 互信息约束（Mutual Information Constraint, R_dec）；(3) 重构约束（Reconstruction Constraint, R_rec）。",
    },
    {
        "question": "在缺乏显式环境标签（E）的情况下，CmIR 是如何模拟不同虚拟环境来进行不变性约束的？",
        "ground_truth": "通过对原始特征注入不同程度的加性高斯噪声（由环境相关系数 α^(e) 控制）来模拟虚拟环境。",
    },
    {
        "question": "CmIR 如何在计算上近似实现最小化不变表示与虚假表示之间的互信息（I(Z_m^inv; Z_m^spu)）？",
        "ground_truth": "通过在特征空间中强制执行正交性（零线性相关，即通过归一化特征的矩阵乘积接近对角阵）作为互信息最小化的计算代理。",
    },
    {
        "question": "重构约束（R_rec）在 CmIR 框架中起到了什么关键作用？",
        "ground_truth": "确保解耦后的因果和虚假表示完整保留了原始输入的所有信息，从而防止信息丢失并避免退化的解耦解。",
    },
    {
        "question": "在 CMU-MOSI（OOD）基准测试中，CmIR 的表现说明了什么？",
        "ground_truth": "证明了其在面对经历过模拟退火算法构造的大幅词语-情感分布偏移的测试集时，依然具有卓越的分布外鲁棒性（OOD Robustness）。",
    },
    {
        "question": "当对所有模态施加高斯噪声（噪声率从 10% 到 70%）时，CmIR 表现出怎样的抗噪特性？",
        "ground_truth": "CmIR 在各项指标（特别是 MAE）上均显著优于其他竞争基线，且随着噪声水平的增加，其性能优势变得更加显著。",
    },
    {
        "question": "消融实验表明，在去除不变性约束（R_inv）、互信息约束（R_dec）或重构约束（R_rec）中，哪项对模型性能损害最大？",
        "ground_truth": "去除不变性约束（R_inv）和互信息约束（R_dec）会导致最显著的性能下降，凸显了它们在捕获稳定因果机制和解耦虚假信息方面的核心地位。",
    },
    {
        "question": "通过 t-SNE 特征可视化，CmIR 提取的因果不变语言表示（Z^inv）展现出了什么特征？",
        "ground_truth": "不同情感类别（正向、负向、中性）的特征在特征空间中边界清晰、聚类良好（尤其是中性样本合理地分布在正负之间），而常规模型提取的特征则存在大量重叠。",
    },
]
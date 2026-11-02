# 时间估算
# FLOPs（Floating Point Operations，浮点运算次数）是衡量计算复杂度的常用指标，表示一个算法或模型在执行过程中所需的浮点加减乘除等运算的总次数。而训练模型本质上是进行浮点运算（FLOPs）。有一个经验公式：
# 总计算量 = 6 * 参数量 * Token 数量
# 为什么是 6 倍，前向传播是 2 * 参数量（乘法 + 加法），后向传播是前向传播的 2 倍
# 计算硬件算力，查阅 NVIDIA H100 的白皮书，其 FP16/BF16 的峰值算力约为 1979 TFLOPS（每秒万亿次浮点运算），但是要注意，这个值是 NVIDIA H100 GPU 在使用 FP16 或 BF16 数据类型、且启用结构化稀疏（Structured Sparsity）时可达到的理论最大计算吞吐量。对于我们训练普通的稠密模型（如标准的 LLaMA-1到LLaMA-3系列、Qwen3-0.6B/1.7B/4B/8B/14B/32B等），理论峰值减半，约 990 TFLOPS，上述 990 TFLOPS 是 H100 的理论峰值，但实际运行模型时，由于各种软硬件开销，你几乎不可能达到 100% 模型算力利用率 (MFU, Model FLOPs Utilization)，通常按 30%–60% 的利用率估算更现实。这里取 50% 用作后续估计

# 内存估算
# 很多初学者认为：8 张 H100（每张 80GB 显存），显存总共 640GB。参数采用 float32存储，一个参数 4 字节，所以 字节 = 1600亿参数（160B）？错！

# 训练时，显存里存的不只是参数 (Parameters)，还有：

# 梯度 (Gradients): 和参数一样大，需要 4 字节
# 优化器状态 (Optimizer States): 比如常用的 AdamW 优化器，需要存储每个参数的一阶矩估计（梯度的指数移动平均）和二阶矩估计（梯度平方的指数移动平均）
# 因此，每个参数在训练时大约占用 16 字节：

# 参数 (FP32): 4 bytes
# 梯度 (FP32): 4 bytes
# 优化器状态 (FP32): 8 bytes (2 个变量 × 4 bytes)

import torch
from torch import nn

def same_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # 从Python列表创建
x = torch.zeros(4, 8)  # 4x8的零矩阵
x = torch.ones(4, 8)   # 4x8的全1矩阵
x = torch.randn(4, 8)  # 4x8的正态分布随机数

# 分配但不初始化值（用于自定义初始化的值）
x = torch.empty(4, 8)  # 未初始化的4x8矩阵
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # 截断正态分布初始化，分布均值为0，标准差为1，截断范围，只保留[-2, 2]区间内的值，超出此范围的值会被重新采样

# 在 Transformer 中常见的 Rank 是 4 维
B = 32   # Batch size（批次大小）
S = 16   # Sequence length（序列长度）
H = 16   # Number of heads（注意力头数）
D = 64   # Hidden dimension per head（每个头的隐藏维度）
x = torch.zeros(B, S, H, D)

x = torch.tensor([[1., 2, 3], [4, 5, 6]])

# 这些操作不复制数据
y = x[0]           # 获取第0行
y = x[:, 1]        # 获取第1列
y = x.view(3, 2)   # 重塑为3x2
y = x.transpose(1, 0)  # 转置

# 验证是否共享存储
assert same_storage(x, y)  # same_storage检查底层存储是否相同

# 注意：修改x会影响y
x[0][0] = 100
assert y[0][0] == 100  # 值也被修改

# 转置后的张量是非连续的
x = torch.tensor([[1., 2, 3], [4, 5, 6]])
y = x.transpose(1, 0)
assert not y.is_contiguous()  # 非连续张量

# 非连续的张量，无法再次 view
# 尝试重塑会失败
try:
    y.view(2, 3)
except RuntimeError as e:
    assert "view size is not compatible with input tensor's size and stride" in str(e)

# 若需要再次 view，需要使用 contiguous 将张量做连续，但是会生成一个新的变量
y = x.contiguous().view(2, 3)
assert not same_storage(x, y)


# 逐元素操作
x = torch.tensor([1, 4, 9])

# 幂运算 (pow)，每个元素平方
assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))    

# 对每个元素开平方根
assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))      

# rsqrt 是 reciprocal of sqrt 的缩写，即对每个元素先开方再取倒数
# 浮点数一般更推荐用 torch.allclose 比较，避免误差导致误判。
assert torch.allclose(x.rsqrt(), torch.tensor([1, 1/2, 1/3])) 

assert torch.equal(x + x, torch.tensor([2, 8, 18])) # 每个元素自加
assert torch.equal(x * 2, torch.tensor([2, 8, 18])) # 每个元素乘以标量2
assert torch.equal(x / 0.5, torch.tensor([2, 8, 18])) # 每个元素除以标量0.5

# 实用操作：下三角矩阵（用于因果注意力掩码）
x = torch.ones(3, 3).tril()  # 创建一个3x3全1矩阵，然后取其下三角部分（含对角线）
# 结果：
# [[1, 0, 0],
#  [1, 1, 0],
#  [1, 1, 1]]
# 含义：第0个词只能看到第0个词，第1个词能看到第0、1个词，依此类推

x = torch.ones(3, 3).triu()   # 上三角（含对角线）
# [[1, 1, 1],
#  [0, 1, 1],
#  [0, 0, 1]]

y = torch.ones(3, 3).triu(1)  # 严格上三角（不含对角线），取反后即为因果掩码
# [[0, 1, 1],
#  [0, 0, 1],
#  [0, 0, 0]]


# 基本矩阵乘法，矩阵乘法是深度学习的基础。矩阵乘法是神经网络中最核心、最频繁的计算操作，无论是全连接层、卷积层还是注意力机制，其底层都离不开矩阵运算。一个标准的矩阵乘法形式为：一个 M×K 的矩阵乘以一个 K×N 的矩阵，得到一个 M×N 的结果矩阵。示例如下：
x = torch.ones(16, 32)  # 16x32矩阵
w = torch.ones(32, 2)   # 32x2权重矩阵
y = x @ w               # 结果：16x2矩阵
assert y.size() == torch.Size([16, 2])

# batch 标签指向堆叠起来的多个矩形，代表一个批次中包含的多个样本。sequence 标签指向每个矩形内部，代表每个样本可能包含的序列长度（例如，一个句子中的词元数量）。
x = torch.ones(4, 8, 16, 32) # 形状为 [批大小, 序列长度, 样本特征, 特征维度]
w = torch.ones(32, 2)        # 权重矩阵
y = x @ w                    # 矩阵乘法
assert y.size() == torch.Size([4, 8, 16, 2]) # 结果形状

# 使用 einops 代替人对于维度的记忆
from einops import einsum

# 使用 jaxtyping 定义两个张量
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)  
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)  

# 传统方式
z = x @ y.transpose(-2, -1)  # batch, sequence, sequence

# Einops方式
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
# 未在输出中命名的维度（hidden）会被自动求和

# 更通用的做法
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")


from einops import reduce

x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

# 传统写法
y = x.mean(dim=-1)  # 对最后一个维度 hidden 上求平均

# Einops 写法
y = reduce(x, "... hidden -> ...", "mean")


# 用 einops.rearrange 拆分/合并维度
# 假设你有一个维度 total_hidden = 8，它实际上是 heads=2 和 hidden1=4 的乘积（即 2 * 4 = 8），也就是 total_hidden 维度实际上是 heads×hidden1 的扁平化表示。
from einops import rearrange

# 情景：hidden维度实际上是heads×hidden1的扁平化表示
x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

# 拆分 total_hidden为 heads 和 hidden1
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)

# 接下来，对拆分后的维度做矩阵乘法：
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")


# 内存
# 内存由张量的元素个数（张量的形状）和每个元素的数据类型（元素占用的字节数）决定的
# 浮点类型：在 LLM 训练中，选择数据类型本质上是在做显存占用、计算速度与数值稳定性之间的权衡（Trade-off）。我们关注的大多数张量（参数、梯度、激活值、优化器状态）都以浮点数形式进行存储，让我们深入理解不同的浮点数据类型

# Float32
# 规格：占用 4 字节 (32 bits)。结构为：1位符号位 + 8位指数位 + 23位尾数位。
# 地位：PyTorch 的默认数据类型，也是科学计算领域的“黄金标准”。
# 优点：数值精度高，动态范围大，训练最稳定，几乎不会出现数值溢出问题。
# 缺点：对于大模型而言太“奢侈”。它占用的显存是 16位格式的两倍，且在现代 GPU（如 H100）上的计算吞吐量远低于低精度格式。
# 炼丹用途：通常用于存储参数的主副本 (Master Weights) 和 优化器状态 (Optimizer States)，以确保在梯度累积和参数更新时不会因为精度丢失而导致模型无法收敛。

# Float16
# 规格：占用 2 字节 (16 bits)。结构为：1位符号位 + 5位指数位 + 10位尾数位。
# 优点：显存占用比 FP32 减少一半，计算速度快。
# 致命缺陷：动态范围（Dynamic Range）太窄。
# 由于指数位只有 5 位，它无法表示非常小的数（会发生下溢 Underflow，直接变成 0）或非常大的数（会发生上溢 Overflow，变成 Infinity）。
# 例如：在 FP16 中，1e-8 这样的小数会被直接当作 0 处理，导致梯度消失。
# 用途：这是上一代 GPU（如 V100）混合精度训练的主流。为了解决溢出问题，必须使用复杂的损失缩放 (Loss Scaling) 技术。目前在 LLM 训练中正逐渐被 BF16 取代

# BFloat16
# 规格：占用 2 字节 (16 bits)。结构为：1位符号位 + 8位指数位 + 7位尾数位。
# 来源：由 Google Brain 专为深度学习设计。
# 设计逻辑：“要范围，不要精度”。
# 深度学习模型（尤其是神经网络）对小数点的后几位精度不敏感，但对数值的范围非常敏感。
# BF16 直接截断了 FP32 的尾数，但保留了和 FP32 相同的 8 位指数位。
# 优点：
# 拥有和 FP32 一样宽广的动态范围，不需要 Loss Scaling 也能稳定训练。
# 显存占用和 FP16 一样少。
# 在 A100/H100 等新硬件上计算速度极快。
# 用途：当前 LLM 训练的绝对主流选择。通常用于存储激活值 (Activations) 以及进行前向和反向传播的矩阵乘法计算。
# Percy 在课上解释说："bf16 牺牲了精度来换取范围。对于深度学习来说，范围比精度重要得多，因为数值稳定性的主要威胁是溢出/下溢，而不是尾数精度不够。

# FP8
# 规格：占用 1 字节 (8 bits)。
# 变体：
# E4M3：4位指数，3位尾数（精度稍高，范围稍小）。
# E5M2：5位指数，2位尾数（范围稍大，精度更低）。
# 优点：极致的显存压缩和计算吞吐量。
# 硬件限制：仅在 NVIDIA H100 及更新的架构（配合 Transformer Engine）上才被原生支持。
# 炼丹用途：目前主要用于推理阶段的量化 (Quantization)。虽然理论上可以用于训练（H100 支持），但由于精度极低，训练极不稳定，属于比较前沿的研究领域。


# FP4/NVFP4
# 2025 年，NVIDIA 开发了 NVFP4，每个值仅 4 bits！

# 可表示的 16 个值（只有这么多可能值，因为 2⁴ = 16）： -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6

# Percy 在课上强调了一个关键细节：NVFP4 不是单纯每个值都只有 4 比特的动态范围。"如果你天真地只用这 16 个值，是无法有效训练的。实际上每个值拥有 4 比特的自由度，但数据被划分成 block（块），每个块有一个共享的缩放因子（scale factor），可以整体放大或缩小。这样单个值实际拥有超过 4 比特的动态范围，但相邻值之间的比值仍然受限于 4 比特。"

# 用一个类比：每个 block 内你可以在 4 比特的精度范围内自由变化，而 block 的缩放因子让整个 block 的值可以移到不同的量级。代价是你不能让一个值非常大而它旁边的值非常小，因为它俩属于同一个 block，共享同一个缩放因子。

# 训练 vs 推理：Percy 特别区分了两者。

# 推理时：可以在 bf16 训练好模型后，将模型权重量化到 1 或 2 比特
# 训练时：用 1 比特训练语言模型要困难得多——Percy 直言"不认为有任何人做过可信的 1-bit 训练"
# 2026 年发布的 Nemotron 3 Super 是在 NVFP4 精度下训练的大型模型。

# 参考：Nemotron 3 Super (NVIDIA, 2026)

# Percy 也提到，fp4 这些底层精度操作实际上是在 NVIDIA 的软件栈中自动完成的，"不是你创建一个张量然后调用 tensor.fp4() —— 很多工作是在底层 under the hood 进行的，用户无法直接控制。"

# 不同精度的运算速度完全不同。Percy 特别强调，现在的 GPU 已经不太优化 fp32 了："如果你现在用 fp32 做训练，会发现真的非常非常慢，因为硬件优化的重点已经转向了 bf16 甚至 fp8。"

# 将张量（tensors）从CPU内存移动到GPU内存
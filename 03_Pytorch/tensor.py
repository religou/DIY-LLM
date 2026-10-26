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
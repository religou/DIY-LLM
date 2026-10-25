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
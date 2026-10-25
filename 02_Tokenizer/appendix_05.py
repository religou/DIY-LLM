import regex

# deepseek tokenizer中使用的经典正则表达式（简化版）
TOKENIZER_REGEX =  r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

# 压缩率计算
def get_compression_ratio(text: str, segments):
    byte_len = len(text.encode("utf-8"))
    token_count = len(segments)
    return byte_len / token_count if token_count > 0 else 1

# 词级分词器
class WordTokenizer:
    def __init__(self, pattern=r"\w+|."):
        """ 
        初始化词级分词器。
        pattern: 用于匹配词语的正则表达式。
        """
        self.pattern = pattern
        self.word2id = {}
        self.id2word = {}
        
    def build_vocab(self, texts):
        """
        根据训练文本建立词表
        """
        vocab = set()
        for text in texts:
            segments = regex.findall(self.pattern, text)
            vocab.update(segments)

        vocab = sorted(vocab)
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text):
        segments = regex.findall(self.pattern, text)
        return [self.word2id.get(seg, -1) for seg in segments], segments

    def decode(self, ids):
        return "".join(self.id2word.get(i, "<UNK>") for i in ids)

# 测试
if __name__ == "__main__":

    string = "It's so supercalifragilisticexpialidocious!👋👋"
    print("原始字符串：", string)

    # 使用基础正则分词（基于空格和标点切分）
    basic_segments = regex.findall(r"\w+|.", string)
    print("基础正则分词结果：")
    print(basic_segments)

    # 使用deepseek风格正则
    segments = regex.findall(TOKENIZER_REGEX, string)
    print(f"deepseek风格分词结果：{segments}")

    # 构建词表
    tokenizer = WordTokenizer(pattern=TOKENIZER_REGEX)
    tokenizer.build_vocab([string])

    print("词表大小：", len(tokenizer.word2id))

    # 编码
    ids, segs = tokenizer.encode(string)
    print(f"编码token IDs：{ids}")

    # 字节序列
    byte_tokens = [b for b in string.encode("utf-8")]
    print(f"UTF-8字节序列：{byte_tokens}")

    print(f"编码segments：{segs}")

    # 解码
    decoded = tokenizer.decode(ids)
    print("解码结果：", decoded)

    # 压缩率
    ratio = get_compression_ratio(string, segs)
    print("压缩率：", ratio)
# 比较字节级分词器、字符级分词器、BPE分词器的编码结果

# 字节级分词器
from collections import Counter


class ByteTokenizer:
    def __init__(self):
        pass

    def encode(self, text):
        """
          将输入文本编码为字节索引列表。
        """
        return list(text.encode("utf-8"))

    def decode(self, indices):
        """
          将字节级别的列表解码为文本。
        """
        return bytes(indices).decode("utf-8")

# 字符级分词器
class CharacterTokenizer:
    def __init__(self):
        pass

    def encode(self, text):
        """
          将输入文本编码为字符索引列表。
        """
        return [ord(c) for c in text]

    def decode(self, indices):
        """
          将字符级别的列表解码为文本。
        """
        return "".join([chr(i) for i in indices])


# 计算压缩比率
def compression_ratio(text: str, token_len: int):
    input_bytes = len(text.encode("utf-8"))
    return input_bytes / token_len if token_len > 0 else 1


# BPE 分词器
class BPETokenizer:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.merges = {}  # {(a,b): new_token_id}
        self.vocab_size = 256  # 从byte开始

    def get_stats(self, tokens):
        pairs = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += 1
        return pairs

    def merge_tokens(self, tokens, pair, new_token):
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
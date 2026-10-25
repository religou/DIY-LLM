import regex
from collections import Counter

# DeepSeek风格正则
DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

# 使用grapheme cluster保持emoji不被拆分
def split_graphemes(token):
    return tuple(regex.findall(r'\X', token))

# BPE训练函数
def train_bpe(texts, num_merges=50):
    """
    texts: 文本列表（用于训练BPE）
    num_merges: BPE 迭代合并的次数
    """
    # 1.构建初始vocab（字符级+</w>结束符）
    vocab = Counter()
    for text in texts:
        tokens = regex.findall(DEEPSEEK_REGEX, text)
        for token in tokens:
            chars = split_graphemes(token) + ('</w>',)
            vocab[chars] += 1
    merges = []
    for _ in range(num_merges):
        # 统计相邻pair出现次数
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += freq
        if not pairs:
            break

        # 找到最常见pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # 合并所有vocab中的该pair
        new_vocab = {}
        for word, freq in vocab.items():
            w = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    w.append(word[i]+word[i+1])
                    i += 2
                else:
                    w.append(word[i])
                    i += 1
            new_vocab[tuple(w)] = freq
        vocab = new_vocab
    return merges, vocab

# BPE Tokenizer类
class BPETokenizer:
    def __init__(self, merges):
        self.merges = merges

    def encode_word(self, token):
        # 初始分成字符+</w>
        word = list(split_graphemes(token)) + ['</w>']
        # 按merge顺序依次合并
        for pair in self.merges:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == pair:
                    new_word.append(word[i]+word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text):
        tokens = regex.findall(DEEPSEEK_REGEX, text)
        bpe_tokens = []
        for t in tokens:
            bpe_tokens.extend(self.encode_word(t))
        return bpe_tokens

    def decode(self, tokens):
        # 拼接tokens并去掉结尾</w>
        text = ''.join(tokens).replace('</w>', '')
        return text

# 测试
if __name__ == "__main__":
    train_texts = ["这只猫🐈很可爱", "the quick brown fox jumps over the lazy 🐕‍🦺"]
    merges, vocab = train_bpe(train_texts, num_merges=20)
    print("BPE合并:", merges)
    tokenizer = BPETokenizer(merges)
    test_text = "敏捷的棕色狐狸🦊"
    encoded = tokenizer.encode(test_text)
    print("编码:", encoded)
    decoded = tokenizer.decode(encoded)
    print("解码:", decoded)
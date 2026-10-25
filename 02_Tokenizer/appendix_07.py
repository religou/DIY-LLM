"""
DeepSeek-V3 Tokenizer简易实现示例
（核心包含：字节级BPE+DeepSeek-V3正则预分词）
"""
import regex as re
from collections import Counter
from typing import List, Tuple, Dict, Iterable
import json
import base64


# 配置：DeepSeek-V3 预分词正则
# 来源：https://huggingface.co/deepseek-ai/DeepSeek-V3 tokenizer.json 中的 pre_tokenizer
# 官方是三个 Split(behavior="Isolated") 依次执行，不是一条正则
DEEPSEEK_SPLIT_PATTERNS = [
    # 1. 数字每 1~3 位切一块：2024 -> 202 / 4
    r"\p{N}{1,3}",
    # 2. 连续的 CJK 汉字(4E00-9FA5) / 平假名(3040-309F) / 片假名(30A0-30FF) 整块切出
    r"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    # 3. GPT 风格主正则
    #    [ASCII标点][A-Za-z]+                 如 's  .com
    #    [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+  可带一个前导字符(通常是空格)的单词，如 ' world'
    #     ?[\p{P}\p{S}]+[\r\n]*               可带前导空格的标点/符号串，可跟换行
    #    \s*[\r\n]+                           换行(连同前面的空白)
    #    \s+(?!\S)                            后面不接非空白的空白
    #    \s+                                  其余空白
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"
    r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
]
_DEEPSEEK_SPLITS = [re.compile(p) for p in DEEPSEEK_SPLIT_PATTERNS]


# 基础函数：预分词与字节处理
def _split_isolated(pieces: List[str], pattern) -> List[str]:
    """
    模拟 HF tokenizers 的 Split(behavior="Isolated")：
    匹配到的部分单独成块，未匹配的部分也保留，交给下一步继续切。
    """
    out = []
    for piece in pieces:
        last = 0
        for m in pattern.finditer(piece):
            if m.start() > last:
                out.append(piece[last:m.start()])
            if m.end() > m.start():
                out.append(m.group())
            last = m.end()
        if last < len(piece):
            out.append(piece[last:])
    return out

def pretokenize(text:str):
    """按DeepSeek-V3的三步正则进行预分词"""
    pieces = [text]
    for pattern in _DEEPSEEK_SPLITS:
        pieces = _split_isolated(pieces, pattern)
    return pieces

def bytes2tokens(b:bytes):
    """
    将UTF-8字节序列转为latin1可表示的token列表。
    每个字节0–255都能被latin1接映射到字符。
    """
    return [bytes([x]).decode('latin1') for x in b]

def tokens2bytes(tokens):
    """将latin1 token列表重新转回原始bytes"""
    return b''.join([t.encode('latin1') for t in tokens])


# BPE训练相关
def build_corpus(texts):
    """
    构建byte-level语料。
    步骤：预分词 → UTF-8编码 → 分解为单字节 → 作为初始token序列。
    """
    corpus = []
    for text in texts:
        for chunk in pretokenize(text):
            corpus.append(bytes2tokens(chunk.encode('utf-8')))
    return corpus

def pair_freq(corpus: List[List[str]]):
    """统计所有token序列中相邻token pair的出现频率"""
    pairs = Counter()
    for word in corpus:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def merge_pair(word: List[str], pair: Tuple[str,str]):
    """将指定的token pair合并成一个token"""
    a, b = pair
    merged = []
    i = 0
    while i < len(word):
        if i < len(word)-1 and word[i]==a and word[i+1]==b:
            merged.append(a+b)   # 合并为一个新token
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged

def train_bpe(texts: Iterable[str], vocab_size: int=5000, num_merges: int=None) -> Tuple[List[Tuple[str,str]], List[str]]:
    """
    训练字节级BPE
    """
    corpus = build_corpus(texts)
    base_tokens = [bytes([i]).decode('latin1') for i in range(256)]
    merges: List[Tuple[str,str]] = []
    merged_set = set()
    cur_vocab_size = 256

    # 若未指定合并次数，则由target vocab来决定
    merge_steps = num_merges or (vocab_size - 256)

    for _ in range(merge_steps):
        pfreq = pair_freq(corpus)
        if not pfreq:
            break

        # 找到出现频率最高的pair
        best_pair, _ = pfreq.most_common(1)[0]

        if cur_vocab_size + 1 > vocab_size:
            break

        merges.append(best_pair)

        # 对整个语料进行合并替换
        corpus = [merge_pair(word, best_pair) for word in corpus]

        # 将新token记入词表
        merged_set.add(best_pair[0]+best_pair[1])
        cur_vocab_size += 1

    # 追加特殊token
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # vocab = 特殊token+ 256 byte token +BPE合并的新token
    vocab_tokens = special_tokens + base_tokens + sorted(merged_set)

    return merges, vocab_tokens



# Tokenizer类
class DeepSeekV3Tokenizer:
    def __init__(self, merges: List[Tuple[str,str]], vocab_tokens: List[str]):
        self.merges = merges
        self.vocab_tokens = vocab_tokens

        # token ↔ id映射
        self.token2id = {tok:i for i, tok in enumerate(vocab_tokens)}
        self.id2token = {i:tok for tok,i in self.token2id.items()}

        # merges pair → 排序index
        self.ranks = {pair:i for i,pair in enumerate(merges)}

        # 特殊token
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def encode_chunk(self, chunk: str) -> List[str]:
        """
        对一个预分词做BPE编码：
        - 转字节token
        - 逐步应用merges
        - 处理OOV：未知token拆回字节或标记为<unk>
        """
        tokens = bytes2tokens(chunk.encode('utf-8'))

        # 应用PE并规则
        for pair in self.merges:
            new_tokens = []
            i = 0
            a,b = pair
            while i < len(tokens):
                if i<len(tokens)-1 and tokens[i]==a and tokens[i+1]==b:
                    new_tokens.append(a+b)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens = new_tokens

        # OOV token拆回字节
        out = []
        for t in tokens:
            if t in self.token2id:
                out.append(t)
            else:
                # 拆分成字节token，如果字节token也不在词表 → <unk>
                out.extend([ch if ch in self.token2id else self.unk_token for ch in t])
        return out

    def encode(self, text: str, add_bos=False, add_eos=False, print_chunks=False):
        """
        编码完整文本：
        - 先预分词
        - 再逐chunk编码
        - 可选打印中间过程
        """
        ids = []

        if add_bos:
            ids.append(self.token2id[self.bos_token])
            if print_chunks: print(f"[Special] <bos> -> {self.token2id[self.bos_token]}")

        for chunk in pretokenize(text):
            toks = self.encode_chunk(chunk)
            chunk_ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in toks]

            if print_chunks:
                readable = []
                for t in toks:
                    try:
                        # 尝试恢复utf-8
                        r = tokens2bytes([t]).decode('utf-8', errors='ignore')
                        readable.append(r if r else t.encode('latin1').hex())
                    except:
                        readable.append(t.encode('latin1').hex())

                print(f"[Chunk] \"{chunk}\" -> {readable} -> IDs: {chunk_ids}")

            ids.extend(chunk_ids)

        if add_eos:
            ids.append(self.token2id[self.eos_token])
            if print_chunks: print(f"[Special] <eos> -> {self.token2id[self.eos_token]}")
        return ids

    def decode(self, ids: Iterable[int]):
        """
        将ID序列还原为utf-8文本：
        """
        byte_seq = bytearray()
        for i in ids:
            tok = self.id2token.get(i, self.unk_token)
            if tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            byte_seq.extend(tokens2bytes(list(tok)))
        return byte_seq.decode('utf-8', errors='replace')

    def save(self, vocab_path: str, merges_path: str):

        # 保存vocab（token2id）
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

        # 保存merges：每个token用base64
        merges_b64 = []
        for a, b in self.merges:
            a_bytes = a.encode('latin1')
            b_bytes = b.encode('latin1')
            merges_b64.append((
                base64.b64encode(a_bytes).decode('ascii'),
                base64.b64encode(b_bytes).decode('ascii')
            ))

        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_b64, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, vocab_path: str, merges_path: str):

        # 加载vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            token2id = json.load(f)
        vocab_tokens = [None] * (max(token2id.values()) + 1)
        for tok, idx in token2id.items():
            vocab_tokens[idx] = tok

        # 加载merges（base64 → bytes → latin1）
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_b64 = json.load(f)

        merges = []
        for a_b64, b_b64 in merges_b64:
            a = base64.b64decode(a_b64).decode('latin1')
            b = base64.b64decode(b_b64).decode('latin1')
            merges.append((a, b))
        return cls(merges, vocab_tokens)


# 提供训练函数
def train_tokenizer(texts, vocab_size=5000, num_merges=None):
    merges, vocab_tokens = train_bpe(texts, vocab_size=vocab_size, num_merges=num_merges)
    return DeepSeekV3Tokenizer(merges, vocab_tokens)

# 示例
if __name__ == "__main__":
    texts = [
        "Transformer是AI的核心技术。",
        "DeepSeek分词器支持中文、英文、emoji等多语言。",
        "Hello, 世界! 🌍🚀",
    ]

    print("训练 Tokenizer (vocab_size=1024)")
    tokenizer = train_tokenizer(texts, vocab_size=1024)
    print(f"完成训练，词表大小: {len(tokenizer.vocab_tokens)}")
    print("-"*50)

    txt = "注意力机制是AI的核心技术。 🚀 🚀"
    print(f"编码文本: {txt}")
    ids = tokenizer.encode(txt, add_bos=True, add_eos=True, print_chunks=True)

    print("-"*50)
    print("Token ID:", ids)
    decoded = tokenizer.decode(ids)
    print("解码结果:", decoded)
    print("是否可逆:", decoded == txt)
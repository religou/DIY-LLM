# 字符级分词器
# 输入：hi，很好的，terrific！🐋
# 输出：编码ID: [104, 105, 65292, 24456, 22909, 30340, 65292, 116, 101, 114, 114, 105, 102, 105, 99, 65281, 128011]
# 压缩比率: 0.47058823529411764

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

if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    text = "hi，很好的，terrific！🐋"
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
    
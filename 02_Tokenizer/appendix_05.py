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
        
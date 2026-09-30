# 基于Unicode的划分，会将一句话中的字符类型划分到同一个token中，常用于多语言混合文本
# 输入：Hello👋👋，DataScience成立于2018年！！！
# 输出：['Hello', '👋👋', '，', 'DataScience', '成立于', '2018', '年', '！！！']

import unicodedata
from typing import List

def get_char_category(ch: str) -> str:
    # 获取Unicode标准定义的分类（如'Lu'代表大写字母,'Po'代表其它标点）
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return "PUNCT"

    # 判定是否为中文字符
    if '\u4e00' <= ch <= '\u9fff':
        return "CJK"

    # 判定是否为数字，需要注意上角标数字也会返回 True
    if ch.isdigit():
        return "DIGIT"

    # 判定是否为字母
    if ch.isalpha():
        return "ALPHA"

    # 其余字符（如 Emoji、空格、控制符等）统一归为 OTHER
    return "OTHER"

    
def segment_by_unicode_category(text: str) -> List[str]:
    if not text:
        return []

    segments = []
    # 初始化缓存区，放入第一个字符
    buffer = [text[0]]
    # 获取第一个字符的类别，作为后续字符类别的参考
    prev_type = get_char_category(text[0])

    for ch in text[1:]:
        curr_type = get_char_category(ch)

        # 如果当前字符的类别与上一个字符的类型相同，则存入缓存区合并
        if curr_type == prev_type:
            buffer.append(ch)
        else:
            # 如果当前字符的类别与上一个字符的类型不同，则将缓存区内容作为一个token加入结果列表
            segments.append(("".join(buffer), prev_type))
            # 重置缓存区，开始记录新的字符
            buffer = [ch]
            prev_type = curr_type
    
    # 处理最后一个留着缓存区里面的字符
    segments.append(("".join(buffer), prev_type))

    tokens = [seg for seg, _ in segments]
    return tokens

if __name__ == "__main__":
    text = "Hello👋👋，DataScience成立于2018年！！！"
    result = segment_by_unicode_category(text)
    print(f"原始文本：{text}")
    print(f"分段结果：{result}")

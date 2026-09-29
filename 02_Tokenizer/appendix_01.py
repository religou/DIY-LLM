# 根据 NER 技术对数据脱敏处理
# 处理前：小明的邮箱是test111@gmail.com，电话是13312311111，现在居住于重庆两江新区小区。
# 处理后：[NAME]的邮箱是[EMAIL]，电话是[PHONE]，现在居住于[PLACE]

import re
from typing import List, Callable
from transformers import pipeline

# 初始化命名实体识别（NER）的流水线
ner_pipeline = pipeline(
    "ner", 
    model="ckiplab/bert-base-chinese-ner",
    aggregation_strategy="simple"
)


def ner_mask(text: str) -> str:
    """
    利用深度学习模型进行语义级别的脱敏（人名和地名）
    """
    entities = ner_pipeline(text)
    spans = []

    for ent in entities:
        label = ent["entity_group"]
        start = ent["start"]
        end = ent["end"]
        if label == "PERSON":
            spans.append((start, end, "[NAME]"))
        if label == "LOC":
            spans.append((start, end, "[PLACE]"))

    spans.sort(key = lambda x: (x[0], -(x[1] - x[0])))

    filtered_spans = []
    last_end = -1
    # 去除重叠或者包含关系的实体区间
    for start, end, tag in spans:
        if start >= last_end:
            filtered_spans.append((start, end, tag))
            last_end = end

    # 根据过滤后的区间重建文本
    result = []
    last_idx = 0
    for start, end, tag in filtered_spans:
        result.append(text[last_idx:start])
        result.append(tag)
        last_idx = end
    result.append(text[last_idx:])

    return "".join(result)


# 脱敏流水线架构设计
class DesensitizationPipeline:
    """
    按照顺序添加多个处理步骤
    """
    def __init__(self):
        self.steps: List[Callable[[str], str]] = []

    def add_step(self, func: Callable[[str], str]):
        self.steps.append(func)

    def run(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text

# 具体脱敏的方法
def normalize_text(text: str) -> str:
    """文本预处理：去除首尾空格"""
    return text.strip()

def mask_phone(text: str) -> str:
    """正则匹配 11 位中国手机号"""
    return re.sub(r'1[3-9]\d{9}', '[PHONE]', text)

def mask_email(text: str) -> str:
    """正则匹配常见邮箱格式"""
    return re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)

def mask_address(text: str) -> str:
    """正则匹配地址"""
    return re.sub(r'(居住于|现居住于|现居于|地址)([\u4e00-\u9fa5A-Za-z0-9]+)',
        r'\1[PLACE]', text)

def mask_name(text: str) -> str:
    """
    兜底策略：匹配出现在句首或标点后的“某某某的”结构
    注：容易误伤，通常放在 NER 步骤之后作为补充
    """
    return re.sub(
        r'(?:(?<=^)|(?<=[，。！？]))([\u4e00-\u9fa5]{2,3})(的)',
        r'[NAME]\2',
        text
    )

def clean_punctuation(text: str) -> str:
    """后处理环节：可根据需求规范化标点符号"""
    return text

# 构建流水线
def build_pipeline():
    """
    组装流水线：建议遵循“预处理 -> 高准确率正则 -> AI识别 -> 低准确率正则兜底”的顺序
    """
    p = DesensitizationPipeline()

    # 基础清理
    p.add_step(normalize_text)

    # 静态规则（手机、邮箱这类模式固定的最先处理，防止被NER误切）
    p.add_step(mask_phone)
    p.add_step(mask_email)

    # AI语义识别（主力：处理复杂的人名、地名）
    p.add_step(ner_mask)

    # 动态规则兜底（针对模型可能漏掉的特定话术）
    p.add_step(mask_address)
    p.add_step(mask_name)

    # 收尾处理
    p.add_step(clean_punctuation)

    return p

if __name__ == "__main__":
    # 测试用例：包含人名、邮箱、电话、地名及详细地址
    test_text = "小明的邮箱是test@gmail.com，电话是13312311111，现在居住于重庆两江新区的xxx小区。"
    
    ds_pipeline = build_pipeline()

    print("--- 脱敏系统测试 ---")
    print("处理前:", test_text)
    print("处理后:", ds_pipeline.run(test_text))

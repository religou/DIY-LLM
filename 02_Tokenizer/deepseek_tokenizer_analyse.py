from transformers import AutoTokenizer

MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"成功加载模型：{MODEL_NAME}的分词器。")
print(f"分词器词表大小V:{len(tokenizer.get_vocab())}")


chinese_text = "注意力机制是AI的核心技术"
encoded_ids = tokenizer.encode(chinese_text, add_special_tokens = False)
tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
print(f"原文：{chinese_text}")
print(f"ids：{encoded_ids}")
print(f"tokens：{tokens}")
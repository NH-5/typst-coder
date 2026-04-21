# test code for loading model from local path
# 从本地目录加载模型的测试代码
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_PATH = Path(__file__).parent.parent
MODEL_PATH = PROJECT_PATH / "model/qwen3.5-0.8b-Base"
IMAGE_PATH = PROJECT_PATH / "test/test_image.jpeg"

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH)

# Some Qwen3.5 model exports only attach the chat template to the tokenizer.
if processor.chat_template is None and getattr(processor.tokenizer, "chat_template", None):
    processor.chat_template = processor.tokenizer.chat_template

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": str(IMAGE_PATH),
            },
            {
                "type": "text",
                "text": "图里是什么",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
print(processor.decode(generated_ids, skip_special_tokens=True).strip())

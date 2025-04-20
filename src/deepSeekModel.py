import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt):
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Cargar modelo y tokenizer (solo la primera vez)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=2
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

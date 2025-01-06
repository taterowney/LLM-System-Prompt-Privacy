import torch
from transformers import AutoTokenizer

from main import get_device, model_name, r, lora_alpha, lora_dropout

device = get_device()

model = torch.load("lora_finetuned_llama.pth", map_location=device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def infer(prompt, max_length=128):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(infer("What is your name?"))

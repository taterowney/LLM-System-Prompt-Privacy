import torch
from transformers import AutoTokenizer

from fine_tune_test import get_device, model_name, r, lora_alpha, lora_dropout, create_model


device = get_device()

model, tokenizer = create_model()
model.to(device)
model.load_state_dict(torch.load("lora_finetuned_llama.pt", weights_only=True))

def infer(prompt, max_length=128):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(infer("What is your name?"))

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Non-streaming version for now
class SQuAD_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.ds = load_dataset("rajpurkar/squad", split=split, streaming=False)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        example = self.ds[item]
        context = example["title"] + "\n" + example["context"] + "\n\n" + example["question"]
        return self.tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": context}, {"role": "assistant", "content":example["answers"]["text"][0]}], tokenize=False)


# TODO: fix streaming of dataset
# def get_SQuAD_dataset(tokenizer, split="train"):
#     def transform(example):
#         print(example)
#         context = example["title"][0] + "\n" + example["context"][0] + "\n\n" + example["question"][0]
#         return tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": context}, {"role": "assistant", "content":example["answers"][0]["text"][0]}], tokenize=False, add_generation_prompt=False)
#
#     # ds = load_dataset("rajpurkar/squad", split=split, streaming=True).with_format("torch").set_transform(transform)
#     ds = load_dataset("rajpurkar/squad", split=split)
#     ds.set_transform(transform)
#     return ds

def get_SQuAD_dataloader(tokenizer, split="train", batch_size=4):
    ds = SQuAD_Dataset(tokenizer, split=split)
    dl = DataLoader(ds, batch_size=batch_size)
    return dl

if __name__ == "__main__":
    from LoRA import get_device, HF_model_with_LoRA
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    device = get_device()
    model, tokenizer = HF_model_with_LoRA(model_name, device=device)

    ds = SQuAD_Dataset(tokenizer, split="train")
    for example in ds:
        print(example)
        break
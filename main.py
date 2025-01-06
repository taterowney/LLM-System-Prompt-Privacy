import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import loralib as lora
from huggingface_hub import login

# Hyperparameters
# model_name = 'meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8'
model_name = 'meta-llama/Llama-3.2-1B'
# model_name = 'meta-llama/Llama-3.2-3B'

learning_rate = 1e-4
batch_size = 4
num_epochs = 5
max_length = 128
r = 8
lora_alpha = 32
lora_dropout = 0.05

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

def authenticate():
    login(token=input("Downloading models, enter HuggingFace API token: "))

# authenticate()
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except:
    authenticate()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)

print(f"Model {model.name_or_path} loaded successfully.")

tokenizer.pad_token = tokenizer.eos_token

# Replace Linear Layers with LoRA

def replace_linear_with_lora(module, r=8, lora_alpha=32, lora_dropout=0.05):
    for name, child in list(module.named_children()):
        # Recursively go deeper
        replace_linear_with_lora(child, r, lora_alpha, lora_dropout)

        if isinstance(child, torch.nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None

            new_lora_layer = lora.Linear(
                in_features, out_features,
                r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias=bias
            )
            # Copy weights
            new_lora_layer.weight = child.weight
            if bias:
                new_lora_layer.bias = child.bias

            setattr(module, name, new_lora_layer)

replace_linear_with_lora(model, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)


# Create a Dataset/Dataloader
class DummyDataset(torch.utils.data.Dataset):
    """
    Example dataset that returns random text tokens for demonstration.
    """
    def __init__(self, tokenizer, num_samples=100):
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = "What is your name? My name is Jeff."
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0)

dataset = DummyDataset(tokenizer, num_samples=500)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Set Up Optimizer
# Freeze all non-LoRA parameters by default
for param in model.parameters():
    param.requires_grad = False

# Only LoRA parameters are trainable
trainable_params = []
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True
        trainable_params.append(param)

optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

print("Model loaded! Beginning training...")

model.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # typical causal LM approach
        )
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "lora_finetuned_llama.pt")

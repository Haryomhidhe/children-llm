import torch
from llm_start import SmallGPT, ModelConfig, CharacterTokenizer

tokenizer = CharacterTokenizer()
torch.manual_seed(42)

with open("data.txt", "r", encoding="utf-8") as file:
  text =file.read()

  tokens =  tokenizer.encode(text)

  print("Total characters:", len(text))
  print("Total tokens:", len(tokens))

split_index = int(0.9*len(tokens))
train_tokens = tokens[:split_index]
val_tokens = tokens[split_index:]

print("Train tokens:", len(train_tokens))
print("Val tokens:", len(val_tokens))

import random 

context_length = 128
batch_size = 16

def get_batch(data):
    inputs = []
    targets = []
    for _ in range (batch_size):
      start = random.randint(0, len(data) - context_length - 1)
      chunk = data[start : start + context_length + 1]
      inputs.append(chunk[:-1])
      targets.append(chunk[1:])
    return torch.tensor(inputs), torch.tensor(targets)

config = ModelConfig()
model = SmallGPT(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
best_val_loss = float('inf')
number_of_steps =2000

for step in range (number_of_steps):
  inputs, targets = get_batch(train_tokens)
  
  logits, loss = model(inputs, targets)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if step % 100 == 0:
   val_inputs, val_targets = get_batch(val_tokens)
   with torch.no_grad():
     _, val_loss = model(val_inputs, val_targets)
   print(f"step {step}, loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")
   if val_loss.item() < best_val_loss:
    best_val_loss = val_loss.item()
    torch.save(model.state_dict(), "best_model.pt")
   if step % 1000 == 0:
    torch.save(model.state_dict(), f"checkpoint_step{step}.pt")

torch.save(model.state_dict(),"model.pt")
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

def generate(prompt, length=200):
  tokens_so_far = tokenizer.encode(prompt)
  for _ in range(length):
    input_tensor = torch.tensor([tokens_so_far[-context_length:]])
    logits, _ = model(input_tensor)
    next_token_logits = logits[0,-1]
    temperature = 0.8
    next_token_logits = next_token_logits / temperature
    probabilities = torch.softmax(next_token_logits, dim=0)
    next_token = torch.multinomial(probabilities, num_samples=1).item()
    tokens_so_far.append(next_token)
  return tokenizer.decode(tokens_so_far)

model.eval()

print("FINAL MODEL:")
print(generate("The sun"))

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

print("BEST MODEL:")
print(generate("The sun"))



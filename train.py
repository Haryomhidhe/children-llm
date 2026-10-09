# Import the random helpers used to build training batches and keep training reproducible.
from random import random

# PyTorch provides the model, automatic differentiation, and tensor operations.
import torch
# This second random import is used for the same seeded training setup.
import random 
# The plotting library is imported but not used in this script's current flow.
import matplotlib.pyplot as plt
# Reuse the model and tokenizer implemented in the main language-model file.
from llm_start import SmallGPT, ModelConfig, CharacterTokenizer

# Create a character-level tokenizer and seed all randomness for deterministic results.
tokenizer = CharacterTokenizer()
torch.manual_seed(42)
random.seed(42)

# Load the text corpus and convert it into byte tokens for training the model.
with open("data.txt", "r", encoding="utf-8") as file:
  text =file.read()

  tokens =  tokenizer.encode(text)

  print("Total characters:", len(text))
  print("Total tokens:", len(tokens))

# Split the sequence into training and validation sets.
split_index = int(0.9*len(tokens))
train_tokens = tokens[:split_index]
val_tokens = tokens[split_index:]

print("Train tokens:", len(train_tokens))
print("Val tokens:", len(val_tokens))

# Training hyperparameters for a compact transformer model.
context_length = 128
batch_size = 16

# Build a small random training batch by sampling windows from the token stream.
def get_batch(data):
    inputs = []
    targets = []
    for _ in range (batch_size):
      start = random.randint(0, len(data) - context_length - 1)
      chunk = data[start : start + context_length + 1]
      inputs.append(chunk[:-1])
      targets.append(chunk[1:])
    return torch.tensor(inputs), torch.tensor(targets)

# Instantiate the model configuration and the language model itself.
config = ModelConfig()
model = SmallGPT(config)

# Use AdamW for optimization with a small learning rate.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
best_val_loss = float('inf')
number_of_steps =1600

# Precompute validation batches so the same validation data is used repeatedly.
val_inputs, val_targets = get_batch(val_tokens)     
train_losses = []
val_losses = []
# Train for a fixed number of optimization steps.
for step in range (number_of_steps):
  inputs, targets = get_batch(train_tokens)
  
  logits, loss = model(inputs, targets)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Evaluate the model every 100 steps and track the best validation score.
  if step % 100 == 0:
   with torch.no_grad():
     _, val_loss = model(val_inputs, val_targets)
   train_losses.append(loss.item())
   val_losses.append(val_loss.item())
   print(f"step {step}, loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")
   if val_loss.item() < best_val_loss:
    best_val_loss = val_loss.item()
    torch.save(model.state_dict(), "best_model.pt")
   if step % 1000 == 0:
    torch.save(model.state_dict(), f"checkpoint_step{step}.pt")

# Save the final model and then restore the best validation checkpoint for generation.
torch.save(model.state_dict(),"model.pt")
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Generate text from a prompt by repeatedly sampling the next token.
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

# Restore the best checkpoint and generate again using the best-performing weights.
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

print("BEST MODEL:")
print(generate("The sun"))

# Used to plot the training loss over time if desired.
steps = list(range(0,number_of_steps,100))

plt.plot(steps, train_losses, label="Train loss")
plt.plot(steps, val_losses, label="validation loss")

plt.xlabel("Training steps")
plt.ylabel("loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.show()

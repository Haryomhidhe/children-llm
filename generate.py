# Import PyTorch and the model/tokenizer classes built in the training file.
import torch 
from llm_start import SmallGPT, ModelConfig, CharacterTokenizer

# Create the tokenizer and model, then load the best saved weights.
tokenizer = CharacterTokenizer()
Config = ModelConfig()
model = SmallGPT(Config)

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Ask the user for a starting prompt and convert it into token IDs.
prompt = input("Enter A prompt:")
prompt_tokens = tokenizer.encode(prompt)
# Repeatedly sample the next token for a short generation loop.
for _ in  range(100):
    prompt_tensor = torch.tensor([prompt_tokens])
    logits, _ = model(prompt_tensor)
    next_token_logits = logits[0,-1]
    probabilities = torch.softmax(next_token_logits, dim=0)
    next_token = torch.multinomial(probabilities, num_samples=1).item()
    prompt_tokens.append(next_token)
# Decode the generated token IDs back into readable text and print the output.
generated_text = tokenizer.decode(prompt_tokens)
print(generated_text)

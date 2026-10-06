import torch 
from llm_start import SmallGPT, ModelConfig, CharacterTokenizer

tokenizer = CharacterTokenizer()
Config = ModelConfig()
model = SmallGPT(Config)

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

prompt = input("Enter A prompt:")

prompt_tokens = tokenizer.encode(prompt)
prompt_tensor = torch.tensor([prompt_tokens])
logits, _ = model(prompt_tensor)
next_token_logits = logits[0,-1]
probabilities = torch.softmax(next_token_logits, dim=0)
next_token = torch.multinomial(probabilities, num_samples=1).item()
prompt_tokens.append(next_token)
generated_text = tokenizer.decode(prompt_tokens)
print(generated_text)
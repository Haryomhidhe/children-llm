"""A small decoder-only language model to use as a starting point."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
	vocab_size: int = 256
	context_length: int = 128
	embedding_size: int = 240
	number_of_layers: int = 4
	number_of_heads: int = 4
	feed_forward_size: int = 960
	dropout: float = 0.0


class CharacterTokenizer:
	"""Maps UTF-8 bytes to token IDs, keeping the first version simple."""

	vocab_size = 256

	def encode(self, text: str) -> list[int]:
		return list(text.encode("utf-8"))

	def decode(self, token_ids: list[int]) -> str:
		return bytes(token_ids).decode("utf-8", errors="replace")


class CausalSelfAttention(nn.Module):
	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		self.attention = nn.MultiheadAttention(
			embed_dim=config.embedding_size,
			num_heads=config.number_of_heads,
			dropout=config.dropout,
			batch_first=True,
		)
		self.register_buffer(
			"causal_mask",
			torch.triu(
				torch.ones(config.context_length, config.context_length), diagonal=1
			).bool(),
		)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		sequence_length = hidden_states.size(1)
		attention_output, _ = self.attention(
			hidden_states,
			hidden_states,
			hidden_states,
			attn_mask=self.causal_mask[:sequence_length, :sequence_length],
			need_weights=False,
		)
		return self.dropout(attention_output)


class TransformerBlock(nn.Module):
	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		self.layer_norm_1 = nn.LayerNorm(config.embedding_size)
		self.attention = CausalSelfAttention(config)
		self.layer_norm_2 = nn.LayerNorm(config.embedding_size)
		self.feed_forward = nn.Sequential(
			nn.Linear(config.embedding_size, config.feed_forward_size),
			nn.GELU(),
			nn.Linear(config.feed_forward_size, config.embedding_size),
			nn.Dropout(config.dropout),
		)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		hidden_states = hidden_states + self.attention(self.layer_norm_1(hidden_states))
		return hidden_states + self.feed_forward(self.layer_norm_2(hidden_states))


class SmallGPT(nn.Module):
	def __init__(self, config: ModelConfig) -> None:
		super().__init__()
		self.config = config
		self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
		self.position_embedding = nn.Embedding(
			config.context_length, config.embedding_size
		)
		self.blocks = nn.ModuleList(
			[TransformerBlock(config) for _ in range(config.number_of_layers)]
		)
		self.final_layer_norm = nn.LayerNorm(config.embedding_size)
		self.output_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
		self.output_head.weight = self.token_embedding.weight
		self.apply(self._initialize_weights)

	@staticmethod
	def _initialize_weights(module: nn.Module) -> None:
		if isinstance(module, (nn.Linear, nn.Embedding)):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			nn.init.zeros_(module.bias)
			nn.init.ones_(module.weight)

	def forward(
		self,
		token_ids: torch.Tensor,
		targets: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor | None]:
		_, sequence_length = token_ids.shape
		if sequence_length > self.config.context_length:
			raise ValueError("Input is longer than the model context length.")

		positions = torch.arange(sequence_length, device=token_ids.device)
		hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)
		for block in self.blocks:
			hidden_states = block(hidden_states)
		logits = self.output_head(self.final_layer_norm(hidden_states))

		loss = None
		if targets is not None:
			loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
		return logits, loss

	@torch.no_grad()
	def generate(self, token_ids: torch.Tensor, number_of_tokens: int) -> torch.Tensor:
		self.eval()
		for _ in range(number_of_tokens):
			context = token_ids[:, -self.config.context_length :]
			logits, _ = self(context)
			next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
			token_ids = torch.cat((token_ids, next_token), dim=1)
		return token_ids


def count_parameters(model: nn.Module) -> int:
	return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
	torch.manual_seed(42)
	tokenizer = CharacterTokenizer()
	model = SmallGPT(ModelConfig())

	prompt = "The beginning of a small language model"
	input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
	logits, loss = model(input_ids, input_ids)

	print(f"Trainable parameters: {count_parameters(model):,}")
	print(f"Input shape: {tuple(input_ids.shape)}")
	print(f"Logits shape: {tuple(logits.shape)}")
	print(f"Example next-token loss: {loss.item():.4f}")


if __name__ == "__main__":
	main()

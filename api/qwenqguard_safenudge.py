import json
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .safenudge import ModelWrapper


class Qwen3GuardStream:
	"""Wrapper around Qwen3Guard-Stream models with helper scoring utilities."""

	def __init__(
		self,
		model_path: str = "Qwen/Qwen3Guard-Stream-0.6B",
		model=None,
		tokenizer=None,
		device_map: str = "auto",
		torch_dtype: torch.dtype = torch.bfloat16,
		trust_remote_code: bool = True,
	):
		self.model_path = model_path
		self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
			model_path,
			trust_remote_code=trust_remote_code,
		)
		self.model = model or AutoModel.from_pretrained(
			model_path,
			device_map=device_map,
			torch_dtype=torch_dtype,
			trust_remote_code=trust_remote_code,
		).eval()
		self.device = self._infer_device()

	def _infer_device(self) -> torch.device:
		try:
			return next(self.model.parameters()).device
		except StopIteration:
			return torch.device("cpu")

	@staticmethod
	def _label_eq(label: str, target: str) -> bool:
		return str(label).strip().lower() == target

	@staticmethod
	def _last_token_probs(logits: torch.Tensor) -> torch.Tensor:
		if logits.dim() == 3:
			# [batch, seq_len, num_labels] -> probs of final token in first batch.
			return F.softmax(logits[0, -1], dim=-1)
		if logits.dim() == 2:
			# [batch, num_labels] -> probs of first batch.
			return F.softmax(logits[0], dim=-1)
		if logits.dim() == 1:
			return F.softmax(logits, dim=-1)
		raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

	def _get_role_maps(self, role: str) -> Tuple[Dict[int, str], Dict[int, str]]:
		if role == "user":
			return self.model.query_risk_level_map, self.model.query_category_map
		if role == "assistant":
			return self.model.response_risk_level_map, self.model.response_category_map
		raise ValueError("Role must be either 'user' or 'assistant'")

	def _combined_unsafe_controversial_prob(
		self,
		probs: torch.Tensor,
		risk_map: Dict[int, str],
	) -> Tuple[float, float, float]:
		unsafe_prob = 0.0
		controversial_prob = 0.0

		for idx, label in risk_map.items():
			if self._label_eq(label, "unsafe"):
				unsafe_prob += float(probs[idx])
			elif self._label_eq(label, "controversial"):
				controversial_prob += float(probs[idx])

		return unsafe_prob + controversial_prob, unsafe_prob, controversial_prob

	@torch.no_grad()
	def stream_moderate_from_ids(
		self,
		token_ids: torch.LongTensor,
		role: str,
		stream_state: Optional[Generator] = None,
	) -> Tuple[Dict, Generator]:
		"""
		Stream moderation using Qwen3GuardStream logits.

		Returns per-call moderation for the last processed token and updated stream state.
		"""
		if not torch.is_tensor(token_ids):
			token_ids = torch.tensor(token_ids, dtype=torch.long)

		token_ids = token_ids.to(self.device)

		if stream_state is None:
			stream_state = self.model.stream_generate(token_ids)
			logits_tuple = next(stream_state)
		else:
			logits_tuple = stream_state.send(int(token_ids.item()))

		risk_map, category_map = self._get_role_maps(role)
		if role == "user":
			risk_level_logits = logits_tuple[2]
			category_logits = logits_tuple[3]
		else:
			risk_level_logits = logits_tuple[0]
			category_logits = logits_tuple[1]

		risk_probs = self._last_token_probs(risk_level_logits)
		category_probs = self._last_token_probs(category_logits)

		pred_risk_idx = int(torch.argmax(risk_probs).item())
		pred_cat_idx = int(torch.argmax(category_probs).item())

		combined_risk_prob, unsafe_prob, controversial_prob = self._combined_unsafe_controversial_prob(
			risk_probs,
			risk_map,
		)

		result = {
			"risk_level": risk_map[pred_risk_idx],
			"risk_prob": round(float(risk_probs[pred_risk_idx]), 4),
			"category": category_map[pred_cat_idx],
			"category_prob": round(float(category_probs[pred_cat_idx]), 4),
			"combined_risk_prob": round(float(combined_risk_prob), 4),
			"unsafe_prob": round(float(unsafe_prob), 4),
			"controversial_prob": round(float(controversial_prob), 4),
		}
		return result, stream_state

	@torch.no_grad()
	def close_stream(self, stream_state: Optional[Generator]) -> None:
		if stream_state is not None:
			self.model.close_stream(stream_state)


class Qwen3GuardSafeNudge(ModelWrapper):
	NUDGE = (
		"Sorry, I was going to generate an unsafe response. Instead, let me correct that and make sure the"
		" response is very safe, ethical and cannot be used to harm society. Here is an alternate, safe"
		" answer to your question:"
	)

	def _user_end_index(self, token_ids: torch.LongTensor, tokenizer) -> int:
		token_ids_list = token_ids.tolist()
		im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
		user_id = tokenizer.convert_tokens_to_ids("user")
		im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

		try:
			last_start = next(
				i
				for i in range(len(token_ids_list) - 1, -1, -1)
				if token_ids_list[i : i + 2] == [im_start_id, user_id]
			)
			return next(i for i in range(last_start + 2, len(token_ids_list)) if token_ids_list[i] == im_end_id)
		except StopIteration:
			# Fallback: if template differs, process the entire prompt as initial context.
			return len(token_ids_list) - 1

	def generate_moderated(
		self,
		prompt,
		guard: Qwen3GuardStream,
		target="",
		tau=0.5,
		max_tokens=100,
		verbose=False,
	):
		nudged = False
		stream_state = None

		input_dialog = [
			{"role": "user", "content": prompt},
			{"role": "assistant", "content": target},
		]

		# Generation prompt for the base model.
		input_ids = self._get_ids(input_dialog)
		sentence = target

		# Moderation stream context (can include assistant prefix).
		moderation_ids = guard.tokenizer.apply_chat_template(
			input_dialog,
			return_tensors="pt",
			add_generation_prompt=False,
		)[0]
		user_end_index = self._user_end_index(moderation_ids, guard.tokenizer)
		_, stream_state = guard.stream_moderate_from_ids(
			moderation_ids[: user_end_index + 1],
			role="user",
			stream_state=None,
		)

		if verbose:
			print(sentence, end="")

		j = 0
		try:
			while True:
				logits_top, logits_top_idx, _ = self.get_top_logits_from_ids(input_ids)
				probs_top = torch.nn.functional.softmax(logits_top / self.temperature, dim=-1)
				next_token = logits_top_idx[
					self._rng.choice(len(logits_top_idx), p=probs_top.detach().numpy())
				]

				j += 1
				if (next_token.item() == self.tokenizer.eos_token_id) or (j > max_tokens):
					if verbose:
						print("\n")
					return sentence, nudged

				if self.tokenizer.name_or_path.find("mistral") > -1:
					next_token_str = self.tokenizer.convert_ids_to_tokens(next_token.item()).replace("▁", " ")
				else:
					next_token_str = self.tokenizer.decode(next_token.item())

				mod_result, stream_state = guard.stream_moderate_from_ids(
					torch.tensor(next_token.item(), dtype=torch.long),
					role="assistant",
					stream_state=stream_state,
				)

				if (j >= 5) and (not nudged) and (mod_result["combined_risk_prob"] >= tau):
					nudge_ids = self.tokenizer(self.NUDGE + sentence)["input_ids"][1:]
					input_ids = torch.cat(
						(input_ids, torch.tensor(nudge_ids, dtype=torch.long).reshape(1, -1)),
						dim=1,
					)
					for token_id in nudge_ids:
						_, stream_state = guard.stream_moderate_from_ids(
							torch.tensor(token_id, dtype=torch.long),
							role="assistant",
							stream_state=stream_state,
						)

					if verbose:
						print("|||", end="")

					nudged = True
					yield json.dumps(
						{
							"idx_counter": -1,
							"texts": [],
							"token_ids": [],
							"probs": [],
							"selected_idx": -1,
							"selected_text": "[NUDGE]",
							# "moderation": mod_result,
						}
					) + "\n"
				else:
					sentence += next_token_str
					input_ids = torch.cat((input_ids, next_token.reshape(1, 1)), dim=1)

					if verbose:
						print(next_token_str, end="")

					yield json.dumps(
						{
							"idx_counter": -1,
							"texts": [],
							"token_ids": [],
							"probs": [],
							"selected_idx": -1,
							"selected_text": next_token_str,
							# "moderation": mod_result,
						}
					) + "\n"
		finally:
			guard.close_stream(stream_state)


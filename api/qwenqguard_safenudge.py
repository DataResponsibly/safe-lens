import json

import torch

from .safenudge import ModelWrapper


class Qwen3GuardSafeNudge(ModelWrapper):
    NUDGE = (
        "Sorry, I was going to generate an unsafe response. Instead, let me correct that and make sure the"
        " response is very safe, ethical and cannot be used to harm society. Here is an alternate, safe"
        " answer to your question:"
    )

    @staticmethod
    def _last_label(label):
        if isinstance(label, (list, tuple)):
            return label[-1] if label else ""
        return label

    @classmethod
    def _should_nudge(cls, moderation_result) -> bool:
        risk_level = str(cls._last_label(moderation_result.get("risk_level", ""))).strip().lower()
        return risk_level in {"unsafe", "controversial"}

    @staticmethod
    def _user_end_index(token_ids: torch.LongTensor, tokenizer) -> int:
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
            return next(
                i for i in range(last_start + 2, len(token_ids_list)) if token_ids_list[i] == im_end_id
            )
        except StopIteration:
            # If the chat template changes, fall back to treating the whole prompt as context.
            return len(token_ids_list) - 1

    @classmethod
    def _prime_guard_stream(cls, prompt, target, guard_model, guard_tokenizer):
        input_dialog = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
        moderation_text = guard_tokenizer.apply_chat_template(
            input_dialog,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        moderation_ids = guard_tokenizer(moderation_text, return_tensors="pt").input_ids[0]
        user_end_index = cls._user_end_index(moderation_ids, guard_tokenizer)
        user_result, stream_state = guard_model.stream_moderate_from_ids(
            moderation_ids[: user_end_index + 1],
            role="user",
            stream_state=None,
        )

        for i in range(user_end_index + 1, len(moderation_ids)):
            _, stream_state = guard_model.stream_moderate_from_ids(
                moderation_ids[i],
                role="assistant",
                stream_state=stream_state,
            )

        return user_result, stream_state

    def generate_moderated(
        self,
        prompt,
        guard_model,
        guard_tokenizer,
        target="",
        tau=0.5,
        max_tokens=100,
        verbose=False,
    ):
        del tau

        nudged = False
        input_dialog = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
        input_ids = self._get_ids(input_dialog)
        sentence = target

        if verbose:
            print(sentence, end="")

        user_result, stream_state = self._prime_guard_stream(
            prompt=prompt,
            target=target,
            guard_model=guard_model,
            guard_tokenizer=guard_tokenizer,
        )

        if self._should_nudge(user_result):
            nudge_text = self.NUDGE + sentence
            nudge_ids = self.tokenizer(nudge_text)["input_ids"][1:]
            input_ids = torch.cat(
                (input_ids, torch.tensor(nudge_ids, dtype=torch.long).reshape(1, -1)),
                dim=1,
            )

            guard_nudge_ids = guard_tokenizer.encode(nudge_text, add_special_tokens=False)
            for guard_token_id in guard_nudge_ids:
                _, stream_state = guard_model.stream_moderate_from_ids(
                    torch.tensor(guard_token_id, dtype=torch.long),
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
                }
            ) + "\n"

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

                moderation_result = None
                guard_token_ids = guard_tokenizer.encode(next_token_str, add_special_tokens=False)
                for guard_token_id in guard_token_ids:
                    moderation_result, stream_state = guard_model.stream_moderate_from_ids(
                        torch.tensor(guard_token_id, dtype=torch.long),
                        role="assistant",
                        stream_state=stream_state,
                    )

                if (not nudged) and (moderation_result is not None) and self._should_nudge(moderation_result):
                    nudge_text = self.NUDGE + sentence
                    nudge_ids = self.tokenizer(nudge_text)["input_ids"][1:]
                    input_ids = torch.cat(
                        (input_ids, torch.tensor(nudge_ids, dtype=torch.long).reshape(1, -1)),
                        dim=1,
                    )

                    guard_nudge_ids = guard_tokenizer.encode(nudge_text, add_special_tokens=False)
                    for guard_token_id in guard_nudge_ids:
                        _, stream_state = guard_model.stream_moderate_from_ids(
                            torch.tensor(guard_token_id, dtype=torch.long),
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
                        }
                    ) + "\n"
        finally:
            if stream_state is not None:
                guard_model.close_stream(stream_state)


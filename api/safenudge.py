import json
import numpy as np
import torch
from ._output_handler import clear_cuda_memory


class ModelWrapper(object):
    def __init__(
        self,
        model,
        tokenizer,
        mode="topk",
        k=100,
        p=0.9,
        temperature=1.0,
        random_state=None,
        cuda=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode
        self.k = k
        self.p = p
        self.temperature = 1.0
        self.random_state = random_state
        self.cuda = cuda

        self._rng = np.random.default_rng(self.random_state)

    def get_hidden_state(self, prompt, response):
        """
        Helper function to get hidden states for a given prompt + response combination.
        """
        input = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        input_ids = self._get_ids(input)

        _, last_hidden_state = self._forward_pass_from_ids(input_ids)
        last_hidden_state = last_hidden_state[0, -1]
        return last_hidden_state

    def generate(self, prompt, target="", max_tokens=100, verbose=False):

        input = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]

        input_ids = self._get_ids(input)
        sentence = target

        if verbose:
            print(sentence, end="")

        j = 0
        while True:

            logits_top, logits_top_idx, _ = self.get_top_logits_from_ids(input_ids)
            probs_top = torch.nn.functional.softmax(
                logits_top / self.temperature, dim=-1
            )
            next_token = logits_top_idx[
                self._rng.choice(len(logits_top_idx), p=probs_top.detach().numpy())
            ]

            j += 1
            if (next_token.item() == self.tokenizer.eos_token_id) or (j > max_tokens):
                _, last_hidden_state = self._forward_pass_from_ids(input_ids)
                if verbose:
                    print("\n")
                return sentence, last_hidden_state

            if self.tokenizer.name_or_path.find("mistral") > -1:
                next_token_str = self.tokenizer.convert_ids_to_tokens(
                    next_token.item()
                ).replace("▁", " ")
            else:
                next_token_str = self.tokenizer.decode(next_token.item())

            sentence += next_token_str
            if verbose:
                print(next_token_str, end="")

            input_ids = torch.cat((input_ids, next_token.reshape(1, 1)), dim=1)
            
            del logits_top
            del logits_top_idx
            del probs_top

    def get_top_logits_from_ids(self, input_ids, need_hidden_states=True):
        logits, last_hidden_state = self._forward_pass_from_ids(
            input_ids, need_hidden_states=need_hidden_states
        )
        logits = logits[-1]
        if last_hidden_state is not None:
            last_hidden_state = last_hidden_state[0]

        if self.mode is None:
            logits_top, logits_top_idx = torch.topk(logits, len(logits))

        if self.mode == "topk":
            logits_top, logits_top_idx = torch.topk(logits, self.k)

        if self.mode == "topp":
            # Need to implement
            return False

        return logits_top, logits_top_idx, last_hidden_state

    def _get_ids(self, input):
        input_ids = self.tokenizer.apply_chat_template(input, return_tensors="pt")
        input_ids = input_ids[0][:-1].reshape(1, -1)

        return input_ids

    def _forward_pass_from_ids(self, input_ids, need_hidden_states=True):
        if self.cuda:
            input_ids = input_ids.cuda()

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=False,
                output_hidden_states=need_hidden_states,
                output_attentions=False,
            )
            logits = outputs["logits"][:, -1, :].clone()
            last_hidden_state = None
            if need_hidden_states:
                last_hidden_state = outputs["hidden_states"][-1][:, -1, :].clone()
            del outputs
            del input_ids

        if self.cuda:
            logits = logits.cpu()
            if last_hidden_state is not None:
                last_hidden_state = last_hidden_state.cpu()
            torch.cuda.empty_cache()

        return logits, last_hidden_state


class SafeNudge(ModelWrapper):
    NUDGE = (
        "Sorry, I was going to generate an unsafe response. Instead, let me correct that and make sure the"
        " response is very safe, ethical and cannot be used to harm society. Here is an alternate, safe"
        " answer to your question:"
    )

    def generate_moderated(
        self, prompt, clf, target="", tau=0.5, max_tokens=100, verbose=False
    ):
        nudged = False
        input = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]

        input_ids = self._get_ids(input)
        sentence = target

        try:
            if verbose:
                print(sentence, end="")

            j = 0
            while True:
                need_hidden = not nudged and j >= 4
                logits_top, logits_top_idx, last_hidden_state = (
                    self.get_top_logits_from_ids(
                        input_ids, need_hidden_states=need_hidden
                    )
                )
                probs_top = torch.nn.functional.softmax(
                    logits_top / self.temperature, dim=-1
                )
                next_token = logits_top_idx[
                    self._rng.choice(len(logits_top_idx), p=probs_top.detach().numpy())
                ]

                j += 1
                if (next_token.item() == self.tokenizer.eos_token_id) or (j > max_tokens):
                    if verbose:
                        print("\n")
                    return

                if self.tokenizer.name_or_path.find("mistral") > -1:
                    next_token_str = self.tokenizer.convert_ids_to_tokens(
                        next_token.item()
                    ).replace("▁", " ")
                else:
                    next_token_str = self.tokenizer.decode(next_token.item())

                # Check if next token would be dangerous
                if (
                    (j >= 5)
                    and (not nudged)
                    and (
                        clf.predict_proba(last_hidden_state.numpy().reshape(1, -1))[:, 1]
                        >= tau
                    )
                ):
                    nudge_ids = self.tokenizer(SafeNudge.NUDGE + sentence)["input_ids"][1:]
                    input_ids = torch.cat(
                        (input_ids, torch.tensor(nudge_ids).reshape(1, -1)), dim=1
                    )
                    if verbose:
                        print("|||", end="")
                    nudged = True

                    del logits_top
                    del logits_top_idx
                    del probs_top
                    del last_hidden_state

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

                    del logits_top
                    del logits_top_idx
                    del probs_top
                    del last_hidden_state

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
            del input_ids
            clear_cuda_memory()

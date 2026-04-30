import json
import re
import ast
import asyncio
from time import sleep
import pandas as pd
import numpy as np
import torch
import gc


def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


async def generate_output_stream(
    init_prompt,
    model,
    tokenizer,
    k=10,
    T=0.5,
    max_new_tokens=100,
    sleep_time=0,
    data=None,
    cuda=False,
    verbose=True,
    random_state=None,
    as_json=True,
):
    rng = np.random.default_rng(random_state)
    TERMINATOR = tokenizer.eos_token
    idx_counter = 0
    prompt_ids = None
    output_ids = None

    try:
        if data is None:
            data = pd.DataFrame(
                columns=[
                    "idx_counter",
                    "texts",
                    "token_ids",
                    "probs",
                    "selected_idx",
                    "selected_text",
                ]
            )

            output = ""
            output_ids = torch.tensor([], dtype=torch.long)

        else:
            output = data["selected_text"].str.cat(sep="")
            output_ids = torch.tensor(
                data.apply(
                    lambda row: row["token_ids"][row["selected_idx"]], axis=1
                ).tolist()
            ).reshape(1, -1)

            previous_content = json.loads(data.to_json(orient="records"))
            for d in previous_content:
                if as_json:
                    d = json.dumps(d) + "\n"
                yield d
                idx_counter += 1

        if verbose:
            print(init_prompt)
            print(output, end="", flush=True)

        prompt = [
            {"role": "user", "content": init_prompt},
            {"role": "assistant", "content": ""},
        ]
        prompt_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt")[0][
            :-1
        ].reshape(1, -1)

        if cuda:
            prompt_ids = prompt_ids.cuda()
            output_ids = output_ids.cuda()

        while not (output.find(TERMINATOR) >= 0 or output_ids.shape[-1] >= max_new_tokens):

            def _forward():
                return model(
                    all_ids,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                )

            with torch.no_grad():
                all_ids = torch.cat([prompt_ids, output_ids], dim=-1)
                outputs = await asyncio.to_thread(_forward)
                logits = outputs["logits"][0, -1, :].clone().float().cpu()
                del outputs
                del all_ids

            logits_topk, logits_topk_idx = torch.topk(logits, k)

            texts_topk = [tokenizer.decode(idx) for idx in logits_topk_idx]
            probs_topk = torch.nn.functional.softmax(logits_topk / T, dim=-1)
            probs_topk = probs_topk.detach().numpy()

            next_idx = rng.choice(len(texts_topk), p=probs_topk)

            d = {
                "idx_counter": idx_counter,
                "texts": texts_topk,
                "token_ids": logits_topk_idx.tolist(),
                "probs": probs_topk.tolist(),
                "selected_idx": next_idx,
                "selected_text": texts_topk[next_idx],
            }
            idx_counter += 1

            data.loc[len(data)] = d

            output += texts_topk[next_idx]
            next_id = logits_topk_idx[next_idx].reshape(1, -1)
            if cuda:
                next_id = next_id.cuda()
            output_ids = torch.cat([output_ids, next_id], dim=-1)

            del logits
            del logits_topk
            del logits_topk_idx

            if verbose:
                print(texts_topk[next_idx], end="", flush=True)

            if as_json:
                d = json.dumps(d) + "\n"

            yield d
            await asyncio.sleep(sleep_time)

        if verbose:
            print()
    finally:
        del prompt_ids
        del output_ids
        clear_cuda_memory()


async def generate_output(
    init_prompt,
    model,
    tokenizer,
    k=10,
    T=0.5,
    max_new_tokens=100,
    data=None,
    cuda=False,
    verbose=True,
    random_state=None,
):
    stream = generate_output_stream(
        init_prompt,
        model,
        tokenizer,
        k=k,
        T=T,
        max_new_tokens=max_new_tokens,
        data=data,
        cuda=cuda,
        verbose=verbose,
        random_state=random_state,
        as_json=False,
    )
    data = [d async for d in stream]
    return pd.DataFrame(data)


def edit_output(data: list, token_pos: int, new_token: int) -> pd.DataFrame:
    data = pd.DataFrame(data)
    new_text = data.loc[token_pos, "texts"][new_token]
    data.loc[token_pos, "selected_idx"] = new_token
    data.loc[token_pos, "selected_text"] = new_text

    # Drop all entries after the modified token
    data.drop(range(token_pos + 1, len(data)), inplace=True)
    return data


def parse_connected_json_objects(s):
    json_objects = re.findall(r"\{.*?\}", s)

    parsed = []
    for obj in json_objects:
        try:
            parsed.append(ast.literal_eval(obj))
        except Exception as e:
            print(f"Failed to parse: {obj}. Error: {e}")

    return parsed

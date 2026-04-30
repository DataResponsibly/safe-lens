import os
import json
import asyncio

from dotenv import dotenv_values
import pandas as pd
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form
from fastapi.routing import APIRoute
from fastapi.responses import StreamingResponse
from typing import Optional
from . import (
    load_model,
    generate_output_stream,
    edit_output,
    parse_connected_json_objects,
)

from .safenudge import SafeNudge
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from .wildguard_safenudge import WildGuard, WildGuardSafeNudge
from .qwenqguard_safenudge import Qwen3GuardSafeNudge

CUDA = torch.cuda.is_available()
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
QWEN_GUARD_MODEL_NAME = "Qwen/Qwen3Guard-Stream-0.6B"
QWEN_GUARD_REVISION = os.getenv(
    "QWEN_GUARD_REVISION", "419364a715de9840d47b1457982f64ff37f90ed4"
)

if "HF_TOKEN" in os.environ:
    TOKEN = os.environ["HF_TOKEN"]
else:
    TOKEN = dotenv_values()["HF_TOKEN"]

MAX_CONCURRENT_REQUESTS = 4
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wildguard
    # ml_models["SAFENUDGE_TOKENIZER"] = AutoTokenizer.from_pretrained(
    #     "allenai/wildguard", token=TOKEN
    # )
    # ml_models["SAFENUDGE_CLF"] = AutoModelForCausalLM.from_pretrained(
    #     "allenai/wildguard", toload_modelken=TOKEN
    # )
    # if CUDA:
    #     ml_models["SAFENUDGE_CLF"].to("cuda")

    # ml_models["WILDGUARD"] = WildGuard(
    #     model=ml_models["SAFENUDGE_CLF"], tokenizer=ml_models["SAFENUDGE_TOKENIZER"]
    # )

    # LLM
    ml_models["model"], ml_models["tokenizer"] = load_model(
        MODEL_NAME, token=TOKEN, cuda=CUDA, torch_dtype=torch.float16
    )
    ml_models["QWEN_GUARD_TOKENIZER"] = AutoTokenizer.from_pretrained(
        QWEN_GUARD_MODEL_NAME,
        revision=QWEN_GUARD_REVISION,
        token=TOKEN,
        trust_remote_code=True,
    )
    ml_models["QWEN_GUARD_MODEL"] = AutoModel.from_pretrained(
        QWEN_GUARD_MODEL_NAME,
        revision=QWEN_GUARD_REVISION,
        token=TOKEN,
        device_map="auto",
        torch_dtype=torch.bfloat16 if CUDA else torch.float32,
        trust_remote_code=True,
    ).eval()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    routes = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path in {"/docs", "/openapi.json", "/redoc"}:
            continue
        methods = sorted(m for m in route.methods or set() if m not in {"HEAD", "OPTIONS"})
        routes.append(
            {
                "path": route.path,
                "methods": methods,
                "name": route.name,
                "summary": route.summary,
            }
        )
    return {"routes": routes}


@app.post("/generate", summary="Generate an output stream based on a prompt, using a LLM (Llama 3.2 1B Instruct).")
async def generate(
    init_prompt: str = Form(...),
    k: int = Form(30),
    T: float = Form(1),
    tau: float = Form(0.5),
    max_new_tokens: int = Form(100),
    sleep_time: float = Form(0.0),
    verbose: bool = Form(False),
    random_state: Optional[int] = Form(None),
    data: Optional[list] = Form(None),
    safenudge: bool = Form(False),
):
    """
    Generate an output stream based on a prompt, using a LLM (Llama 3.2 1B Instruct).
    """
    print(safenudge)

    async def _safe_generate():
        async with semaphore:
            stream = generate_output_stream(
                init_prompt=init_prompt,
                model=ml_models["model"],
                tokenizer=ml_models["tokenizer"],
                k=k,
                T=T,
                max_new_tokens=max_new_tokens,
                sleep_time=sleep_time,
                cuda=CUDA,
                verbose=verbose,
                random_state=random_state,
                data=data,
            )
            async for chunk in stream:
                yield chunk

    async def _safe_moderated_generate():
        async with semaphore:
            generator = Qwen3GuardSafeNudge(
                model=ml_models["model"],
                tokenizer=ml_models["tokenizer"],
                mode="topk",
                k=k,
                temperature=T,
                random_state=random_state,
                cuda=CUDA,
            ).generate_moderated(
                prompt=init_prompt,
                guard_model=ml_models["QWEN_GUARD_MODEL"],
                guard_tokenizer=ml_models["QWEN_GUARD_TOKENIZER"],
                target="",
                tau=tau,
                max_tokens=max_new_tokens,
                verbose=verbose,
            )
            # Run generator steps in a thread to unblock the event loop
            while True:
                try:
                    chunk = await asyncio.to_thread(next, generator)
                    yield chunk
                except StopIteration:
                    break

    if not safenudge:
        return StreamingResponse(_safe_generate(), media_type="application/json")
    else:
        return StreamingResponse(_safe_moderated_generate(), media_type="application/json") 


def edit(
    data: list,
    token_pos: int,
    new_token: int,
):
    """
    Edit an output based on a token position and a new token and returns the edited
    output, truncated to the token position.
    """
    data = pd.DataFrame(json.loads(data))
    data = edit_output(data, token_pos, new_token)
    return data


@app.post("/regenerate", summary="Generate an output stream based on a prompt, using a LLM (Llama 3.2 1B Instruct). This endpoint is for regenerating output when user edits.")
async def regenerate(
    init_prompt: str = Form(...),
    k: int = Form(30),
    T: float = Form(1),
    max_new_tokens: int = Form(100),
    sleep_time: float = Form(0.0),
    verbose: bool = Form(False),
    random_state: Optional[int] = Form(None),
    content: Optional[str] = Form(None),
    token_pos: Optional[int] = Form(None),
    new_token: Optional[str] = Form(None),
):
    """
    Generate an output stream based on a prompt, using a LLM (Llama 3.2 1B Instruct).
    """
    if (content is not None) and (token_pos is not None) and (new_token is not None):
        content = parse_connected_json_objects(content)
        new_token_idx = content[token_pos]["texts"].index(new_token)
        content = json.dumps(content)
        data = edit(content, token_pos, new_token_idx)
    else:
        data = None

    async def _safe_regenerate():
        async with semaphore:
            stream = generate_output_stream(
                init_prompt=init_prompt,
                model=ml_models["model"],
                tokenizer=ml_models["tokenizer"],
                k=k,
                T=T,
                max_new_tokens=max_new_tokens,
                sleep_time=sleep_time,
                cuda=CUDA,
                verbose=verbose,
                random_state=random_state,
                data=data,
            )
            async for chunk in stream:
                yield chunk

    return StreamingResponse(_safe_regenerate(), media_type="application/json")

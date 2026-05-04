import os
import json
import logging

from dotenv import dotenv_values
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
from . import metrics
import pickle

CUDA = torch.cuda.is_available()
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

if "HF_TOKEN" in os.environ:
    TOKEN = os.environ["HF_TOKEN"]
else:
    TOKEN = dotenv_values()["HF_TOKEN"]

logger = logging.getLogger(__name__)

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"], ml_models["tokenizer"] = load_model(
        MODEL_NAME, token=TOKEN, cuda=CUDA, torch_dtype=torch.float16
    )
    with open("api/artifacts/clf_mlp_llama_1b.pkl", "rb") as f:
        ml_models['SAFENUDGE_MODEL'] = pickle.load(f)
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(metrics.router)


@app.get("/")
async def root():
    routes = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path == "/metrics":
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
    init_prompt: str = Form(..., min_length=1, max_length=10000),
    k: int = Form(30, ge=1, le=50),
    T: float = Form(1, gt=0.0, le=10.0),
    tau: float = Form(0.5, ge=0.0, le=1.0),
    max_new_tokens: int = Form(100, ge=1, le=4096),
    sleep_time: float = Form(0.0, ge=0.0, le=2.0),
    verbose: bool = Form(False),
    random_state: Optional[int] = Form(None),
    data: Optional[list] = Form(None),
    safenudge: bool = Form(False),
):
    """
    Generate an output stream based on a prompt, using a LLM (Llama 3.2 1B Instruct).
    """
    logger.info("generate called: safenudge=%s", safenudge)
    if not safenudge:
        data = generate_output_stream(
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
        return StreamingResponse(data, media_type="application/json")
    else:
        data = SafeNudge(
            model=ml_models["model"],
            tokenizer=ml_models["tokenizer"],
            mode="topk",
            k=k,
            temperature=T,
            random_state=random_state,
            cuda=CUDA,
        ).generate_moderated(
            prompt=init_prompt,
            clf=ml_models["SAFENUDGE_MODEL"],
            target="",
            tau=tau,
            max_tokens=max_new_tokens,
            verbose=verbose,
        )
        return StreamingResponse(data, media_type="application/json") 


@app.post("/regenerate", summary="Generate an output stream based on a prompt, using a LLM (Llama 3.2 1B Instruct). This endpoint is for regenerating output when user edits.")
async def regenerate(
    init_prompt: str = Form(..., min_length=1, max_length=10000),
    k: int = Form(30, ge=1, le=50),
    T: float = Form(1, gt=0.0, le=10.0),
    max_new_tokens: int = Form(100, ge=1, le=4096),
    sleep_time: float = Form(0.0, ge=0.0, le=2.0),
    verbose: bool = Form(False),
    random_state: Optional[int] = Form(None),
    content: Optional[str] = Form(None),
    token_pos: Optional[int] = Form(None),
    new_token: Optional[str] = Form(None),
):
    """
    Generate an output stream based on a prompt, using a LLM (Llama 3.2 1B Instruct).
    """
    data = None
    if (content is not None) and (token_pos is not None) and (new_token is not None):
        content_list = parse_connected_json_objects(content)
        
        try:
            new_token_idx = content_list[token_pos]["texts"].index(new_token)
        except (IndexError, ValueError) as e:
            # If the index is out of bounds (e.g. truncated history) or token isn't found
            return StreamingResponse(
                iter([json.dumps({"error": f"Invalid token or position. Details: {e}"}) + "\n"]),
                media_type="application/json"
            )
            
        data = edit_output(content_list, token_pos, new_token_idx)

    result = generate_output_stream(
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
    return StreamingResponse(result, media_type="application/json")

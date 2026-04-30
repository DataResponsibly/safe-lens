from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(
    model_path: str = "mistralai/Mistral-7B-Instruct-v0.3",
    # cache_dir: str = None,
    token: str = None,
    cuda: bool = False,
    **kwargs
):
    """
    Load a model from Hugging Face model hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, token=token, **kwargs  # , cache_dir=cache_dir, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=token, **kwargs  # , cache_dir=cache_dir, use_safetensors=True
    )

    if cuda:
        model.to("cuda")

    return model, tokenizer

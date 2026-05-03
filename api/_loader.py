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
        model_path,
        token=token,
        attn_implementation="flash_attention_2" if cuda else "eager",
        **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)

    if cuda:
        model.to("cuda")

    return model, tokenizer

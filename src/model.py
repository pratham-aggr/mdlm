import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"


def load_model():
    print(f"Loading {MODEL_ID} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
    ).cuda()
    model.eval()

    MASK_ID = getattr(model.config, "mask_token_id", None)
    if MASK_ID is None:
        vocab = tokenizer.get_vocab()
        MASK_ID = vocab.get("<mask>", vocab.get("[MASK]"))

    print(f"Loaded. device={next(model.parameters()).device}  MASK_ID={MASK_ID}")
    return model, tokenizer, MASK_ID

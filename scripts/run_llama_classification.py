import os
import getpass
import pandas as pd
import torch
import random
from transformers import pipeline
from utils import batched_iterable, extract_label

# ---------------- Config ----------------
try:
    import config  # optional config file
    SEED = getattr(config, "SEED", 42)
    BATCH_SIZE = getattr(config, "BATCH_SIZE", 64)
    MAX_NEW_TOKENS = getattr(config, "LLAMA_MAX_NEW_TOKENS", 10)
    DATA_PATH = getattr(config, "DATA_PATH", "../data/dummy_data.csv")
except ImportError:
    SEED = 42
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 10
    DATA_PATH = "../data/dummy_data.csv"

# Detect HPC user ID
USER_ID = getpass.getuser()

# Explicit output directory in scratch
OUTPUT_DIR = f"/scratch/network/{USER_ID}/LLM-to-HPC/output"

# Model path with user ID interpolation
LLAMA_MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    f"/scratch/network/{USER_ID}/.cache/huggingface/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)

torch.manual_seed(SEED)
random.seed(SEED)

# ---------------- Load model ----------------
try:
    if not os.path.exists(LLAMA_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Llama model path does not exist at {LLAMA_MODEL_PATH}. "
            "Please update LLAMA_MODEL_PATH or download the model."
        )
    pipe = pipeline(
        "text-generation",
        model=LLAMA_MODEL_PATH,
        tokenizer=LLAMA_MODEL_PATH,
        device=0
    )
except Exception as e:
    raise RuntimeError(f"Error loading Llama model: {e}")

# ---------------- Load dataset ----------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading data from {DATA_PATH}: {e}")

df = df[df['Text'].notna()].copy()

# ---------------- Prompt ----------------
prompt_prefix = """
Classify the following message as Preamble (1) or Not Preamble (0). 
Respond with only a single digit 0 or 1.
"""

generation_kwargs = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "eos_token_id": pipe.tokenizer.encode("\n")[0],
}

# ---------------- Run inference ----------------
results = []
texts = df['Text'].astype(str).tolist()
prompts = [prompt_prefix + f"\nText: {t}\nLabel:" for t in texts]

with torch.inference_mode():
    for batch_prompts in batched_iterable(prompts, BATCH_SIZE):
        outputs = pipe(batch_prompts, **generation_kwargs)
        for i, out in enumerate(outputs):
            raw = out[0]["generated_text"] if isinstance(out, list) else str(out)
            prompt_text = batch_prompts[i]
            model_output = raw[len(prompt_text):].strip() if raw.startswith(prompt_text) else raw.strip()
            label = extract_label(model_output)
            results.append(label)

# ---------------- Save results ----------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "Test_ratings_llama.csv")
df["LLAMA_ratings"] = results
df.to_csv(out_path, index=False)
print(f"✅ Saved Llama binary ratings to {out_path}")

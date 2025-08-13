import os
import pandas as pd
import torch
import random
from transformers import pipeline
from utils import batched_iterable, extract_label

# ---------------- Config ----------------
SEED = 42
BATCH_SIZE = 64
MAX_NEW_TOKENS = 10
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "/scratch/network/USER_ID/.cache/huggingface/llama-8B")
DATA_PATH = os.environ.get("DATA_PATH", "../data/dummy_data.csv")
OUTPUT_DIR = "../data/output"

torch.manual_seed(SEED)
random.seed(SEED)

# ---------------- Load model ----------------
try:
    assert os.path.exists(LLAMA_MODEL_PATH), f"Model path {LLAMA_MODEL_PATH} does not exist. Update LLAMA_MODEL_PATH!"
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

# ---------------- Attach ratings ----------------
df["LLAMA_ratings"] = results

# ---------------- Save results ----------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "TalkSpaceData_EXP_text_level_LLAMA_ratings.csv")
df.to_csv(out_path, index=False)
print(f"✅ Saved Llama binary ratings to {out_path}")

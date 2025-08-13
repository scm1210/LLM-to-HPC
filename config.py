import os
import getpass

# ---------------- USER SETTINGS ----------------
# The script will automatically use your HPC user ID
USER_ID = getpass.getuser()

# You can adjust these for faster/slower runs
SEED = 42
BATCH_SIZE = 64            # <-- change this if needed
GPT_MAX_NEW_TOKENS = 256   # <-- change this if needed
LLAMA_MAX_NEW_TOKENS = 10  # For the smaller Llama runs
# -----------------------------------------------

# ---------------- PATHS ----------------
GPT_MODEL_PATH = f"/scratch/network/{USER_ID}/.cache/huggingface/gpt-oss-20b"
LLAMA_MODEL_PATH = f"/scratch/network/{USER_ID}/.cache/huggingface/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

OUTPUT_DIR = f"/scratch/network/{USER_ID}/MADS/data/output/preamble"
DATA_PATH = f"/scratch/network/{USER_ID}/MADS/data/TalkSpaceData_EXP_text_level.csv"
# -----------------------------------------------

# ---------------- CHECK PATHS ----------------
for path_name, path in [
    ("GPT_MODEL_PATH", GPT_MODEL_PATH),
    ("LLAMA_MODEL_PATH", LLAMA_MODEL_PATH),
    ("DATA_PATH", DATA_PATH)
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ {path_name} does not exist at {path}. "
            "Please update the path to match your environment."
        )

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✅ Config loaded. Using USER_ID='{USER_ID}'")
print(f"Output will be saved to: {OUTPUT_DIR}")
print(f"Batch size: {BATCH_SIZE}, GPT max tokens: {GPT_MAX_NEW_TOKENS}, Llama max tokens: {LLAMA_MAX_NEW_TOKENS}")

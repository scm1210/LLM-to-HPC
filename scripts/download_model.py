from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

model_repo = "openai/gpt-oss-20b"
local_dir = "/scratch/network/sm9518/.cache/huggingface/gpt-oss-20b"  # local folder for the model

# Download model files locally (only runs if not already downloaded)
snapshot_download(repo_id=model_repo, local_dir=local_dir)

# Load tokenizer and model from local folder, not from Hugging Face hub
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir)

print(f"{model_repo} downloaded and loaded successfully from {local_dir}!")

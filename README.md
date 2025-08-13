# LLM-to-HPC

Repo for Running LLMs on Princeton HPC

Please refer to this
[guide](https://scm1210.github.io/posts/LLM%20on%20HPC/LLM-to-HPC.html)
to set-up LLM on adroit.

## Repo Structure

```
LLM-to-HPC/
├── scripts/
│ ├── run_llama_classification.py # Run classification with LLaMA
│ ├── run_gpt_classification.py # Run classification with GPT
├── utils.py # Helper functions
├── data/
│ ├── dummy_data.csv # Example dataset
└── README.md
```

## Usage 

Create conda env with the necessary dependencies:

```
conda activate yourENV
pip install -r requirements.txt
```

## ⚙️ Config
Model path is auto-detected from your $USER HPC username, or can be set via environment variable:
```
export LLAMA_MODEL_PATH=/path/to/llama
export GPT_MODEL_PATH=/path/to/gpt
```

## 📝 Notes
Large model files are not stored in the repo — download them to HPC scratch space.

data/dummy_data.csv is included as an example.

All outputs are ignored by .gitignore except the dummy dataset.

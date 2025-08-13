#!/bin/bash
#SBATCH --job-name=llama-binary           # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --constraint=a100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sm9518@princeton.edu # <-- update to your email

# Load required modules
module purge
module load anaconda3/2024.6
module load cudatoolkit/11.8 

# Activate your conda environment
source activate talkspaceMADS

# Move to your working directory
cd /scratch/network/sm9518/LLM-to-HPC/scripts   # <-- update USERID

echo "Job started at $(date)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running Llama Binary Classification"
python run_llama_classification.py

echo "Job completed at $(date)"

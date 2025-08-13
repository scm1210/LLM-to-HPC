#!/bin/bash
#SBATCH --job-name=GPT-binary           # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1               # CPU cores per task
#SBATCH --mem=36G      
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --constraint=a100                   # Use A100 GPU
#SBATCH --nodelist=adroit-h11g1      
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

echo "Running GPT Binary Classification"
python run_gpt_classification.py

echo "Job completed at $(date)"

#!/bin/bash
#SBATCH --partition=exbio-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=10000
#SBATCH --time=48:00:00
#SBATCH --job-name=reg_drug_true
#SBATCH --output=./Results/Regression/reg_normal_final.out
#SBATCH --error=./Results/Regression/reg_normal_final.err
#SBATCH --cpus-per-gpu=6
#SBATCH --nodelist=gpu02.exbio.wzw.tum.de

python run_DeepCDR.py
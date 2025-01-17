#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --requeue
#SBATCH --job-name=test-ddp
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH --time=02:00:00
#SBATCH --output=test.out
#SBATCH --error=test.err

cd /scratch/xl598/nn02h/lightning

nvidia-smi
module load cuda/12.1.0
nvcc --version
which nvcc
module load python/3.8.2
pip3 install virtualenv
virtualenv --no-download research
source research/bin/activate
pip3 install torch torchvision torchaudio
export LD_LIBRARY_PATH=/scratch/xl598/nn02h/lightning/research/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
pip3 install lightning
pip3 install tensorboard
pip3 install deepspeed -y
srun python train.py

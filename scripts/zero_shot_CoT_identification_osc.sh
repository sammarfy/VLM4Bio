#!/bin/bash

#SBATCH -J LV13bBiA
#SBATCH --output=/users/PAS2136/marufm/vlm_projects/slurms/slurm-%x.%j.out
#SBATCH --cpus-per-task=8 # this requests 1 node, 16 core. 
#SBATCH --time=0-50:00:00 
#SBATCH --gres=gpu:1
#SBATCH --cluster=ascend
#SBATCH --account=PAS2136

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load miniconda3
conda init 
source ~/.bashrc

# MODEL_NAME=gpt-4v
# MODEL_NAME=llava-v1.5-7b
MODEL_NAME=llava-v1.5-13b
source activate llava15

# MODEL_NAME=cogvlm-chat
# MODEL_NAME=cogvlm-grounding-generalist
# source activate vlm_env

# MODEL_NAME=minigpt4-vicuna-7B
# MODEL_NAME=minigpt4-vicuna-13B
# source activate minigptv

# MODEL_NAME=blip-flan-xxl
# MODEL_NAME=blip-flan-xl
# source activate blip

# MODEL_NAME=instruct-vicuna7b
# MODEL_NAME=instruct-vicuna13b
# MODEL_NAME=instruct-flant5xl
# MODEL_NAME=instruct-flant5xxl
# source activate instruct_blip

which python

# DATASET=fish-10k
DATASET=bird

# ##### Identification
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 0 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 1 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 2 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 3 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 4 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 5 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 6 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 7 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 8 -s osc
python zero_shot_CoT/identification.py -m $MODEL_NAME -d $DATASET -c 9 -s osc

exit;

#!/bin/bash

#SBATCH --cpus-per-task=8 # this requests 1 node, 16 core. 
#SBATCH --time=0-3:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_normal_q
#SBATCH --account=ml4science

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load Anaconda3/2020.11
conda init 
source ~/.bashrc

# MODEL_NAME=gpt-4o
# MODEL_NAME=llava-v1.5-7b
# MODEL_NAME=llava-v1.5-13b
# source activate llava_new

# MODEL_NAME=cogvlm-chat
# MODEL_NAME=cogvlm-grounding-generalist
# source activate vlm_env

# MODEL_NAME=blip-flan-xxl
# MODEL_NAME=blip-flan-xl
# source activate blip

# MODEL_NAME=instruct-flant5xl
# MODEL_NAME=instruct-flant5xxl
# MODEL_NAME=instruct-vicuna7b
# MODEL_NAME=instruct-vicuna13b
# source activate instruct_blip

# MODEL_NAME=minigpt4-vicuna-7B
# MODEL_NAME=minigpt4-vicuna-13B
# source activate minigptv

which python


python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 0
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 1
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 2
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 3
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 4
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 5
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 6
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 7
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 8
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 9

python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 0
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 1
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 2
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 3
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 4
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 5
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 6
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 7
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 8
python zero_shot_eval/classification_easy_medium_hard.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 9

exit;
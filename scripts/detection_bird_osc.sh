#!/bin/bash

#SBATCH -J BxxAGZse
#SBATCH --output=/users/PAS2136/marufm/vlm_projects/slurm_outputs/bird_detection_zse/slurm-%x.%j.out
#SBATCH --cpus-per-task=8 # this requests 1 node, 16 core. 
#SBATCH --time=50:00:00 
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
# MODEL_NAME=llava-v1.5-13b
# source activate llava15

# MODEL_NAME=cogvlm-chat
# MODEL_NAME=cogvlm-grounding-generalist
# source activate vlm_env

# MODEL_NAME=minigpt4-vicuna-7B
# MODEL_NAME=minigpt4-vicuna-13B
# source activate minigptv

MODEL_NAME=blip-flan-xxl
# MODEL_NAME=blip-flan-xl
source activate blip

# MODEL_NAME=instruct-vicuna7b
# MODEL_NAME=instruct-vicuna13b
# MODEL_NAME=instruct-flant5xl
# MODEL_NAME=instruct-flant5xxl
# source activate instruct_blip

which python

DATASET=bird

# ##### Zero Shot Eval
# python zero_shot_eval/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'beak' -n 500 -s osc
# python zero_shot_eval/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'head' -n 500 -s osc
# python zero_shot_eval/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'eye' -n 500 -s osc
# python zero_shot_eval/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'wings' -n 500 -s osc
# python zero_shot_eval/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'tail' -n 500 -s osc

# ##### Factual Prompting
# python factual_prompting/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'beak' -n 500 -s osc
# python factual_prompting/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'head' -n 500 -s osc
# python factual_prompting/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'eye' -n 500 -s osc
# python factual_prompting/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'wings' -n 500 -s osc
# python factual_prompting/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'tail' -n 500 -s osc

# ##### Zero_Shot_CoT
# python zero_shot_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'beak' -n 500 -s osc
# python zero_shot_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'head' -n 500 -s osc
# python zero_shot_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'eye' -n 500 -s osc
# python zero_shot_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'wings' -n 500 -s osc
# python zero_shot_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'tail' -n 500 -s osc

##### Zero_Shot_SC_CoT
python zero_shot_SC_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'beak' -n 500 -s osc -u 3
python zero_shot_SC_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'head' -n 500 -s osc -u 3
python zero_shot_SC_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'eye' -n 500 -s osc -u 3
python zero_shot_SC_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'wings' -n 500 -s osc -u 3
python zero_shot_SC_CoT/detection_bird_osc.py -m $MODEL_NAME -t grounding -r 'tail' -n 500 -s osc -u 3


exit;

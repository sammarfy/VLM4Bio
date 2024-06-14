#!/bin/bash

#SBATCH --cpus-per-task=8 # this requests 1 node, 16 core. 
#SBATCH --time=0-1:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_normal_q
#SBATCH --account=ml4science

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load Anaconda3/2020.11
conda init 
source ~/.bashrc

source activate minigptv

which python

# MODEL_NAME=gpt-4v
# MODEL_NAME=llava-v1.5-7b
# MODEL_NAME=llava-v1.5-13b
# MODEL_NAME=cogvlm-chat
# MODEL_NAME=cogvlm-grounding-generalist

# MODEL_NAME=minigpt4-vicuna-7B
# MODEL_NAME=minigpt4-vicuna-13B
# MODEL_NAME=blip-flan-xxl
# MODEL_NAME=blip-flan-xl
# MODEL_NAME=instruct-vicuna7b
# MODEL_NAME=instruct-vicuna13b
# MODEL_NAME=instruct-flant5xl
# MODEL_NAME=instruct-flant5xxl

NUM_IMAGES=500

# ##### CLASSIFICATION
# python zero_shot_eval/classification.py -m $MODEL_NAME -t "direct" -n $NUM_IMAGES
# python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -n $NUM_IMAGES

# # ##### IDENTIFICATION
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "adipose fin" -n $NUM_IMAGES
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "caudal fin" -n $NUM_IMAGES
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "dorsal fin" -n $NUM_IMAGES
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "pectoral fin" -n $NUM_IMAGES
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "pelvic fin" -n $NUM_IMAGES
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "anal fin" -n $NUM_IMAGES
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "eye" -n $NUM_IMAGES
# python zero_shot_eval/identification.py -m $MODEL_NAME -t "head" -n $NUM_IMAGES


# #### GROUNDING 

# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "adipose fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "caudal fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "dorsal fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "pectoral fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "pelvic fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "anal fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "eye" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r "head" -n $NUM_IMAGES -b True


# ##### REFERRING 

# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "adipose fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "caudal fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "dorsal fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "pectoral fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "pelvic fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "anal fin" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "eye" -n $NUM_IMAGES -b True
# python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r "head" -n $NUM_IMAGES -b True


# ##### COUNTING
# python zero_shot_eval/counting.py -m $MODEL_NAME -t direct -n $NUM_IMAGES
# python zero_shot_eval/counting.py -m $MODEL_NAME -t selection -n $NUM_IMAGES


# ##### SPATIAL RELATIONSHIP
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "set" -r "adipose fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "count" -r "adipose fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "set" -r "pectoral fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "count" -r "pectoral fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "set" -r "pelvic fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "count" -r "pelvic fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "set" -r "anal fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "count" -r "anal fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "set" -r "dorsal fin" -n $NUM_IMAGES
# python zero_shot_eval/spatial_rel.py -m $MODEL_NAME -t "count" -r "dorsal fin" -n $NUM_IMAGES


# # ##### SIZE DETECTION
# python zero_shot_eval/size_detection.py -m $MODEL_NAME -t direct -s largest -n $NUM_IMAGES
# python zero_shot_eval/size_detection.py -m $MODEL_NAME -t direct -s smallest -n $NUM_IMAGES
# python zero_shot_eval/size_detection.py -m $MODEL_NAME -t selection -s largest -n $NUM_IMAGES
# python zero_shot_eval/size_detection.py -m $MODEL_NAME -t selection -s smallest -n $NUM_IMAGES

# # ##### CLOSEST FIN DETECTION
# python zero_shot_eval/closest.py -m $MODEL_NAME -r "dorsal fin" -n $NUM_IMAGES
# python zero_shot_eval/closest.py -m $MODEL_NAME -r "caudal fin" -n $NUM_IMAGES
# python zero_shot_eval/closest.py -m $MODEL_NAME -r "pectoral fin" -n $NUM_IMAGES
# python zero_shot_eval/closest.py -m $MODEL_NAME -r "pelvic fin" -n $NUM_IMAGES
# python zero_shot_eval/closest.py -m $MODEL_NAME -r "anal fin" -n $NUM_IMAGES
# python zero_shot_eval/closest.py -m $MODEL_NAME -r "adipose fin" -n $NUM_IMAGES

exit;
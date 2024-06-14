#!/bin/bash

#SBATCH --cpus-per-task=8 # this requests 1 node, 16 core. 
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=p100_normal_q
#SBATCH --account=ml4science

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load Anaconda3/2020.11
conda init 
source ~/.bashrc

# MODEL_NAME=gpt-4v
# MODEL_NAME=gpt-4o
# MODEL_NAME=llava-v1.5-7b
# MODEL_NAME=llava-v1.5-13b
source activate llava_new

# MODEL_NAME=cogvlm-chat
# source activate vlm_env

# MODEL_NAME=blip-flan-xxl
# MODEL_NAME=blip-flan-xl
# source activate blip


which python

python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 0
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 1
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 2
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 3
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 4
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 5
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 6
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 7
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 8
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "fish-prompting" -q "no-prompting" -c 9


python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 0
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 1
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 2
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 3
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 4
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 5
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 6
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 7
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 8
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "bird-prompting" -q "no-prompting" -c 9



python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 0
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 1
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 2
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 3
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 4
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 5
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 6
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 7
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 8
python prompting/classification_prompting.py -m gpt-4v -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 9



python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 0
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 1
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 2
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 3
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 4
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 5
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 6
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 7
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 8
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "fish-prompting" -q "no-prompting" -c 9


python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 0
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 1
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 2
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 3
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 4
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 5
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 6
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 7
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 8
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "bird-prompting" -q "no-prompting" -c 9



python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 0
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 1
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 2
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 3
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 4
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 5
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 6
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 7
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 8
python prompting/classification_prompting.py -m gpt-4o -t "selection" -d "butterfly-prompting" -q "no-prompting" -c 9

exit;
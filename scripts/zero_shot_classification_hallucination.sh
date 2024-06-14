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

# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 0
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 1
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 2
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 3
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 4
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 5
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 6
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 7
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 8
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "nota" -c 9


python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 0
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 1
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 2
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 3
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 4
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 5
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 6
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 7
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 8
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "nota" -c 9



python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 0
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 1
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 2
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 3
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 4
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 5
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 6
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 7
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 8
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "nota" -c 9


# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 0
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 1
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 2
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 3
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 4
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 5
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 6
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 7
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 8
# python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'fish-prompting' -t "selection"  -r "fct" -c 9


python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 0
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 1
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 2
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 3
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 4
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 5
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 6
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 7
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 8
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'bird-prompting' -t "selection"  -r "fct" -c 9



python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 0
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 1
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 2
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 3
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 4
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 5
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 6
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 7
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 8
python zero_shot_eval/classification_hallucination.py -m "gpt-4o" -d 'butterfly-prompting' -t "selection"  -r "fct" -c 9




exit;

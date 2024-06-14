#!/bin/bash

# MODEL_NAME=gpt-4v
# MODEL_NAME=gpt-4o
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


# DATASET=fish-prompting
# DATASET=bird-prompting
# DATASET=butterfly-prompting

# TECHNIQUE=no-prompting
# TECHNIQUE=contextual
# TECHNIQUE=dense-caption
# TECHNIQUE=cot



python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 0
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 1
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 2
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 3
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 4
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 5
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 6
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 7
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 8
python zero_shot_eval/classification_prompting.py -m $MODEL_NAME -t "selection" -d $DATASET -q $TECHNIQUE -c 9

exit;
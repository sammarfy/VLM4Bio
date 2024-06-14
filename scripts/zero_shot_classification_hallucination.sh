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


python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 0
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 1
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 2
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 3
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 4
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 5
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 6
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 7
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 8
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "fct" -c 9


python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 0
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 1
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 2
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 3
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 4
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 5
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 6
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 7
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 8
python zero_shot_eval/classification_hallucination.py -m $MODEL_NAME -d $DATASET -t "selection"  -r "nota" -c 9

exit;

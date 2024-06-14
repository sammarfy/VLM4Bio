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

python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t grounding -r 'beak' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t grounding -r 'head' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t grounding -r 'eye' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t grounding -r 'wings' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t grounding -r 'tail' -n 500

python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t referring -r 'beak' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t referring -r 'head' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t referring -r 'eye' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t referring -r 'wings' -n 500
python zero_shot_eval/detection_bird.py -m $MODEL_NAME -t referring -r 'tail' -n 500

python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'dorsal fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'adipose fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'caudal fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'anal fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'pelvic fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'pectoral fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'head' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t grounding -r 'eye' -n 500


python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'dorsal fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'adipose fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'caudal fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'anal fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'pelvic fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'pectoral fin' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'head' -n 500
python zero_shot_eval/detection.py -m $MODEL_NAME -t referring -r 'eye' -n 500


exit;

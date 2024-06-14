#!/bin/bash

# MODEL_NAME=gpt-4v
# MODEL_NAME=gpt-4o
# MODEL_NAME=llava-v1.5-7b
# MODEL_NAME=llava-v1.5-13b
# MODEL_NAME=cogvlm-chat
# MODEL_NAME=cogvlm-grounding-generalist
# MODEL_NAME=blip-flan-xxl
# MODEL_NAME=blip-flan-xl
# MODEL_NAME=instruct-flant5xl
# MODEL_NAME=instruct-flant5xxl
# MODEL_NAME=instruct-vicuna7b
# MODEL_NAME=instruct-vicuna13b
# MODEL_NAME=minigpt4-vicuna-7B
# MODEL_NAME=minigpt4-vicuna-13B

python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 0
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 1
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 2
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 3
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 4
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 5
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 6
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 7
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 8
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-easy" -c 9


python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 0
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 1
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 2
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 3
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 4
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 5
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 6
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 7
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 8
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "fish-medium" -c 9


python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 0
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 1
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 2
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 3
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 4
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 5
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 6
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 7
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 8
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-easy" -c 9


python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 0
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 1
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 2
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 3
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 4
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 5
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 6
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 7
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 8
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "bird-medium" -c 9

python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 0
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 1
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 2
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 3
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 4
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 5
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 6
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 7
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 8
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-easy" -c 9


python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 0
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 1
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 2
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 3
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 4
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 5
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 6
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 7
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 8
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-medium" -c 9

python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 0
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 1
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 2
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 3
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 4
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 5
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 6
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 7
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 8
python zero_shot_eval/classification.py -m $MODEL_NAME -t "selection" -d "butterfly-hard" -c 9

exit;
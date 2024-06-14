import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--task_option", "-t", type=str, default='direct', help="task option: 'direct', 'selection' ")
# updated
parser.add_argument("--dataset", "-d", type=str, default='fish-500', help="dataset option: 'fish-10k', 'fish-500', 'bird', 'butterfly' ")

args = parser.parse_args()

def read_jsonl(path: str):
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]

result_dir = f'/data/VLM4Bio/results/{args.dataset}/classification/{args.task_option}/'

if os.path.exists(result_dir)==False:
    raise Exception(f'Result files not found. Check the result directory: {result_dir}')


files = os.listdir(result_dir)

files = [file for file in files if 'bioclip' in file]
result_paths = [os.path.join(result_dir, file) for file in files]

# read each chunk of the results
target_class = []
prediction_class = []
top_5_class = []


for path in result_paths:

    result_list = read_jsonl(path)

    for dicts in tqdm(result_list):

        target_class.append(dicts['target-class'])
        prediction_class.append(dicts['output'])

        if args.task_option == 'direct':
            top_5_class.append(dicts['top5'].split(','))



correct_prediction = 0
correct_top5_prediction = 0

for idx, gt in enumerate(target_class):
    if gt==prediction_class[idx]:
        correct_prediction += 1 

    if args.task_option == 'direct':
        if prediction_class[idx] in top_5_class[idx]:
            correct_top5_prediction += 1

print('Accuracy: {:.2f}%'.format(correct_prediction*100/len(target_class)))

if args.task_option == 'direct':
    print('Top-5 accuracy: {:.2f}%'.format(correct_top5_prediction*100/len(target_class)))






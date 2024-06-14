import json
from tqdm import tqdm 
import argparse
import os
import pandas as pd
import numpy as np
import pdb
import warnings
import os.path as osp
warnings.filterwarnings('ignore')
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(1, parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='bioclip', help="")
parser.add_argument("--task_option", "-t", type=str, default='direct', help="task option: 'direct', 'selection' ")
parser.add_argument("--result_dir", "-o", type=str, default='results/', help="path to output")
parser.add_argument("--data_dir", "-o", type=str, default='data/', help="path to output")
parser.add_argument("--num_queries", "-n", type=int, default=-1, help="number of images to query from dataset")
parser.add_argument("--chunk_id", "-c", type=int, default=0, help="0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
parser.add_argument("--dataset", "-d", type=str, default='fish', help="dataset option: 'fish', 'bird', 'butterfly' ")

args = parser.parse_args()

if args.dataset == 'fish':

    args.result_dir = osp.join(args.result_dir, 'fish')
    images_list_path = osp.join(args.data_dir, 'datasets/Fish/metadata/imagelist_10k.txt')
    image_dir = osp.join(args.data_dir, 'datasets/Fish/images')
    img_metadata_path = osp.join(args.data_dir, 'datasets/Fish/metadata/metadata_10k.csv')
    organism = 'fish'

elif args.dataset == 'bird':

    args.result_dir = osp.join(args.result_dir, 'bird')
    images_list_path = osp.join(args.data_dir, 'datasets/Bird/metadata/imagelist_10k.txt')
    image_dir = osp.join(args.data_dir, 'datasets/Bird/images')
    img_metadata_path = osp.join(args.data_dir, 'datasets/Bird/metadata/metadata_10k.csv')
    organism = 'bird'

elif args.dataset == 'butterfly':

    args.result_dir = osp.join(args.result_dir, 'butterfly')
    images_list_path = osp.join(args.data_dir, 'datasets/Butterfly/metadata/imagelist_10k.txt')
    image_dir = osp.join(args.data_dir, 'datasets/Butterfly/images')
    img_metadata_path = osp.join(args.data_dir, 'datasets/Butterfly/metadata/metadata_10k.csv')
    organism = 'butterfly'


args.result_dir = os.path.join(args.result_dir, 'classification' ,args.task_option)

os.makedirs(args.result_dir, exist_ok=True)


print("Arguments Provided: ", args)


with open(images_list_path, 'r') as file:
    lines = file.readlines()
images_list = [line.strip() for line in lines]

img_metadata_df = pd.read_csv(img_metadata_path)
species_list = img_metadata_df['scientificName'].unique().tolist()
species_list = [sp for sp in species_list if sp==sp]

def get_options(options):

    result = []

    current_prefix = ''

    for option in options:

        if option.endswith(')'):
            sp_name = current_prefix.split(',')[0]
            if sp_name != '':
                result.append(sp_name)
            current_prefix = ''
            pass

        else:

            current_prefix = (current_prefix + option) if current_prefix=='' else (current_prefix + ' ' + option)

    sp_name = current_prefix.split('.')[0].split(',')[0]
    result.append(sp_name)
    
    return result




import open_clip
from PIL import Image
import torch 

device = torch.device("cuda")

model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
text = tokenizer(species_list).to(device)
model = model.eval()
model = model.to(device)


from vlm_datasets.species_dataset import SpeciesClassificationDataset
import jsonlines
import json


chunk_len = len(images_list)//10
start_idx = chunk_len * args.chunk_id
end_idx = len(images_list) if args.chunk_id == 9 else (chunk_len * (args.chunk_id+1))
images_list = images_list[start_idx:end_idx]
args.num_queries = len(images_list) if args.num_queries == -1 else args.num_queries


species_dataset = SpeciesClassificationDataset(images_list=images_list, 
                                               image_dir=image_dir, 
                                               img_metadata_path=img_metadata_path)

args.num_queries = min(len(species_dataset), args.num_queries)


out_file_name = "{}/classification_{}_{}_num_{}_chunk_{}.jsonl".format(args.result_dir, args.model, args.task_option, args.num_queries, args.chunk_id)

if os.path.exists(out_file_name):

    print('Existing result file found!')
    queried_files = []

    # read the files that has been already written
    with open(out_file_name, 'r') as file:
        # Iterate over each line
        for line in file:
            # Parse the JSON data
            data = json.loads(line)
            queried_files.append(data['image-path'].split('/')[-1])


    images_list = list(set(images_list) - set(queried_files))
    print(f'Running on the remaining {len(images_list)} files.')

    species_dataset = SpeciesClassificationDataset(images_list=images_list, 
                                               image_dir=image_dir, 
                                               img_metadata_path=img_metadata_path)

    args.num_queries = min(len(species_dataset), args.num_queries)


    writer = jsonlines.open(out_file_name, mode='a')

else:
    writer = jsonlines.open(out_file_name, mode='w')


correct_prediction = 0
partial_prediction = 0
incorrect_prediction = 0

for idx in tqdm(range(args.num_queries)):

    batch = species_dataset[idx]

    if os.path.exists(batch['image_path']) is False:
        print(f"{batch['image_path']} does not exist!")
        continue

    pil_image = Image.fromarray(batch['image'])
    image = preprocess(pil_image).unsqueeze(0).to(device)

    target_sp = batch['species_name']
    if target_sp != target_sp:
        continue

    target_idx = species_list.index(target_sp)

    if args.task_option == 'selection':
        options = batch['option_templates']['selection'].split(' ')[1:]
        sp_list = get_options(options)
        if target_sp == ' ':
            continue
        target_idx = sp_list.index(target_sp)
        text = tokenizer(sp_list).to(device)


    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    ranks = np.argsort(text_probs[0].detach().cpu().numpy())[::-1]
    
    result = dict()

    if args.task_option == 'direct':
        top1_idx = ranks[:1]
        pred_sp = species_list[top1_idx[0].item()] 

        top5_idx = ranks[:5]
        top5_sp = [species_list[idx] for idx in top5_idx]
        top5_score = [str(text_probs[0, idx].item()) for idx in top5_idx]

        result['target-class'] = target_sp
        result['output'] = pred_sp
        result['top5'] = ','.join(top5_sp)
        result['top5_score'] = ','.join(top5_score)

    else:
        top1_idx = ranks[:1]
        pred_sp = sp_list[top1_idx[0].item()] 
        result['target-class'] = target_sp
        result['output'] = pred_sp

    if pred_sp == target_sp:
        correct_prediction += 1
    else:
        genus = target_sp.split(' ')[0]

        if genus in pred_sp:
            partial_prediction += 1
        else:
            incorrect_prediction += 1

    writer.write(result)
    writer.close()
    writer = jsonlines.open(out_file_name, mode='a')
writer.close()


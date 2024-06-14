##############################################################################
# class ----> GPT_4V         model_name ----> "gpt-4v"                       #
# class ----> LLaVA          model_name ----> "llava-v1.5-7b"                #
#                                       ----> "llava-v1.5-13b"               #
# class ----> CogVLM         model_name ----> "cogvlm-chat"                  #
# class ----> MinGPT4        model_name ----> "minigpt4-vicuna-7B"           #
#                                       ----> "minigpt4-vicuna-13B"          #
# class ----> BLIP-2FLAN     model_name ----> "blip-flan-xxl"                #
#                                       ----> "blip-flan-xl"                 #
# class ----> Instruct_BLIP  model_name ----> "instruct-vicuna7b"            #
#                                       ----> "instruct-vicuna13b"           #
#                                       ----> "instruct-flant5xl"            #
#                                       ----> "instruct-flant5xxl"           #
##############################################################################

import json
from tqdm import tqdm
import argparse
import os
import random
import warnings
warnings.filterwarnings('ignore')
import os.path as osp
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(1, parent_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='llava-v1.5-7b', help="multimodal-model, option: 'gpt-4v', 'llava-v1.5-7b', 'llava-v1.5-13b', 'cogvlm-grounding-generalist', 'cogvlm-chat'")
parser.add_argument("--task_option", "-t", type=str, default='direct', help="task option: 'direct', 'selection' ")
parser.add_argument("--result_dir", "-o", type=str, default='results/', help="path to output")
parser.add_argument("--data_dir", "-p", type=str, default='data/', help="path to datasets")
parser.add_argument("--num_queries", "-n", type=int, default=-1, help="number of images to query from dataset")
parser.add_argument("--chunk_id", "-c", type=int, default=0, help="0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
parser.add_argument("--dataset", "-d", type=str, default='fish-prompting', help="dataset option: 'fish-prompting','bird-prompting', 'butterfly-prompting' ")
parser.add_argument("--reasoning_test", "-r", type=str, default='nota', 
                    help="type of reasoning test: fct (false confidence test) or nota (none of the above test)",
                    choices=['fct', 'nota'])
parser.add_argument("--llava_model_dir", "-l", type=str, default='/projects/ml4science/maruf/new_llava_models/') # We saved the LLaVA-v1.5-7b and LLaVA-v1.5-13b models in this path.

args = parser.parse_args()


if args.dataset == 'fish-prompting':
    args.result_dir = osp.join(args.result_dir, 'hallucination_results', 'fish-prompting')
    images_list_path = osp.join(args.data_dir, 'VLM4Bio/datasets/Fish/metadata/imagelist_prompting.txt')
    image_dir = osp.join(args.data_dir, 'VLM4Bio/datasets/Fish/images')
    img_metadata_path = osp.join(args.data_dir, 'VLM4Bio/datasets/Fish/metadata/metadata_prompting.csv')
    organism = 'fish'

elif args.dataset == 'bird-prompting':
    args.result_dir = osp.join(args.result_dir, 'hallucination_results', 'bird-prompting')
    images_list_path = osp.join(args.data_dir, 'VLM4Bio/datasets/Bird/metadata/imagelist_prompting.txt')
    image_dir = osp.join(args.data_dir, 'VLM4Bio/datasets/Bird/images')
    img_metadata_path = osp.join(args.data_dir, 'VLM4Bio/datasets/Bird/metadata/metadata_prompting.csv')
    organism = 'bird'

elif args.dataset == 'butterfly-prompting':
    args.result_dir = osp.join(args.result_dir, 'hallucination_results', 'butterfly-prompting')
    images_list_path = osp.join(args.data_dir, 'VLM4Bio/datasets/Butterfly/metadata/imagelist_prompting.txt')
    image_dir = osp.join(args.data_dir, 'VLM4Bio/datasets/Butterfly/images')
    img_metadata_path = osp.join(args.data_dir, 'VLM4Bio/datasets/Butterfly/metadata/metadata_prompting.csv')
    organism = 'butterfly'


args.result_dir = os.path.join(args.result_dir, 'classification' , args.reasoning_test, args.task_option)

os.makedirs(args.result_dir, exist_ok=True)

print("Arguments Provided: ", args)

if args.model == 'gpt-4v':
    from interface.gpt import GPT_4V
    model = GPT_4V(model_name="gpt-4v")
    print(f'{args.model} loaded successfully.')

if args.model == 'gpt-4o':    
    from interface.gpt import GPT_4o
    model = GPT_4o(model_name="gpt-4o")
    print(f'{args.model} loaded successfully.')

if args.model in ['llava-v1.5-7b', 'llava-v1.5-13b']:
    from interface.llava import LLaVA
    model_version = args.model                    
    model = LLaVA(
        model_name = model_version,
        saved_model_dir = osp.join(args.llava_model_dir, f"{model_version}.pt")
    )

if args.model in ['cogvlm-chat']:
    from interface.cogvlm import CogVLM
    model = CogVLM(model_name=args.model)

if args.model in ['minigpt4-vicuna-7B', 'minigpt4-vicuna-13B']:
    from interface.minigpt4 import MiniGPT4
    model = MiniGPT4(model_name=args.model,
        cfg_path=f'minigpt4_eval_configs/eval_{args.model}.yaml',
        model_cfg_name=f'{args.model}.yaml'
    )

if args.model in ['blip-flan-xxl', 'blip-flan-xl']:
    from interface.blip import BLIP
    model = BLIP(model_name=args.model)

if args.model in ['instruct-vicuna7b', 'instruct-vicuna13b', 'instruct-flant5xl', 'instruct-flant5xxl']:
    from interface.instruct_blip import Instruct_BLIP
    model = Instruct_BLIP(model_name=args.model)


from vlm_datasets.species_dataset import SpeciesClassificationDataset
import jsonlines
import json 

with open(images_list_path, 'r') as file:
    lines = file.readlines()
images_list = [line.strip() for line in lines]

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


for idx in tqdm(range(args.num_queries)):

    batch = species_dataset[idx]

    if os.path.exists(batch['image_path']) is False:
        print(f"{batch['image_path']} does not exist!")
        continue

    result = dict()

    target_species = batch['target_outputs'][args.task_option]
    option_gt = batch['option_gt'][args.task_option]
    questions = batch['question_templates'][args.task_option] 
    options = batch['option_templates'][args.task_option] 
    answer_template = batch['answer_templates'][args.task_option] 


    if args.reasoning_test=='nota':
        none_above = "None of the Above."
        correct_option = batch['option_gt'][args.task_option]
        ind_to_options = {0:'A', 1:'B', 2:'C', 3:'D'}

        # Split the string to isolate just the options part
        parts = options.split(": ")
        intro = parts[0]  # "Options"
        options_list = parts[1].split(", ")
        
        # replace correct answer with "None of the Above."
        for i, opt in enumerate(options_list):
            options_list[i] = opt.replace('.', '')
            if opt.startswith(correct_option):
                options_list[i] = f"{correct_option}) {none_above}"
        
        none_above_index = next(i for i, opt in enumerate(options_list) if none_above in opt)
        options_list.append(options_list.pop(none_above_index))  # Remove and append to the end

        # update the option number
        for ind, opt in enumerate(options_list):
            if ind_to_options[ind]==opt.split(')')[0]:
                continue
            else:
                options_list[ind] = f"{ind_to_options[ind]}){opt.split(')')[1]}"
        
        options = f"{intro}: " + ", ".join(options_list) # updated list of options for nota test

        # add the original ground-truths - to log the original ground-truths (not to be used for evaluation)
        reasoning_test_target_class_key = "nota-target-class"
        reasoning_test_target_class_value = target_species
        reasoning_test_option_gt_key = "nota-option-gt"
        reasoning_test_option_gt_value = option_gt
        
        # update the correct option and correct option index (to be used for evaluation)
        target_species = none_above # the correct answer is constant for the nota test
        option_gt = 'D' # the correct answer index is constant for the nota test
    
    elif args.reasoning_test=='fct':
        # randomly select an option to be our suggested answer
        suggested_answer_index = random.randint(0,3) 

        # Split the string to isolate just the options part
        parts = options.split(": ")
        intro = parts[0]  # "Options"
        options_list = parts[1].split(", ")
        
        suggested_answer = options_list[suggested_answer_index].split(') ')[1].replace('.', '')
        answer_template = "Please provide: 1) Whether the chosen answer is correct (True/False). 2) The correct answer "
        options += f" Chosen Answer: {suggested_answer}."

        reasoning_test_target_class_key = "fct-suggested-answer"
        reasoning_test_target_class_value = suggested_answer
        reasoning_test_option_gt_key = "fct-suggested-answer-index"
        reasoning_test_option_gt_value = suggested_answer_index


    instruction = f"{questions} {options} {answer_template}."

    instruction = instruction.replace('fish', organism)

    model_output = model.prompt(
        prompt_text= instruction,
        image_path = batch['image_path'],
    )

    result['question'] = instruction
    result['target-class'] = target_species

    if model_output is None:
        response = "No response received."
    else:
        response = model_output['response']
    
    result["output"] = response

    result["image-path"] = batch['image_path']
    result["option-gt"] = option_gt
    
    result[reasoning_test_target_class_key] = reasoning_test_target_class_value
    result[reasoning_test_option_gt_key] = reasoning_test_option_gt_value

    writer.write(result)
    writer.close()
    writer = jsonlines.open(out_file_name, mode='a')
writer.close()
##################################################################################################################################################
# class ----> GPT_4V         model_name ----> "gpt-4v"                                 (implemented, env: llava)                                 #
# class ----> LLaVA          model_name ----> "llava-v1.5-7b"                          (implemented, env: llava)                                 #
#                                       ----> "llava-v1.5-13b"                         (implemented, env: llava)                                 #
# class ----> OFA            model_name ----> "ofa-large"                                                                                        #
#                                       ----> "ofa-huge"                                                                                         #
# class ----> CogVLM         model_name ----> "cogvlm-grounding-generalist"            (implemented, env: vlm_env, gpu:2, --ntasks-per-node=8)   #
#                                       ----> "cogvlm-chat"                            (implemented, env: vlm_env, gpu:2, --ntasks-per-node=8)   #
# class ----> MinGPT4        model_name ----> "minigpt4-vicuna-7B"                     (implemented, env: minigptv)                              #
#                                       ----> "minigpt4-vicuna-13B"                    (implemented, env: minigptv)                              #
# class ----> BLIP-2FLAN     model_name ----> "blip-flan-xxl"                          (implemented, env: blip, --cpus-per-task=8)               #
#                                       ----> "blip-flan-xl"                           (implemented, env: blip, --cpus-per-task=8)               #
# class ----> Instruct_BLIP  model_name ----> "instruct-vicuna7b"                   (implemented, env: instruct_blip, --cpus-per-task=8)         #
#                                       ----> "instruct-vicuna13b"                  (implemented, env: instruct_blip, --cpus-per-task=8)         #
#                                       ----> "instruct-flant5xl"                   (implemented, env: instruct_blip, --cpus-per-task=8)         #
#                                       ----> "instruct-flant5xxl"                  (implemented, env: instruct_blip, --cpus-per-task=8)         #
##################################################################################################################################################

import json
from tqdm import tqdm
import argparse
import os

##################################################################################################################################################
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(1, parent_dir)
##################################################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default='llava-v1.5-7b', help="multimodal-model, option: 'gpt-4v', 'llava-v1.5-7b', 'llava-v1.5-13b', 'cogvlm-grounding-generalist', 'cogvlm-chat'")
parser.add_argument("--trait_option", "-t", type=str, default='head', choices=['dorsal fin','adipose fin','caudal fin','anal fin','pelvic fin','pectoral fin','head','eye'])
parser.add_argument("--result_dir", "-o", type=str, default='/projects/ml4science/project_LMM/results/identification', help="path to output")
parser.add_argument("--num_queries", "-n", type=int, default=5, help="number of images to query from dataset")

args = parser.parse_args()
os.makedirs(args.result_dir, exist_ok=True)

print("Arguments: ", args)

if args.model == 'gpt-4v':
    
    from interface.gpt import GPT_4V
    model = GPT_4V(model_name="gpt-4v")
    print(f'{args.model} loaded successfully.')

if args.model in ['llava-v1.5-7b', 'llava-v1.5-13b']:

    from interface.llava import LLaVA
    model_version = args.model                    
    model = LLaVA(
        model_name = model_version,
        saved_model_dir = f"/projects/ml4science/maruf/llava_models/{model_version}.pt"
    )

if args.model in ['cogvlm-grounding-generalist', 'cogvlm-chat']:

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

##########################################################################################################################
from vlm_datasets.identification_dataset import IdentificationDataset
import jsonlines
import json 

images_list_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/sample_images.txt'
image_dir = '/projects/ml4science/maruf/Fish_Data/bg_removed/INHS'
img_metadata_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/INHS.csv'
trait_map_path = '/projects/ml4science/maruf/Fish_Data/bg_removed/metadata/trait_map.pth'
segmentation_dir = '/projects/ml4science/maruf/Fish_Data/bg_removed/INHS_seg_mask/'

with open(images_list_path, 'r') as file:
    lines = file.readlines()
images_list = [line.strip() for line in lines]



identification_dataset = IdentificationDataset(
                                image_dir=image_dir,
                                trait_map_path=trait_map_path,
                                segmentation_dir=segmentation_dir,
                                images_list=images_list,
                                img_metadata_path=img_metadata_path)

args.num_queries = min(len(identification_dataset), args.num_queries)

out_file_name = "{}/identification_{}_{}_num_{}.jsonl".format(args.result_dir, args.model, args.trait_option, args.num_queries)

writer = jsonlines.open(out_file_name, mode='w')

for idx in tqdm(range(args.num_queries)):
    
    batch = identification_dataset[idx]

    if os.path.exists(batch['image_path']) is False:
        print(f"{batch['image_path']} does not exist!")
        continue

    result = dict()

    instruction = f"{batch['question_templates'][args.trait_option]} {batch['answer_templates'][args.trait_option]}."
    target_id = f"{batch['target_outputs'][args.trait_option]}"

    result['trait'] = args.trait_option
    result['question'] = instruction
    result['target-identification'] = target_id

    model_output = model.prompt(
            prompt_text= instruction,
            image_path = batch['image_path'],
        )
    if model_output is None:
        response = "No response received."
    else:
        response = model_output['response']
    
    result["output"] = response

    result["image-path"] = batch['image_path']

    writer.write(result)

writer.close()


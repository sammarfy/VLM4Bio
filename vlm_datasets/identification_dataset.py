import os
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v2 as imageio
import pandas as pd
import random
import vlm_datasets.utils as utils
import pickle

from vlm_datasets.base_dataset import BaseDataset

class IdentificationDataset(BaseDataset):
    def __init__(self,
                 image_dir,
                 trait_map_path,
                 segmentation_dir,
                 images_list=None,
                 img_metadata_path="",
                ):
        super().__init__(image_dir=image_dir, 
                         trait_map_path=trait_map_path,
                         images_list=images_list, 
                         segmentation_dir=segmentation_dir, 
                         img_metadata_path=img_metadata_path)
        
        # unique list of species in the image metadata
        self.species_list = self.img_metadata_df['scientificName'].unique()
        
    def get_question_template(self, present_traits, absent_traits):
        question_templates = {}
        for trait in present_traits:
            question_templates[trait] = f"Is there a {trait} visible in the fish shown in the image?"
        
        for trait in absent_traits:
            question_templates[trait] = f"Is there a {trait} visible in the fish shown in the image?"
        return question_templates
    
    def get_answer_template(self, present_traits, absent_traits):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in present_traits:
            answer_templates[key]=answer_
        for key in absent_traits:
            answer_templates[key]=answer_
        return answer_templates
    
    def get_target_outputs(self, present_traits, absent_traits):
        target_outputs = {}
        for trait in present_traits:
            target_outputs[trait] = "Yes"
        for trait in absent_traits:
            target_outputs[trait] = "No"   
        return target_outputs
        
    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        seg_map = self.load_seg_mask(image_name)
        present_traits, absent_traits = self.find_unique_traits(seg_map, return_id=False)
        
        question_templates = self.get_question_template(present_traits, absent_traits)
        answer_templates = self.get_answer_template(present_traits, absent_traits)
        target_outputs = self.get_target_outputs(present_traits, absent_traits)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "seg_map":seg_map,
            "present_traits":present_traits,
            "absent_traits":absent_traits,
            "question_templates":question_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
        }
        return batch

class FishIdentificationDataset(BaseDataset):
    def __init__(self,
                 image_dir,
                 identification_metadata_path,
                 images_list=None,
                 img_metadata_path="",
                ):
        super().__init__(image_dir=image_dir, 
                         images_list=images_list, 
                         img_metadata_path=img_metadata_path)
        
        # unique list of species in the image metadata
        # self.species_list = self.img_metadata_df['scientificName'].unique()
        self.dataframe = pd.read_csv(identification_metadata_path)
        
        
    def get_question_template(self, unique_traits):
        question_templates = {}
        for trait in unique_traits:
            question_templates[trait] = f"Is there {trait} visible in the fish shown in the image?"
        
        return question_templates
    
    def get_answer_template(self, unique_traits):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in unique_traits:
            answer_templates[key]=answer_
        return answer_templates
    
    def get_target_outputs(self, image_name, unique_traits):
        filtered_df = self.dataframe[self.dataframe.fileNameAsDelivered==image_name]
        target_outputs = {}
        for trait in unique_traits:
            if int(filtered_df[trait].values[0])==1:
                target_outputs[trait] = 'Yes'
            else:
                target_outputs[trait] = 'No'
        return target_outputs
    
    def find_unique_traits(self, image_name):
        filtered_df = self.dataframe[self.dataframe.fileNameAsDelivered==image_name]
        unique_traits = [col for col in filtered_df.columns if filtered_df[col].values[0] ==filtered_df[col].values[0]]
        unique_traits.remove('scientificName')
        unique_traits.remove('fileNameAsDelivered')
        return unique_traits
    
    def find_option_id(self, option_list, gt_options):
        gt_opt_id = []
        for gt_opt in gt_options:
            gt_opt_id.append(option_list.index(gt_opt))
        return gt_opt_id
    
    def get_options_template(self, unique_traits, target_outputs):
        
        options_templates = {}
        options_gt = {}
        
        for trait in unique_traits:
            options_templates[trait] = f"Options: A) Yes, B) No."
            gt_options = target_outputs[trait]
            if gt_options == 'Yes':
                options_gt[trait]='A'
            elif gt_options == 'No':
                options_gt[trait]='B'

        return options_templates, options_gt
        
    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        unique_traits = self.find_unique_traits(image_name)
        answer_templates = self.get_answer_template(unique_traits)
        target_outputs = self.get_target_outputs(image_name, unique_traits)
        options_templates, options_gt = self.get_options_template(unique_traits, target_outputs)
        question_templates = self.get_question_template(unique_traits)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "unique_traits":unique_traits,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch




class BirdIdentificationDataset(BaseDataset):
    def __init__(self,
                 image_dir,
                 identification_metadata_path,
                 images_list=None,
                 img_metadata_path="",
                 trait_category_map_path="",
                ):
        super().__init__(image_dir=image_dir, 
                         images_list=images_list, 
                         img_metadata_path=img_metadata_path)
        
        # unique list of species in the image metadata
        # self.species_list = self.img_metadata_df['scientificName'].unique()
        self.dataframe = pd.read_csv(identification_metadata_path)
        with open(trait_category_map_path, 'rb') as f:
            self.trait_category_map = pickle.load(f)
        
    def get_question_template(self, unique_traits):
        question_templates = {}
        for trait in unique_traits:
            question_templates[trait] = f"What is the {trait} visible in the bird shown in the image?"
        
        return question_templates
    
    def get_answer_template(self, unique_traits):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in unique_traits:
            answer_templates[key]=answer_
        return answer_templates
    
    def get_target_outputs(self, image_name, unique_traits):
        filtered_df = self.dataframe[self.dataframe.fileNameAsDelivered==image_name]
        target_outputs = {}
        for trait in unique_traits:
            string_value = filtered_df[trait].values[0]    
            target_outputs[trait] = string_value.strip().split(' ')
        return target_outputs
    
    def find_unique_traits(self, image_name):
        filtered_df = self.dataframe[self.dataframe.fileNameAsDelivered==image_name]
        unique_traits = [col for col in filtered_df.columns if filtered_df[col].values[0] ==filtered_df[col].values[0]]
        return unique_traits
    
    def find_option_id(self, option_list, gt_options):
        gt_opt_id = []
        for gt_opt in gt_options:
            gt_opt_id.append(option_list.index(gt_opt))
        return gt_opt_id
    
    def get_options_template(self, unique_traits, target_outputs):
        
        options_templates = {}
        options_gt = {}
        
        modified_unique_traits = []
        modified_target_outputs = {}
        
        str_map = {0:'A) ', 1:'B) ', 2:'C) ', 3:'D) ', 4:'E) ', 5:'F) ', 6:'G) '}

        for trait in unique_traits:
            
            gt_options = target_outputs[trait]
            if len(gt_options) > 4:
                continue
                
            modified_unique_traits.append(trait)
            modified_target_outputs[trait] = gt_options
            all_options = self.trait_category_map[trait]
            non_gt_options = list(set(all_options) - set(gt_options))
            
            output_option_list = [opt for opt in gt_options]
            
            if len(gt_options)==1:
                other_options = random.sample(non_gt_options, min(3, len(non_gt_options)))
                
            elif len(gt_options)==2:
                other_options = random.sample(non_gt_options, min(2, len(non_gt_options)))
                
            elif len(gt_options)==3:
                other_options = random.sample(non_gt_options, min(2, len(non_gt_options)))
                
            elif len(gt_options)==4:
                other_options = random.sample(non_gt_options, min(2, len(non_gt_options)))
                
            output_option_list+=other_options
            random.shuffle(output_option_list)
            
            option_str=""
            for idx, opt in enumerate(output_option_list):
                option_str+=f"{str_map[idx]}{opt} "
            
            gt_ids = self.find_option_id(output_option_list, gt_options)
            
            gt_ids = [str_map[idx].strip().replace(')', '') for idx in gt_ids]
            
            options_templates[trait] = f"Options: {option_str}\nSelect all that apply."
            options_gt[trait] = gt_ids
            
            
        return options_templates, options_gt, modified_unique_traits, modified_target_outputs
            
        
    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        unique_traits = self.find_unique_traits(image_name)
        unique_traits.remove('fileNameAsDelivered')

        answer_templates = self.get_answer_template(unique_traits)
        target_outputs = self.get_target_outputs(image_name, unique_traits)

        options_templates, options_gt, unique_traits, target_outputs = self.get_options_template(unique_traits, target_outputs)
        question_templates = self.get_question_template(unique_traits)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "unique_traits":unique_traits,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch

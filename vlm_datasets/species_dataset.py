import os
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v2 as imageio
import pandas as pd
import random
import vlm_datasets.utils as utils

from vlm_datasets.base_dataset import BaseDataset

class SpeciesClassificationDataset(BaseDataset):
    def __init__(self,
                 image_dir,
                 img_metadata_path,
                 images_list=None,
                ):
        super().__init__(image_dir=image_dir, 
                         images_list=images_list, 
                         img_metadata_path=img_metadata_path)
        
        self.species_list = self.img_metadata_df['scientificName'].unique()
        self.template_keys = ["direct", "selection"]
        
    def get_question_template(self):
        question_templates = {
            "direct": "What is the scientific name of the fish in the image?",
            "selection": "What is the scientific name of the fish in the image?",
        }
        return question_templates
    
    def get_options_template(self, species):
        options = list(self.species_list.copy())
        options.remove(species)
        rand_options = list(np.random.choice(options, 3, replace=False))
        rand_options.append(species)
        random.shuffle(rand_options)
        option_id = utils.find_option_id(options=rand_options, correct_option=species)

        option_str = utils.get_option_str_from_list(rand_options)
        options_templates = {
            "direct": "",
            "selection": f"Options: {option_str}",
        }
        options_gt = {
            "direct": "-1",
            "selection": option_id,
        }
        return options_templates, options_gt
    
    def get_answer_template(self):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in self.template_keys:
            answer_templates[key]=answer_
        return answer_templates
    
    def get_target_outputs(self, species):
        target_outputs = {}
        for key in self.template_keys:
            target_outputs[key]=species
        return target_outputs
       
    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        species = self.get_species(image_name)
        question_templates = self.get_question_template()
        option_templates, options_gt = self.get_options_template(species)
        answer_templates = self.get_answer_template()
        target_outputs = self.get_target_outputs(species)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "species_name":species,
            "question_templates":question_templates,
            "option_templates":option_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch


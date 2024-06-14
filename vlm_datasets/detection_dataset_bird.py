import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v2 as imageio
import pandas as pd
import random
import vlm_datasets.utils as utils

from vlm_datasets.base_dataset import BaseDataset

class DetectionGroundingDataset(BaseDataset):
    def __init__(self,
                 image_dir,
                 image_trait_bbox_map_path,
                 images_list=None,
                 img_metadata_path="",
                 normalize_bbox=False,
                ):
        super().__init__(image_dir=image_dir, 
                         trait_map_path="",
                         images_list=images_list,
                        img_metadata_path=img_metadata_path)
        
        # unique list of species in the image metadata
#         self.species_list = self.img_metadata_df['scientificName'].unique()
        self.image_trait_bbox_map_path = image_trait_bbox_map_path
        self.normalize_bbox = normalize_bbox
        self.bbox_description = {
            "True": "The bounding box coordinates are represented as [x1, y1, x2, y2] with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner of the rectangle.",
            "False":"The bounding box is defined by the coordinates [x1, y1, x2, y2], where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner of the rectangle."
        }

    def get_question_template(self, bbox_trait):
        bbox_description = self.bbox_description[str(self.normalize_bbox)]
        question_templates = {}
        for trait in bbox_trait.keys():
            question_templates[trait] = f"{bbox_description} What is the Bounding Box coordinates of the {trait} in the bird shown in the image?"
        return question_templates
    
    def get_options_template(self, bbox_trait):
        # converting bounding box values to string for options
        for key in bbox_trait.keys():
            bbox_trait[key] = str(bbox_trait[key]).replace('(','[').replace(')',']')
        
        options_templates = {}
        options_gt = {}
        for trait in bbox_trait.keys():
            copy_bbox_trait = bbox_trait.copy() #creating a copy for each dictionary
            bbox_coorect_trait = copy_bbox_trait.pop(trait) # removing the correct trait from the copied dictionary
            all_options = list(copy_bbox_trait.values())

            rand_options = list(np.random.choice(all_options, 3, replace=True))
            rand_options.append(bbox_coorect_trait)
            random.shuffle(rand_options)
            option_id = utils.find_option_id(options=rand_options, correct_option=bbox_coorect_trait)

            option_str = utils.get_option_str_from_list(rand_options)
            options_templates[trait] = f"Options: {option_str}"
            options_gt[trait]=option_id
        return options_templates, options_gt

    def get_answer_template(self, bbox_trait):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in bbox_trait.keys():
            answer_templates[key]=answer_
        return answer_templates
    
    def get_target_outputs(self, bbox_trait):
        # converting bounding box values to string for options
        target_outputs = {}
        for trait in bbox_trait.keys():
            target_outputs[trait] = str(bbox_trait[trait]).replace('(','[').replace(')',']')
        return target_outputs
    
    def load_image_trait_bbox_map(self, image_name):
        trait_bbox_map_data = open(self.image_trait_bbox_map_path)
        trait_bbox_map = json.load(trait_bbox_map_data)
        image_trait_bbox_map = trait_bbox_map[image_name]
        return image_trait_bbox_map
    
    def find_present_traits(self, image_trait_bbox_map):
        present_traits = list(image_trait_bbox_map.keys())
        return present_traits
    
    def get_trait_bbox_mapping(self, image, image_trait_bbox_map, present_traits, normalize_bbox=False):
        if len(image.shape) == 2:
            H, W = image.shape
        elif len(image.shape) == 3:
            H, W, C = image.shape
        bbox_traits = {}
        for trait in present_traits:
            bbox = image_trait_bbox_map[trait]["bbox"]
            if normalize_bbox:
                bbox = utils.normalize_bbox_coords(bbox, H=H, W=W, fmt="xyxy")
                bbox = tuple(round(value, 2) for value in bbox)
            bbox_traits[trait] = bbox
        return bbox_traits

        
    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        image_trait_bbox_map = self.load_image_trait_bbox_map(image_name)
        present_traits = self.find_present_traits(image_trait_bbox_map)
        bbox_trait = self.get_trait_bbox_mapping(image, image_trait_bbox_map, present_traits, normalize_bbox=self.normalize_bbox) # returns dictionary of traits -> bounding boxes

        question_templates = self.get_question_template(bbox_trait)
        options_templates, options_gt = self.get_options_template(bbox_trait)
        answer_templates = self.get_answer_template(bbox_trait)
        target_outputs = self.get_target_outputs(bbox_trait)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "present_traits":present_traits,
            "bbox_trait":bbox_trait,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch
    
class DetectionReferringDataset(BaseDataset):
    def __init__(self,
                 image_dir,
                 image_trait_bbox_map_path,
                 images_list=None,
                 img_metadata_path="",
                 normalize_bbox=False,
                ):
        super().__init__(image_dir=image_dir, 
                         trait_map_path="",
                         images_list=images_list,  
                         img_metadata_path=img_metadata_path)
        
        # unique list of species in the image metadata
#         self.species_list = self.img_metadata_df['scientificName'].unique()
        self.image_trait_bbox_map_path = image_trait_bbox_map_path
        self.normalize_bbox = normalize_bbox
        self.bbox_description = {
            "True": "The bounding box coordinates are represented as [x1, y1, x2, y2] with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner of the rectangle.",
            "False":"The bounding box is defined by the coordinates [x1, y1, x2, y2], where [x1, y1] is the top-left corner and [x2, y2] is the bottom-right corner of the rectangle."
        }

    def get_question_template(self, bbox_trait):
        bbox_description = self.bbox_description[str(self.normalize_bbox)]
        question_templates = {}
        for trait in bbox_trait.keys():
            bbox = str(bbox_trait[trait]).replace('(','[').replace(')',']')
            question_templates[trait] = f"{bbox_description} What is the trait of the bird that correspond to the bounding box region {bbox} in the image?"
        return question_templates
        
    def get_options_template(self, bbox_trait):
        # converting bounding box values to string for options

        options_templates = {}
        options_gt = {}
        for trait in bbox_trait.keys():
            copy_bbox_trait = bbox_trait.copy() #creating a copy for each dictionary
            _ = copy_bbox_trait.pop(trait) # removing the correct trait from the copied dictionary
            all_options = list(copy_bbox_trait.keys())

            rand_options = list(np.random.choice(all_options, 3, replace=True))
            rand_options.append(trait)
            random.shuffle(rand_options)
            option_id = utils.find_option_id(options=rand_options, correct_option=trait)

            option_str = utils.get_option_str_from_list(rand_options)
            options_templates[trait] = f"Options: {option_str}"
            options_gt[trait]=option_id
        return options_templates, options_gt

    def get_answer_template(self, bbox_trait):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in bbox_trait.keys():
            answer_templates[key]=answer_
        return answer_templates
    
    def get_target_outputs(self, bbox_trait):
        # converting bounding box values to string for options
        target_outputs = {}
        for trait in bbox_trait.keys():
            target_outputs[trait] = trait  
        return target_outputs
    
    def load_image_trait_bbox_map(self, image_name):
        trait_bbox_map_data = open(self.image_trait_bbox_map_path)
        trait_bbox_map = json.load(trait_bbox_map_data)
        image_trait_bbox_map = trait_bbox_map[image_name]
        return image_trait_bbox_map

    def find_present_traits(self, image_trait_bbox_map):
        present_traits = list(image_trait_bbox_map.keys())
        return present_traits

    def get_trait_bbox_mapping(self, image, image_trait_bbox_map, present_traits, normalize_bbox=False):
        if len(image.shape) == 2:
            H, W = image.shape
        elif len(image.shape) == 3:
            H, W, C = image.shape
        bbox_traits = {}
        for trait in present_traits:
            bbox = image_trait_bbox_map[trait]["bbox"]
            if normalize_bbox:
                bbox = utils.normalize_bbox_coords(bbox, H=H, W=W, fmt="xyxy")
                bbox = tuple(round(value, 2) for value in bbox)
            bbox_traits[trait] = bbox
        return bbox_traits
        
    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        image_trait_bbox_map = self.load_image_trait_bbox_map(image_name)
        present_traits = self.find_present_traits(image_trait_bbox_map)
        bbox_trait = self.get_trait_bbox_mapping(image, image_trait_bbox_map, present_traits, normalize_bbox=self.normalize_bbox) # returns dictionary of traits -> bounding boxes

        question_templates = self.get_question_template(bbox_trait)
        options_templates, options_gt = self.get_options_template(bbox_trait)
        answer_templates = self.get_answer_template(bbox_trait)
        target_outputs = self.get_target_outputs(bbox_trait)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "present_traits":present_traits,
            "bbox_trait":bbox_trait,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch

# Wrapper Class for Detection Dataset
class DetectionDataset:
    def __init__(self,
                 image_dir,
                 image_trait_bbox_map_path,
                 images_list=None,
                 img_metadata_path="",
                 detection_type="grounding",
                 normalize_bbox=False,
                ):
        assert detection_type in ["grounding", "referring"]

        if detection_type == 'grounding':
            self.inner_class = DetectionGroundingDataset(image_dir=image_dir,
                                                         image_trait_bbox_map_path=image_trait_bbox_map_path,
                                                         images_list=images_list,
                                                         img_metadata_path=img_metadata_path,
                                                         normalize_bbox=normalize_bbox,
                                                         )
        elif detection_type == 'referring':
            self.inner_class = DetectionReferringDataset(image_dir=image_dir,
                                                         image_trait_bbox_map_path=image_trait_bbox_map_path,
                                                         images_list=images_list,
                                                         img_metadata_path=img_metadata_path,
                                                         normalize_bbox=normalize_bbox,
                                                         )


    def __getattr__(self, attr):
        return getattr(self.inner_class, attr)
    
    def __getitem__(self, idx):
        return self.inner_class[idx]
    
    def __len__(self):
        return len(self.inner_class)

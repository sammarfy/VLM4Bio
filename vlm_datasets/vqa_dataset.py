import os
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v2 as imageio
import pandas as pd
import random
import vlm_datasets.utils as utils
from vlm_datasets.base_dataset import BaseDataset

class BasicCounting(BaseDataset):
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
        self.template_keys = ["direct", "selection"]
        
    def get_question_template(self):
        question_templates = {}
        for key in self.template_keys:
            question_templates[key] = f"How many unique fins are visible in the fish shown in the image? The fins that are normally present in a fish are dorsal fin, caudal fin, pectoral fin, pelvic fin, anal fin and adipose fin."
        return question_templates
    
    def get_options_template(self, num_fins_present):
        options = list(np.arange(len(self.fins_list)+1))
        options.remove(num_fins_present) # removing the correct option
        rand_options = list(np.random.choice(options, 3, replace=False))
        rand_options.append(num_fins_present)
        random.shuffle(rand_options)
        option_id = utils.find_option_id(options=rand_options, correct_option=num_fins_present)

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
    
    def get_target_outputs(self, num_fins_present):
        target_outputs = {}
        for key in self.template_keys:
            target_outputs[key]=num_fins_present
        return target_outputs

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        seg_map = self.load_seg_mask(image_name)
        present_traits, _ = self.find_unique_traits(seg_map, return_id=False)
        num_fins_present = self.count_fins(present_traits)

        question_templates = self.get_question_template()
        options_templates, options_gt = self.get_options_template(num_fins_present)
        answer_templates = self.get_answer_template()
        target_outputs = self.get_target_outputs(num_fins_present)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "seg_map":seg_map,
            "present_traits":present_traits,
            "num_fins_present":num_fins_present,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch


class SpatialRelation(BaseDataset):
    def __init__(self,
                 image_dir,
                 trait_map_path,
                 segmentation_dir,
                 spatial_relationship_path,
                 images_list=None,
                 img_metadata_path="",
                 mode = "set", #options: set, count
                ):
        super().__init__(image_dir=image_dir, 
                         trait_map_path=trait_map_path,
                         images_list=images_list, 
                         segmentation_dir=segmentation_dir, 
                         img_metadata_path=img_metadata_path,
                         spatial_relationship_path=spatial_relationship_path,
                         )
        
        self.mode = mode
        assert mode in ["set", "count"]

        # unique list of species in the image metadata
        self.species_list = self.img_metadata_df['scientificName'].unique()
        
    def get_question_template(self):
        question_templates = {}
        for trait, trait_dict in self.spatial_trait_dict.items():
            for relation in trait_dict.keys():
                question_key = f"{trait}-{relation}"
                if self.mode == "set":
                    question_templates[question_key] = f"What are the unique fins {relation}-to the {trait} of the fish shown in the image? The fins that are normally present in a fish are dorsal fin, caudal fin, pectoral fin, pelvic fin, anal fin and adipose fin."
                elif self.mode == "count":
                    question_templates[question_key] = f"How many unique fins are {relation}-to the {trait} of the fish shown in the image? The fins that are normally present in a fish are dorsal fin, caudal fin, pectoral fin, pelvic fin, anal fin and adipose fin."
        return question_templates
    
    def get_options_template(self):
        options_templates = {}
        options_gt = {}
        for trait, trait_dict in self.spatial_trait_dict.items():
            for relation in trait_dict.keys():
                option_key = f"{trait}-{relation}"
                options_templates[option_key]=""
                options_gt[option_key]=""
        return options_templates, options_gt

    def get_answer_template(self):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for trait, trait_dict in self.spatial_trait_dict.items():
            for relation in trait_dict.keys():
                answer_key = f"{trait}-{relation}"
                answer_templates[answer_key]=answer_
        return answer_templates
    
    def get_target_outputs(self, present_traits):
        target_outputs = {}
        for trait, trait_dict in self.spatial_trait_dict.items():
            for relation, possible_fins in trait_dict.items():
                target_key = f"{trait}-{relation}" #example : dorsal fin-right
                list_fins = set()
                for fin in possible_fins: # possible fins: dorsal fin-right=['adipose fin', 'caudal fin'] 
                    if fin in present_traits:
                        list_fins.add(fin)
                if self.mode == "set":
                    target_outputs[target_key] = list_fins
                elif self.mode == "count":
                    target_outputs[target_key] = len(list_fins)
        return target_outputs

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        seg_map = self.load_seg_mask(image_name)
        present_traits, _ = self.find_unique_traits(seg_map, return_id=False)
        
        question_templates = self.get_question_template()
        options_templates, options_gt = self.get_options_template()
        answer_templates = self.get_answer_template()
        target_outputs = self.get_target_outputs(present_traits)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "seg_map":seg_map,
            "present_traits":present_traits,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch

class SizeDetectionDataset(BaseDataset):
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
                         img_metadata_path=img_metadata_path,
                         )
        
        # unique list of species in the image metadata
        self.species_list = self.img_metadata_df['scientificName'].unique()
        self.template_keys = ["largest", "smallest"]
        self.question_type = ["direct", "selection"]

    def get_largest_smallest(self, segmap):
        size_dict = {}
        for fin in self.fins_list:
            trait_key = utils.find_key_for_value(self.trait_map, fin)
            area = len(np.where(segmap==trait_key)[0])
            if area > 0:
                size_dict[fin] = area
        largest_fin = max(size_dict, key=size_dict.get)
        smallest_fin = min(size_dict, key=size_dict.get)
        return largest_fin, smallest_fin, size_dict

    def get_question_template(self):
        question_templates = {}
        for key in self.template_keys:
            for qs_type in self.question_type:
                qs_key = f"{key}-{qs_type}"
                question_templates[qs_key] = f"What is the {key} fin visible in the fish shown in the image? The fins that are normally present in a fish are dorsal fin, caudal fin, pectoral fin, pelvic fin, anal fin and adipose fin."
        return question_templates
    
    def get_options_template(self, largest_fin, smallest_fin):
        options_templates = {}
        options_gt = {}
        for key in self.template_keys:
            for qs_type in self.question_type:
                qs_key = f"{key}-{qs_type}"
                if qs_type == "direct":
                    options_templates[qs_key]=""
                    options_gt[qs_key]="-1"
                else:
                    options = self.fins_list.copy()
                    correct_fin = largest_fin if key == "largest" else smallest_fin
                    options.remove(correct_fin)
                    rand_options = list(np.random.choice(options, 3, replace=False))
                    rand_options.append(correct_fin)
                    random.shuffle(rand_options)

                    option_id = utils.find_option_id(options=rand_options, correct_option=correct_fin)
                    option_str = utils.get_option_str_from_list(rand_options)
                    options_templates[qs_key]=option_str
                    options_gt[qs_key]=option_id
        return options_templates, options_gt

    def get_answer_template(self):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in self.template_keys:
            for qs_type in self.question_type:
                qs_key = f"{key}-{qs_type}"
                answer_templates[qs_key]=answer_
        return answer_templates
    
    def get_target_outputs(self, largest_fin, smallest_fin):
        target_outputs = {}
        for key in self.template_keys:
            for qs_type in self.question_type:
                qs_key = f"{key}-{qs_type}"
                if key == "largest":
                    target_outputs[qs_key]=largest_fin
                elif key == "smallest":
                    target_outputs[qs_key]=smallest_fin
        return target_outputs

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        seg_map = self.load_seg_mask(image_name)
        largest_fin, smallest_fin, size_dict = self.get_largest_smallest(seg_map)

        question_templates = self.get_question_template()
        options_templates, options_gt = self.get_options_template(largest_fin, smallest_fin)
        answer_templates = self.get_answer_template()
        target_outputs = self.get_target_outputs(largest_fin, smallest_fin)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "seg_map":seg_map,
            "size_dict":size_dict,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch
    

class ClosestFinDataset(BaseDataset):
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
                         img_metadata_path=img_metadata_path,
                         )
        
        # unique list of species in the image metadata
        self.species_list = self.img_metadata_df['scientificName'].unique()

    def get_closest_fin(self, seg_map, thres = 0.2): #theshold for minimum distance
        present_traits, _ = self.find_unique_traits(seg_map, return_id=False)
        bbox_trait = self.get_trait_bbox_mapping(seg_map, present_traits, normalize_bbox=False) # returns dictionary of traits -> bounding boxes     
        bbox_trait = {key: value for key, value in bbox_trait.items() if "fin" in key} # filter only the present fins

        dist_mat = np.full((len(bbox_trait), len(bbox_trait)), np.inf)
        keys_list = list(bbox_trait.keys())
        closest_fin = {}
        for i, key1 in enumerate(bbox_trait.keys()):
            for j, key2 in enumerate(bbox_trait.keys()):
                if i==j:
                    continue
                dist_mat[i, j] = utils.bbox_distance(bbox_trait[key1], bbox_trait[key2])
            closest_ids = np.argsort(dist_mat[i, :])[:2]
            min_dist1 = dist_mat[i, closest_ids[0]]
            min_dist2 = dist_mat[i, closest_ids[1]]
            dist_ratio = (min_dist2-min_dist1)/(min_dist1+1e-6)
            if dist_ratio < thres:
                continue
            closest_fin[key1] = keys_list[closest_ids[0]] 
        return closest_fin

    def get_question_template(self, closest_fin):
        question_templates = {}
        for key in closest_fin.keys():
            question_templates[key] = f"What is the closest fin to the {key} in the fish shown in the image? The fins that are normally present in a fish are dorsal fin, caudal fin, pectoral fin, pelvic fin, anal fin and adipose fin."
        return question_templates
    
    def get_options_template(self, closest_fin):
        options_templates = {}
        options_gt = {}
        for key in closest_fin.keys():
            options = self.fins_list.copy()
            correct_fin = closest_fin[key] 
            options.remove(correct_fin)
            options.remove(key)
            rand_options = list(np.random.choice(options, 3, replace=False))
            rand_options.append(correct_fin)
            random.shuffle(rand_options)

            option_id = utils.find_option_id(options=rand_options, correct_option=correct_fin)
            option_str = utils.get_option_str_from_list(rand_options)
            options_templates[key]=option_str
            options_gt[key]=option_id
        return options_templates, options_gt

    def get_answer_template(self, closest_fin):
        answer_ = "Write the answer after writing 'The answer is: '"
        answer_templates = {}
        for key in closest_fin.keys():
            answer_templates[key]=answer_
        return answer_templates
    
    def get_target_outputs(self, closest_fin):
        target_outputs = {}
        for key in closest_fin.keys():
            target_outputs[key] = closest_fin[key]
        return target_outputs

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image = self.load_image(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        seg_map = self.load_seg_mask(image_name)
        closest_fin = self.get_closest_fin(seg_map)

        question_templates = self.get_question_template(closest_fin)
        options_templates, options_gt = self.get_options_template(closest_fin)
        answer_templates = self.get_answer_template(closest_fin)
        target_outputs = self.get_target_outputs(closest_fin)
        
        batch = {
            "image_path":image_path,
            "image":image,
            "seg_map":seg_map,
            "closest_fin":closest_fin,
            "question_templates":question_templates,
            "option_templates":options_templates,
            "answer_templates":answer_templates,
            "target_outputs":target_outputs,
            "option_gt":options_gt,
        }
        return batch
    
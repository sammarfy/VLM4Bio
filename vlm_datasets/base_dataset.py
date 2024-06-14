import os
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v2 as imageio
import pandas as pd
import random
import vlm_datasets.utils as utils

class BaseDataset(Dataset):
    def __init__(self,
                 image_dir,
                 images_list = None,
                 segmentation_dir = "",
                 img_metadata_path = "",
                 species_metadata_path = "",
                 trait_map_path= "",
                 spatial_relationship_path= "",
                ):
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.species_metadata_path = species_metadata_path
        self.img_metadata_path = img_metadata_path
        self.trait_map_path = trait_map_path
        self.spatial_relationship_path = spatial_relationship_path

        self.images_list = os.listdir(self.image_dir) if images_list is None else images_list
        
        # loading the image metadata path
        self.img_metadata_df = pd.read_csv(self.img_metadata_path)
        
        #load trait-maps
        if self.trait_map_path!="":
            self.trait_map = torch.load(self.trait_map_path)
            self.max_num_traits = len(self.trait_map)
            self.fins_list = [trait for trait in self.trait_map.values() if 'fin' in trait]
        
        if self.spatial_relationship_path!="":
            self.spatial_trait_dict=torch.load(self.spatial_relationship_path)
            
        #to do
        # species morpho metadata
        
    def load_image(self, image_name):
        img_array = np.asarray(imageio.imread(os.path.join(self.image_dir, image_name)))
        return img_array
    
    def get_image_shape(self, img_array):
        H, W, _ = img_array.shape
        return H, W

    def load_seg_mask(self, image_name):
        seg_mask_name = image_name.split(".")[0] + ".npy"
        seg_mask = np.load(os.path.join(self.segmentation_dir, seg_mask_name))
        return seg_mask
    
    def get_species(self, image_name):
        filtered_df = self.img_metadata_df[self.img_metadata_df['fileNameAsDelivered'] == image_name]
        scientific_name = filtered_df["scientificName"].values[0]
        return scientific_name
    
    def count_fins(self, present_traits):
        num_fins_present = 0
        for present_trait in present_traits:
            if present_trait in self.fins_list:
                num_fins_present+=1
        return num_fins_present

    def find_unique_traits(self, segmap, return_id=True):
        present_traits = set(np.unique(segmap))
        all_traits = set(list(self.trait_map.keys()))
        absent_traits = all_traits - present_traits
        
        #removing background from present trait, assuming bg = 0
        present_traits = present_traits-{0}
        
        #returns integer trait ids
        if return_id:
            return present_traits, absent_traits
        
        #encoding traits using trait-mapping
        present_traits = [self.trait_map[key] for key in present_traits]
        absent_traits = [self.trait_map[key] for key in absent_traits]
        return set(present_traits), set(absent_traits)
    
    def get_trait_bbox_mapping(self, segmap, present_traits, normalize_bbox=False):
        H, W = segmap.shape
        trait_fmt = utils.check_trait_type(present_traits) #checking trait format
        bbox_traits = {}
        for trait in present_traits:
            target_class = utils.find_key_for_value(self.trait_map, trait) if trait_fmt=="string" else trait
            bbox = utils.find_bounding_box_from_segmap(segmap, target_class=target_class)
            if normalize_bbox:
                bbox = utils.normalize_bbox_coords(bbox, H=H, W=W, fmt="xyxy")
                bbox = tuple(round(value, 2) for value in bbox)
            bbox_traits[trait] = bbox
        return bbox_traits

    def __getitem__(self, idx):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.images_list)
    
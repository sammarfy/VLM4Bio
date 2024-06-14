import os
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd


def get_image_list_from_path(image_path):
    """
    Given the image path, the function reads the txt file.
    """
    image_list = []
    with open(image_path, 'r') as file:
        for line in file:
            image_list.append(line.strip())
    return image_list

def get_option_str_from_list(options):
    assert len(options) == 4
    return f"A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}."

def find_bounding_box_from_segmap(segmentation_map, target_class):
    y_coords, x_coords = np.where(segmentation_map == target_class)

    # Check if the target class exists in the segmentation map
    if y_coords.size > 0 and x_coords.size > 0:
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        return (min_x, min_y, max_x, max_y)
    else:
        return None  # Target class not found

def find_option_id(options, correct_option):
    mapping = {0:"A", 1:"B", 2:"C", 3:"D"}
    if correct_option in options:
        location_id = options.index(correct_option)
        return mapping[location_id]
    else:
        return -1

def normalize_bbox_coords(bbox, H, W, fmt="xyxy"):
    if fmt in ["xyxy", "xywh"]:
        return (bbox[0]/W, bbox[1]/H, bbox[2]/W, bbox[3]/H)
    else:
        print("Format Not Defined!")

def plot_bbox_with_segmentation(segmap, bbox, figsize=None, cmap="tab20b"):
    x1, y1, x2, y2 = bbox
    
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.imshow(segmap, cmap=cmap)
    bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

    # Add the bounding box to the plot
    plt.gca().add_patch(bbox)
    plt.show()

def check_trait_type(elements):
    has_string = False
    has_np_integer = False
    for elem in elements:
        if isinstance(elem, str):
            has_string = True
        elif isinstance(elem, np.integer):
            has_np_integer = True
        else:
            raise TypeError("Only String and Integers are allowed!")
        if has_string and has_np_integer:
            return TypeError("The traits have to be all integers or strings!")

    # Return the type based on the flags
    if has_string:
        return "string"
    elif has_np_integer:
        return "integer"

def find_key_for_value(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return "Value not found"


def bbox_distance(bbox1, bbox2):
    # Unpack the bounding boxes
    x1, y1, x2, y2 = bbox1
    a1, b1, a2, b2 = bbox2

    # Check if the bounding boxes overlap horizontally or vertically
    horizontal_overlap = (x1 <= a2 and x2 >= a1)
    vertical_overlap = (y1 <= b2 and y2 >= b1)

    # Compute horizontal and vertical distances
    horizontal_distance = 0 if horizontal_overlap else min(abs(x2 - a1), abs(a2 - x1))
    vertical_distance = 0 if vertical_overlap else min(abs(y2 - b1), abs(b2 - y1))

    # If overlapping in one direction, return distance in the other direction
    if horizontal_overlap:
        return vertical_distance
    if vertical_overlap:
        return horizontal_distance

    # If not overlapping in either direction, return the diagonal distance
    return (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5
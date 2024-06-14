from interface.framework import *
import argparse
import os
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.backends.cudnn as cudnn


from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

class MiniGPT4(Framework):

    def __init__(self, model_name, cfg_path, model_cfg_name, gpu_id=0):
        super().__init__(model_name)  

        self.model_name = model_name
        # self.cfg_path = cfg_path
        # self.model_cfg_name = model_cfg_name
        # self.gpu_id = 0

        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                     'pretrain_llama2': CONV_VISION_LLama2}
        
        variables_dict = {"cfg_path": cfg_path,
                          "model_cfg_name": model_cfg_name,
                          "gpu_id": gpu_id,
                          "options": None}

        args = argparse.Namespace(**variables_dict) 

        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        
        self.CONV_VISION = conv_dict[model_config.model_type]

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)


    def prompt(self, 
            prompt_text, 
            image_path
    ):
        pil_img = self.load_image(image_path)
        chat_state = self.CONV_VISION.copy()
        img_list = []
        llm_message = self.chat.upload_img(pil_img, chat_state, img_list)
        self.chat.encode_img(img_list)

        user_message = prompt_text
        self.chat.ask(user_message, chat_state)

        llm_message = self.chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]

        out_dict = {
            'prompt_text': prompt_text,
            'prompt_img': image_path,
            'response': llm_message.strip()
        }

        return out_dict

    
    def load_image(self, image_path):
        if os.path.exists(image_path) == False:
            print(f'There is no image in {image_path}, Please Check.')
            return None

        image = Image.open(image_path).convert('RGB')
        return image


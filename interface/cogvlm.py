from interface.framework import *
from models.cogvlm_model import CogVLMModel
from cogvlm_utils.language import llama2_tokenizer, llama2_text_processor_inference
from cogvlm_utils.vision import get_image_processor
from cogvlm_utils.chat import chat
from sat.model.mixins import CachedAutoregressiveMixin
import argparse
import torch
import time 
from PIL import Image
from cogvlm_utils.parser import parse_response

class CogVLM(Framework):
    def __init__(self, model_name):
        super().__init__(model_name)  # Call the constructor of the Parent class
        self.model_name = model_name
        self.is_grounding = False

        if "cogvlm" in self.model_name:
            if torch.cuda.device_count() < 2:
                print(f"Can not {self.model_name}. At least 2 A100 GPU required.")
            if 'grounding' in self.model_name:
                self.is_grounding = True

        model, model_args = CogVLMModel.from_pretrained(
                f"{self.model_name}",
                args=argparse.Namespace(
                    deepspeed=None,
                    local_rank=0,
                    rank=0,
                    world_size=1,
                    model_parallel_size=1,
                    mode='inference',
                    skip_init=True,
                    fp16=False,
                    bf16=True,
                    use_gpu_initialization=True,
                    device='cuda',
                ))
        self.model = model.eval()
        self.model_args = model_args
        print(f"{self.model_name} is sucessfully loaded")

        self.tokenizer = llama2_tokenizer("lmsys/vicuna-7b-v1.5", signal_type="chat")
        self.image_processor = get_image_processor(self.model_args.eva_args["image_size"][0])
        self.model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        self.text_processor_infer = llama2_text_processor_inference(self.tokenizer, None, self.model.image_length)

    def load_image(self, image_path):
        if os.path.exists(image_path) == False:
            print(f'There is no image in {image_path}, Please Check.')
            return None

        image = Image.open(image_path)
        return image



    def prompt(self, 
            prompt_text, 
            image_path
    ):
        
        model = self.model
        model_args = self.model_args
        
        tokenizer = self.tokenizer
        image_processor = self.image_processor
        text_processor_infer = self.text_processor_infer

        pil_img = self.load_image(image_path)
        
        if pil_img is None:
            return None
        
        if self.self_consistency:
            top_k_val = 10
        else:
            top_k_val = 1

        with torch.no_grad():
            response, history, cache_image = chat(
                "", 
                model, 
                text_processor_infer,
                image_processor,
                prompt_text, 
                history=[],
                image=pil_img,
                max_length=2048, 
                top_p=0.4, 
                temperature=0.8,
                top_k=top_k_val,
                invalid_slices=text_processor_infer.invalid_slices,
                no_prompt=False
                )
        if self.is_grounding:
            timestamp = int(time.time())
            ext = image_path.split(".")[-1]
            img_name = image_path.split("/")[-1][:-4]
            parse_response(pil_img, response, f"/home/marufm/lmm-projects/LMM/output_images/grounding/{timestamp}_{img_name}.{ext}")

        out_dir = {
            'prompt_text': prompt_text,
            'prompt_img':  image_path,
            'response': response
        }

        return out_dir


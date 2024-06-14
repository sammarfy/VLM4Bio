import os
from interface.framework import *
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
# Define a class for each model type, implementing the prompting behavior

class LLaVA(Framework):

    def load_image(self, image_path):
        if os.path.exists(image_path) == False:
            print(f'There is no image in {image_path}, Please Check.')
            return None

        image = Image.open(image_path).convert("RGB")
        return [image]



    def prompt(self, 
            prompt_text, 
            image_path, 
            args_temperature = 0.2,
            args_num_beams = 1,
            args_max_new_tokens = 512,
            args_top_p = None
    ):
        
        # self.model_dict contains the loaded model
        model = self.model_dict['model']
        tokenizer = self.model_dict['tokenizer']
        image_processor = self.model_dict['image_processor']
        context_len = self.model_dict['context_len']
        args_conv_mode = None

        images = self.load_image(image_path)
        if images is None:
            return None
        
        qs = prompt_text

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in qs:

            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # if "llama-2" in self.model_name.lower():
        #     conv_mode = "llava_llama_2"
        # elif "v1" in self.model_name.lower():
        #     conv_mode = "llava_v1"
        # elif "mpt" in self.model_name.lower():
        #     conv_mode = "mpt"
        # else:
        #     conv_mode = "llava_v0"

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"


        if args_conv_mode is not None and conv_mode != args_conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args_conv_mode, args_conv_mode
                )
            )
        else:
            args_conv_mode = conv_mode

        conv = conv_templates[args_conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        model = model.eval()

        # new code
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args_temperature > 0 else False,
                temperature=args_temperature,
                top_p=args_top_p,
                num_beams=args_num_beams,
                max_new_tokens=args_max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(outputs)

        ####



        
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         images=images_tensor,
        #         do_sample=True if args_temperature > 0 else False,
        #         temperature=args_temperature,
        #         top_p=args_top_p,
        #         num_beams=args_num_beams,
        #         max_new_tokens=args_max_new_tokens,
        #         use_cache=True,
        #         stopping_criteria=[stopping_criteria],
        #     )

        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(
        #         f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        #     )

        # outputs = tokenizer.batch_decode(
        #     output_ids[:, input_token_len:], skip_special_tokens=True
        # )[0]
        # outputs = outputs.strip()
        # if outputs.endswith(stop_str):
        #     outputs = outputs[: -len(stop_str)]
        # outputs = outputs.strip()

        out_dir = {
            'prompt_text': prompt_text,
            'prompt_img':  image_path,
            'response': outputs
        }

        return out_dir


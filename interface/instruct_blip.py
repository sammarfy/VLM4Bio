from interface.framework import *
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

class Instruct_BLIP(Framework):

    def __init__(self, model_name):
        super().__init__(model_name)  

        self.model_name = model_name

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        if self.model_name=='instruct-vicuna7b':
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=self.device)
        elif self.model_name=='instruct-vicuna13b':
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna13b", is_eval=True, device=self.device)
        elif self.model_name=='instruct-flant5xl':
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=self.device)
        elif self.model_name=='instruct-flant5xxl':
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xxl", is_eval=True, device=self.device)


    def prompt(self, 
            prompt_text, 
            image_path
    ):
        pil_img = self.load_image(image_path)
        image = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)

        response = self.model.generate({"image": image, "prompt": prompt_text})[0]

        out_dict = {
            'prompt_text': prompt_text,
            'prompt_img': image_path,
            'response': response.strip()
        }

        return out_dict

    
    def load_image(self, image_path):
        if os.path.exists(image_path) == False:
            print(f'There is no image in {image_path}, Please Check.')
            return None

        image = Image.open(image_path).convert('RGB')
        return image


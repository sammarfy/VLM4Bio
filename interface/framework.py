from openai import OpenAI
import base64
import os
import torch
import sys

# Define the base class for the framework which can be extended to support various models
class Framework:

    def __init__(self, model_name, saved_model_dir=None, self_consistency=False):

        self.model_name = model_name
        self.model_dir = saved_model_dir
        self.client = None
        self.model_dict = None
        self.self_consistency = self_consistency
        # self.is_grounding = False

        # if "cogvlm" in self.model_name:
        #     if torch.cuda.device_count() < 2:
        #         print(f"Can not {self.model_name}. At least 2 A100 GPU required.")
        #     if 'grounding' in self.model_name:
        #         self.is_grounding = True


        if self.model_name in ["gpt-4v", "gpt-4o"]:

            # Setting up the api
            api_key, org_key = self.set_keys()

            if api_key is None:
                print('There is an issue with the OpenAI key. Please check.')
                self.client = OpenAI()

            else:
                self.client = OpenAI(
                    api_key=api_key,
                    organization = org_key
                )

        if "llava" in self.model_name:
            self.model_dict = self.load_model()
            
            if self.model_dict is not None:
                print(f"{self.model_name} is sucessfully loaded from {self.model_dir}.")

    
    def set_keys(self):

        # set the open ai api keys
        api_key, org_key = None, None 

        with open("gpt_api/api_key.txt", "r") as f:
            api_key = f.read().strip()
        with open("gpt_api/org_key.txt", "r") as f:
            org_key = f.read().strip()

        return api_key, org_key

    def load_model(self):

        if os.path.exists(self.model_dir) == False:

            print('There is no saved model in the saved_model_dir')
            return None
        
        else:

            return torch.load(self.model_dir)


    def prompt(self, input_text):
        raise NotImplementedError("Subclasses should implement this method")



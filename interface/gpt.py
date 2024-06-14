from openai import OpenAI
import base64
import os
from interface.framework import *

# Define a class for each model type, implementing the prompting behavior

class GPT_4V(Framework):

    def encode_image(self, image_path):

        if os.path.exists(image_path) is False:

            print(f'Image file does not exist in {image_path} .')

            return None

        with open(image_path, "rb") as image_file:

            return base64.b64encode(image_file.read()).decode('utf-8')

    def prompt(self, prompt_text, image_path):

        client = self.client

        if client is None:
            print("There is an issue with the client. Please check.")
            return None
        
        # encode the image of the image_path

        base64_image = self.encode_image(image_path)

        if base64_image is None:
            return None

        # Implement the specific prompting logic for GPT-4v

        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": [
                    #         {'type': "text", "text": "You are an expert visual AI assistant and are great at detecting species name and scientific name from fish images. You also understands the scientific terminologies used in fish study. You only answer question about fish. If someone asks a question that is not about fish or the traits present in the fish, you let them know that you are unable to answer based on the type of AI assistant you are."}

                    #     ],
                    # },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt_text}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            output = response.choices[0].message.content.strip()
        
        except Exception as error:
            output = f"An error occurred: {error}"

        out_dir = {
            'prompt_text': prompt_text,
            'prompt_img':  image_path,
            'response': output
        }

        return out_dir

class GPT_4o(Framework):

    def encode_image(self, image_path):

        if os.path.exists(image_path) is False:

            print(f'Image file does not exist in {image_path} .')

            return None

        with open(image_path, "rb") as image_file:

            return base64.b64encode(image_file.read()).decode('utf-8')

    def prompt(self, prompt_text, image_path):

        client = self.client

        if client is None:
            print("There is an issue with the client. Please check.")
            return None
        
        # encode the image of the image_path

        base64_image = self.encode_image(image_path)

        if base64_image is None:
            return None

        # Implement the specific prompting logic for GPT-4o

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": [
                    #         {'type': "text", "text": "You are an expert visual AI assistant and are great at detecting species name and scientific name from fish images. You also understands the scientific terminologies used in fish study. You only answer question about fish. If someone asks a question that is not about fish or the traits present in the fish, you let them know that you are unable to answer based on the type of AI assistant you are."}

                    #     ],
                    # },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{prompt_text}"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            output = response.choices[0].message.content.strip()
        
        except Exception as error:
            output = f"An error occurred: {error}"

        out_dir = {
            'prompt_text': prompt_text,
            'prompt_img':  image_path,
            'response': output
        }

        return out_dir
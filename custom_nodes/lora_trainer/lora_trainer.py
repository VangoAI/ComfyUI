import requests
from PIL import Image
from io import BytesIO

class LoraTrainer:
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "dataset": ("STRING",),
                "validation_prompts": ("VALIDATION_PROMPTS",),
                "network_dim": ("INT", {
                    "default": 32, 
                    "min": 1, #Minimum value
                    "max": 128, #Maximum value
                    "step": 1 #Slider's step
                }),
                "network_alpha": ("INT", {
                    "default": 16, 
                    "min": 1, #Minimum value
                    "max": 128, #Maximum value
                    "step": 1 #Slider's step
                }),
                "conv_dim": ("INT", {
                    "default": 32, 
                    "min": 1, #Minimum value
                    "max": 128, #Maximum value
                    "step": 1 #Slider's step
                }),
                "conv_alpha": ("INT", {
                    "default": 16, 
                    "min": 1, #Minimum value
                    "max": 128, #Maximum value
                    "step": 1 #Slider's step
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.00001, 
                    "min": 0.0, #Minimum value
                    "max": 1.0, #Maximum value
                    "step": 0.00001 #Slider's step
                }),
                "epochs": ("INT", {
                    "default": 20, 
                    "min": 0, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1 #Slider's step
                }),
                "repeats": ("INT", {
                    "default": 1, 
                    "min": 0, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1 #Slider's step
                }),
                "batch_size": ("INT", {
                    "default": 1, 
                    "min": 0, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1 #Slider's step
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 0xffffffffffffffff, #Maximum value
                    "step": 1 #Slider's step
                }),
                "save_checkpoint_every_n_epochs": ("INT", {
                    "default": 1, 
                    "min": 0, #Minimum value
                    "max": 20, #Maximum value
                    "step": 1 #Slider's step
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "train_lora"

    CATEGORY = "Train"

    def train_lora(self, dataset, validation_prompts, network_dim, network_alpha, conv_dim, conv_alpha, learning_rate, epochs, repeats, batch_size, seed, save_checkpoint_every_n_epochs):
        images = []
        response = requests.get("https://th.bing.com/th/id/OIG.CO2sHWK_IEYIwzXsC2hX")
        for i in range(100):
            images.append(Image.open(BytesIO(response.content)))
        return images

class ValidationPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt1": ("STRING", {"multiline": True}),
                "prompt2": ("STRING", {"multiline": True}),
                "prompt3": ("STRING", {"multiline": True}),
                "prompt4": ("STRING", {"multiline": True}),
                "prompt5": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("VALIDATION_PROMPTS",)

    FUNCTION = "validation_prompts"

    CATEGORY = "Train"

    def validation_prompts(self, prompt1, prompt2, prompt3, prompt4, prompt5):
        return [prompt1, prompt2, prompt3, prompt4, prompt5]

NODE_CLASS_MAPPINGS = {
    "LoraTrainer": LoraTrainer,
    "ValidationPrompts": ValidationPrompts,
}

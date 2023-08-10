import os
import folder_paths

class LoadZip:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".zip")]
        return {"required":
                    {"zip": (sorted(files), )},
                }

    CATEGORY = "Train"

    RETURN_TYPES = ("ZIP", )
    FUNCTION = "load_zip"

    def load_zip(self, zip):
        zip_path = folder_paths.get_annotated_filepath(zip)
        return (zip_path, )

class ValidationPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt1": ("STRING", {"multiline": True, "default": "prompt 1"}),
                "prompt2": ("STRING", {"multiline": True, "default": "prompt 2"}),
                "prompt3": ("STRING", {"multiline": True, "default": "prompt 3"}),
                "prompt4": ("STRING", {"multiline": True, "default": "prompt 4"}),
                "prompt5": ("STRING", {"multiline": True, "default": "prompt 5"}),
            }
        }
    
    RETURN_TYPES = ("VALIDATION_PROMPTS",)

    FUNCTION = "validation_prompts"

    CATEGORY = "Train"

    def validation_prompts(self, prompt1, prompt2, prompt3, prompt4, prompt5):
        return ([prompt1, prompt2, prompt3, prompt4, prompt5], )

NODE_CLASS_MAPPINGS = {
    "LoadZip": LoadZip,
    "ValidationPrompts": ValidationPrompts,
}

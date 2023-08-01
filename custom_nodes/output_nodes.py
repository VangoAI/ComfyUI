class ImageOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "execute"

    CATEGORY = "Output"

    def execute(self, image_input):
        return image_input
    
class LatentOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "execute"

    CATEGORY = "Output"

    def execute(self, latent_input):
        return latent_input


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageOutput": ImageOutput,
    "LatentOutput": LatentOutput,
}

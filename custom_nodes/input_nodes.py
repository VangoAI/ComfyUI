class StringInput:
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
                "input": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "execute"

    CATEGORY = "Input"

    def execute(self, string_input):
        return string_input
    
class LatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "execute"

    CATEGORY = "Input"

    def execute(self, latent_input):
        return latent_input


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "StringInput": StringInput,
    "LatentInput": LatentInput,
}

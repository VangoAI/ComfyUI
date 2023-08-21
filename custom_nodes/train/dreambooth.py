import requests
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import torch
import os
import zipfile
import uuid
import time
import subprocess
import toml
import glob
import shutil
import ast
import re

class DreamboothTrainer:
    @classmethod
    def INPUT_TYPES(s):
        BASE_MODELS = ["SDXL 0.9", "SDXL 1.0", "SD 1.5", "SD 2.1", "Realistic Vision 5.0", "Realistic Vision 1.4", "DreamShaper", "DreamShaper XL1.0", "MajicMix Realistic 2.5"]
        return {
            "required": {
                "base_model": (BASE_MODELS, ),
                "dataset": ("ZIP",),
                "validation_prompts": ("VALIDATION_PROMPTS",),
                "model_name": ("STRING", {"default": "my_dreambooth_model"}),
                "token": ("STRING", {"default": "token"}),
                "network_dim": ("INT", {
                    "default": 32, 
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "network_alpha": ("INT", {
                    "default": 16, 
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "conv_dim": ("INT", {
                    "default": 32, 
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "conv_alpha": ("INT", {
                    "default": 16, 
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.00001, 
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.00001
                }),
                "epochs": ("INT", {
                    "default": 20, 
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "repeats": ("INT", {
                    "default": 1, 
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "batch_size": ("INT", {
                    "default": 1, 
                    "min": 0,
                    "max": 100,
                    "step": 1
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1
                }),
                "save_checkpoint_every_n_epochs": ("INT", {
                    "default": 1, 
                    "min": 0,
                    "max": 20,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "train_lora"

    CATEGORY = "Train"

    def extract_dataset(self, zip_path):
        train_path = "/home/ubuntu/vango_ComfyUI/train_data/"
        train_path += str(uuid.uuid4())
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        
        data_path = train_path + "/data"
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                if not member.is_dir():
                    target_path = os.path.join(data_path, os.path.basename(member.filename))
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)

        # remove AppleDouble files in the data_path
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.startswith("._"):
                    os.remove(os.path.join(root, file))
        return train_path

    def generate_latents(self, train_path):
        train_data_dir = train_path + "/data"

        bucketing_json    = os.path.join(train_path, "meta_lat.json")
        metadata_json     = os.path.join(train_path, "meta_clean.json")
        bucket_resolution = 1024
        mixed_precision   = "no"
        flip_aug          = False
        clean_caption     = False
        recursive         = False

        metadata_config = {
            "_train_data_dir": train_data_dir,
            "_out_json": metadata_json,
            "recursive": recursive,
            "full_path": recursive,
            "clean_caption": clean_caption
        }

        bucketing_config = {
            "_train_data_dir": train_data_dir,
            "_in_json": metadata_json,
            "_out_json": bucketing_json,
            "_model_name_or_path": "/home/ubuntu/vango_ComfyUI/models/checkpoints/sd_xl_base_0.9.safetensors",
            "recursive": recursive,
            "full_path": recursive,
            "flip_aug": flip_aug,
            "batch_size": 4,
            "max_data_loader_n_workers": 2,
            "max_resolution": f"{bucket_resolution}, {bucket_resolution}",
            "mixed_precision": mixed_precision,
        }

        def generate_args(config):
            args = ""
            for k, v in config.items():
                if k.startswith("_"):
                    args += f'"{v}" '
                elif isinstance(v, str):
                    args += f'--{k}="{v}" '
                elif isinstance(v, bool) and v:
                    args += f"--{k} "
                elif isinstance(v, float) and not isinstance(v, bool):
                    args += f"--{k}={v} "
                elif isinstance(v, int) and not isinstance(v, bool):
                    args += f"--{k}={v} "
            return args.strip()

        merge_metadata_args = generate_args(metadata_config)
        prepare_buckets_args = generate_args(bucketing_config)

        merge_metadata_command = f"python3 /home/ubuntu/vango_ComfyUI/custom_nodes/train/kohya-trainer/finetune/merge_all_to_metadata.py {merge_metadata_args}"
        prepare_buckets_command = f"python3 /home/ubuntu/vango_ComfyUI/custom_nodes/train/kohya-trainer/finetune/prepare_buckets_latents.py {prepare_buckets_args}"

        subprocess.run(merge_metadata_command, shell=True)
        time.sleep(1)
        subprocess.run(prepare_buckets_command, shell=True)

        return bucketing_json
        

    def get_lora_config(self, network_dim, network_alpha, conv_dim, conv_alpha):
        network_args    = ""
        network_module  = "networks.lora"

        lora_config = {
            "additional_network_arguments": {
                "no_metadata"                     : False,
                "network_module"                  : network_module,
                "network_dim"                     : network_dim,
                "network_alpha"                   : network_alpha,
                "network_args"                    : network_args,
                "network_train_unet_only"         : True,
                "training_comment"                : None,
            },
        }
        return lora_config

    def get_optimizer_config(self, learning_rate):
        optimizer_type = "AdaFactor"
        optimizer_args = "[ \"scale_parameter=False\", \"relative_step=False\", \"warmup_init=False\" ]"
        lr_scheduler = "constant_with_warmup"
        lr_warmup_steps = 100
        lr_scheduler_num = 0

        if isinstance(optimizer_args, str):
            optimizer_args = optimizer_args.strip()
            if optimizer_args.startswith('[') and optimizer_args.endswith(']'):
                try:
                    optimizer_args = ast.literal_eval(optimizer_args)
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing optimizer_args: {e}\n")
                    optimizer_args = []
            elif len(optimizer_args) > 0:
                print(f"WARNING! '{optimizer_args}' is not a valid list! Put args like this: [\"args=1\", \"args=2\"]\n")
                optimizer_args = []
            else:
                optimizer_args = []
        else:
            optimizer_args = []

        optimizer_config = {
            "optimizer_arguments": {
                "optimizer_type"          : optimizer_type,
                "learning_rate"           : learning_rate,
                "max_grad_norm"           : 0,
                "optimizer_args"          : optimizer_args,
                "lr_scheduler"            : lr_scheduler,
                "lr_warmup_steps"         : lr_warmup_steps,
                "lr_scheduler_num_cycles" : lr_scheduler_num if lr_scheduler == "cosine_with_restarts" else None,
                "lr_scheduler_power"      : lr_scheduler_num if lr_scheduler == "polynomial" else None,
                "lr_scheduler_type"       : None,
                "lr_scheduler_args"       : None,
            },
        }
        return optimizer_config

    def get_advanced_training_config(self):
        noise_control_type        = "none"
        noise_offset_num          = 0.1
        adaptive_noise_scale      = 0.01
        multires_noise_iterations = 6
        multires_noise_discount = 0.3
        min_snr_gamma             = 5

        advanced_training_config = {
            "advanced_training_config": {
                "noise_offset"              : noise_offset_num if noise_control_type == "noise_offset" else None,
                "adaptive_noise_scale"      : adaptive_noise_scale if adaptive_noise_scale and noise_control_type == "noise_offset" else None,
                "multires_noise_iterations" : multires_noise_iterations if noise_control_type =="multires_noise" else None,
                "multires_noise_discount"   : multires_noise_discount if noise_control_type =="multires_noise" else None,
                "min_snr_gamma"             : min_snr_gamma if not min_snr_gamma == -1 else None,
            }
        }
        return advanced_training_config

    def get_training_config(self, validation_prompts, lora_name, train_path, meta_lat_json_path, network_dim, network_alpha, conv_dim, conv_alpha, learning_rate, epochs, repeats, batch_size, seed, save_checkpoint_every_n_epochs):
        train_data_dir              = train_path + "/data"
        output_dir                  = "/home/ubuntu/vango_ComfyUI/models/loras"

        project_name                = lora_name
        wandb_api_key               = ""
        in_json                     = meta_lat_json_path
        gradient_checkpointing      = True
        _half_vae                   = True
        cache_text_encoder_outputs  = True
        min_timestep                = 0
        max_timestep                = 1000
        num_repeats                 = repeats
        resolution                  = 1024
        keep_tokens                 = 0
        num_epochs                  = epochs
        train_batch_size            = batch_size
        mixed_precision             = "fp16"
        seed                        = seed
        optimization                = "scaled dot-product attention"
        save_precision              = "fp16"
        save_every_n_epochs         = save_checkpoint_every_n_epochs
        enable_sample               = True
        sampler                     = "euler_a"
        positive_prompt             = ""
        negative_prompt             = ""
        custom_prompts              = validation_prompts
        prompt_from_caption         = "none"
        num_prompt                  = 2
        logging_dir                 = os.path.join(train_path, "logs")

        prompt_config = {
            "prompt": {
                "negative_prompt" : negative_prompt,
                "width"           : resolution,
                "height"          : resolution,
                "scale"           : 7,
                "sample_steps"    : 28,
                "subset"          : [],
            }
        }

        train_config = {
            "sdxl_arguments": {
                "cache_text_encoder_outputs" : cache_text_encoder_outputs,
                "no_half_vae"                : True,
                "min_timestep"               : min_timestep,
                "max_timestep"               : max_timestep,
                "shuffle_caption"            : True if not cache_text_encoder_outputs else False,
            },
            "model_arguments": {
                "pretrained_model_name_or_path" : "/home/ubuntu/vango_ComfyUI/models/checkpoints/sd_xl_base_0.9.safetensors",
                "vae"                           : "/home/ubuntu/vango_ComfyUI/models/vae/sdxl_vae.safetensors",
            },
            "dataset_arguments": {
                "debug_dataset"                 : False,
                "in_json"                       : in_json,
                "train_data_dir"                : train_data_dir,
                "dataset_repeats"               : num_repeats,
                "keep_tokens"                   : keep_tokens,
                "resolution"                    : str(resolution) + ',' + str(resolution),
                "color_aug"                     : False,
                "face_crop_aug_range"           : None,
                "token_warmup_min"              : 1,
                "token_warmup_step"             : 0,
            },
            "training_arguments": {
                "output_dir"                    : output_dir,
                "output_name"                   : project_name if project_name else "last",
                "save_precision"                : save_precision,
                "save_every_n_epochs"           : save_every_n_epochs,
                "save_n_epoch_ratio"            : None,
                "save_last_n_epochs"            : None,
                "save_state"                    : None,
                "save_last_n_epochs_state"      : None,
                "resume"                        : None,
                "train_batch_size"              : train_batch_size,
                "max_token_length"              : 225,
                "mem_eff_attn"                  : False,
                "sdpa"                          : True if optimization == "scaled dot-product attention" else False,
                "xformers"                      : True if optimization == "xformers" else False,
                "max_train_epochs"              : num_epochs,
                "max_data_loader_n_workers"     : 8,
                "persistent_data_loader_workers": True,
                "seed"                          : seed if seed > 0 else None,
                "gradient_checkpointing"        : gradient_checkpointing,
                "gradient_accumulation_steps"   : 1,
                "mixed_precision"               : mixed_precision,
            },
            "logging_arguments": {
                "log_with"          : "wandb" if wandb_api_key else "tensorboard",
                "log_tracker_name"  : project_name if wandb_api_key and not project_name == "last" else None,
                "logging_dir"       : logging_dir,
                "log_prefix"        : project_name if not wandb_api_key else None,
            },
            "sample_prompt_arguments": {
                "sample_every_n_steps"    : None,
                "sample_every_n_epochs"   : save_every_n_epochs if enable_sample else None,
                "sample_sampler"          : sampler,
            },
            "saving_arguments": {
                "save_model_as": "safetensors"
            },
        }

        def write_file(filename, contents):
            with open(filename, "w") as f:
                f.write(contents)

        def prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompts):
            if enable_sample:
                search_pattern = os.path.join(train_data_dir, '**/*' + prompt_from_caption)
                caption_files = glob.glob(search_pattern, recursive=True)

                if not caption_files:
                    if not custom_prompts:
                        custom_prompts = ["pine tree"]
                    new_prompt_config = prompt_config.copy()
                    new_prompt_config['prompt']['subset'] = [
                        {"prompt": positive_prompt + custom_prompt if positive_prompt else custom_prompt} for custom_prompt in custom_prompts
                    ]
                else:
                    selected_files = random.sample(caption_files, min(num_prompt, len(caption_files)))

                    prompts = []
                    for file in selected_files:
                        with open(file, 'r') as f:
                            prompts.append(f.read().strip())

                    new_prompt_config = prompt_config.copy()
                    new_prompt_config['prompt']['subset'] = []

                    for prompt in prompts:
                        new_prompt = {
                            "prompt": positive_prompt + prompt if positive_prompt else prompt,
                        }
                        new_prompt_config['prompt']['subset'].append(new_prompt)

                return new_prompt_config
            else:
                return prompt_config

        def eliminate_none_variable(config):
            for key in config:
                if isinstance(config[key], dict):
                    for sub_key in config[key]:
                        if config[key][sub_key] == "":
                            config[key][sub_key] = None
                elif config[key] == "":
                    config[key] = None

            return config

        optimizer_config = self.get_optimizer_config(learning_rate)
        lora_config = self.get_lora_config(network_dim, network_alpha, conv_dim, conv_alpha)
        advanced_training_config = self.get_advanced_training_config()

        train_config.update(optimizer_config)
        train_config.update(lora_config)
        train_config.update(advanced_training_config)

        prompt_config = prompt_convert(enable_sample, num_prompt, train_data_dir, prompt_config, custom_prompts)

        config_path         = os.path.join(train_path, "config_file.toml")
        prompt_path         = os.path.join(train_path, "sample_prompt.toml")
        config_str          = toml.dumps(eliminate_none_variable(train_config))
        prompt_str          = toml.dumps(eliminate_none_variable(prompt_config))
        write_file(config_path, config_str)
        write_file(prompt_path, prompt_str)

        return prompt_path, config_path

    def get_validation_outputs(self, lora_name):
        sample_path = "/home/ubuntu/vango_ComfyUI/models/loras/sample"
        sample_filenames = [f for f in os.listdir(sample_path) if os.path.isfile(os.path.join(sample_path, f)) and f.startswith(lora_name)]
        sample_files_data = []
        for filename in sample_filenames:
            epoch_match = re.search(r"_e(\d+)_", filename)
            epoch_num = int(epoch_match.group(1))

            prompt_match = re.search(r"_e\d+_(\d+)", filename)
            prompt_num = int(prompt_match.group(1))

            sample_files_data.append((filename, epoch_num, prompt_num))
        
        sample_files_data.sort(key=lambda x: (x[1], x[2]))
        print("FILES DATA", sample_files_data)

        images = []
        for filename, _, _ in sample_files_data:
            i = Image.open(os.path.join(sample_path, filename))
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)
            images.append(image)
        
        return images

    def train_lora(self, base_model, dataset, validation_prompts, model_name, token, network_dim, network_alpha, conv_dim, conv_alpha, learning_rate, epochs, repeats, batch_size, seed, save_checkpoint_every_n_epochs):
        train_path = self.extract_dataset(dataset)
        meta_lat_json_path = self.generate_latents(train_path)
        sample_prompt, config_file = self.get_training_config(validation_prompts, model_name, train_path, meta_lat_json_path, network_dim, network_alpha, conv_dim, conv_alpha, learning_rate, epochs, repeats, batch_size, seed, save_checkpoint_every_n_epochs)

        def read_file(filename):
            with open(filename, "r") as f:
                contents = f.read()
            return contents

        def train(config):
            args = ""
            for k, v in config.items():
                if k.startswith("_"):
                    args += f'"{v}" '
                elif isinstance(v, str):
                    args += f'--{k}="{v}" '
                elif isinstance(v, bool) and v:
                    args += f"--{k} "
                elif isinstance(v, float) and not isinstance(v, bool):
                    args += f"--{k}={v} "
                elif isinstance(v, int) and not isinstance(v, bool):
                    args += f"--{k}={v} "

            return args

        accelerate_conf = {
            "config_file" : "/home/ubuntu/vango_ComfyUI/custom_nodes/train/kohya-trainer/accelerate_config/config.yaml",
            "num_cpu_threads_per_process" : 1,
        }

        train_conf = {
            "sample_prompts"  : sample_prompt if os.path.exists(sample_prompt) else None,
            "config_file"     : config_file,
        }

        accelerate_args = train(accelerate_conf)
        train_args = train(train_conf)

        final_args = f"accelerate launch {accelerate_args} /home/ubuntu/vango_ComfyUI/custom_nodes/train/kohya-trainer/sdxl_train_network.py {train_args}"
        subprocess.run(final_args, shell=True)

        images = self.get_validation_outputs(model_name)
        return (images, )

NODE_CLASS_MAPPINGS = {
    "DreamboothTrainer": DreamboothTrainer,
}

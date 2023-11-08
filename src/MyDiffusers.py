from diffusers import DiffusionPipeline
from io import BytesIO
from os import path
import threading
import torch
import time
import sys
import os
import gc


images = None
pipeline = None


#_is_init = False

def init():
	global _is_init, _variant, _torch_dtype, _device, _tor_gen
	if not _is_init:
		_variant = "fp16"
		_torch_dtype = torch.float16
		#_tor_gen = None
		#_gpu0 = torch.device("cuda:0")
		#_gpu1 = torch.device("cuda:1")
		_device = "cuda"
		_tor_gen = torch.Generator(_device)
		_is_init = True

def update_pipeline(custom_pipeline):
	global pipeline
	pipeline = DiffusionPipeline.from_config(pipeline, custom_pipeline=custom_pipeline)

def build_pipeline(repository, custom_pipeline):
	if custom_pipeline: _create_pipeline(repository, custom_pipeline)
	else: _create_pipeline(repository, None)

def _create_pipeline(repository, custom_pipeline):
	global pipeline
	pipeline = DiffusionPipeline.from_pretrained(
		repository, 
		variant=_variant, 
		torch_dtype=_torch_dtype, 
		custom_pipeline=custom_pipeline,
		safety_checker=None, 
		feature_extractor=None, 
		requires_safety_checker=False
	).to(_device)
	
	#pipeline.set_progress_bar_config(disable=True)
	#pipeline.enable_xformers_memory_efficient_attention()

def generate_seed():
	return _tor_gen.seed()

def load_lora(repository):
	''' Example repository:
	online: "ostris/ikea-instructions-lora-sdxl"
	Local: "/kaggle/input/locallora"
	'''
	# weight_name is optional
	# pipeline.load_lora_weights(repository, weight_name="mylora.safetensors")
	pipeline.load_lora_weights(repository)

def unload_lora():
	pipeline.unload_lora_weights()


def fuse_lora(scale):
	'scale 0.00 to 1.00'
	pipeline.fuse_lora(lora_scale=scale)

def unfuse_lora():
	pipeline.unfuse_lora()


def clear_gpu():
	global pipeline
	del pipeline
	gc.collect()
	torch.cuda.empty_cache()
	pipeline = None

def load_lora(model_path):
	'''example pretrained model (Hugging face)
	1. goofyai/3d_render_style_xl
	2. Fictiverse/Voxel_XL_Lora
	'''
	pipeline.load_lora_weights(model_path)
	pipeline.unload_lora_weights()
	
def unload_lora():
	pipeline.unload_lora_weights()

def generate(prompt, negative_prompt, width, height, seed, guidance_scale, num_inference_steps):
	global images

	images = pipeline(
			prompt,
			negative_prompt = negative_prompt,
			width = width,
			height = height,
			generator = _tor_gen.manual_seed(seed),
			guidance_scale = guidance_scale,
			num_inference_steps = num_inference_steps
		).images

	return images



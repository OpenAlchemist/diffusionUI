from diffusers import DiffusionPipeline
from IPython.display import display, clear_output
import ipywidgets as widgets
from io import BytesIO
from os import path
import threading
import zipfile
import shutil
import torch
import time
import sys
import os
import gc





variant = "fp16"
torch_dtype = torch.float16

pipeline = None
tor_gen = None

device = "cuda"
#gpu0 = torch.device("cuda:0")
#gpu1 = torch.device("cuda:1")


is_init = False

def init():
	global is_init, tor_gen
	if is_init: return
	tor_gen = torch.Generator(device)
	is_init = True

def update_pipeline(custom_pipeline):
	global pipeline
	pipeline = DiffusionPipeline.from_config(pipeline, custom_pipeline=custom_pipeline)

def build_pipeline(repository, custom_pipeline):
	global pipeline
	pipeline = DiffusionPipeline.from_pretrained(
					repository, 
					variant=variant, 
					torch_dtype=torch_dtype, 
					custom_pipeline=custom_pipeline,
					safety_checker=None, 
					feature_extractor=None, 
					requires_safety_checker=False).to(device)
	#pipeline.set_progress_bar_config(disable=True)
	#pipeline.enable_xformers_memory_efficient_attention()


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




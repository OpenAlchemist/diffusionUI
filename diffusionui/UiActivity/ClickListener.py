from IPython.display import display, clear_output

import threading
import time
import sys
import os

import FileOperations as fileos
import MyDiffusers as diffuser
import Interface


def onclick_load_checkpoint(button):
	with Interface.log_output:
		clear_output()
		#diffuser.build_pipeline(checkpoint_et.value, custom_pipeline_et.value)
		diffuser.build_pipeline(Interface.checkpoint_et.value, None)
		clear_output()


def onclick_load_lora(button):
	with Interface.log_output:
		clear_output()
		diffuser.load_lora(Interface.lora_et.value)
		clear_output()


def onclick_unload_lora(button):
	with Interface.log_output:
		clear_output()
		diffuser.unload_lora()
		clear_output()


def onclick_generate(button):
	button.disabled = True
	button.description = "Generating..."
	if Interface.rand_seed_cb.value:
		Interface.seed_et.value = diffuser.generate_seed()

	with Interface.progress_output:
		diffuser.generate(
			Interface.prompt_et.value,
			Interface.neg_prompt_et.value,
			Interface.width_slider.value,
			Interface.height_slider.value,
			Interface.seed_et.value,
			Interface.cfg_slider.value,
			Interface.steps_slider.value
		)
		clear_output()
	button.description = "Generated!"

	image = diffuser.images[0]
	Interface.generated_iv.value = fileos.get_webp_bytes(image)
	
	button.description = "Generate"
	button.disabled = False
	fileos.save_t2i(image)


def setup_listeners():
	Interface.load_checkpoint_btn.on_click(onclick_load_checkpoint)
	Interface.generate_btn.on_click(onclick_generate)


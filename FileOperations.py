from io import BytesIO
from os import path
import threading
import zipfile
import shutil
import time
import os



output_dir = '/kaggle/working/outputs'

text2image_dir = f'{output_dir}/text2image'
image2image_dir = f'{output_dir}/image2image'
inpaint_dir = f'{output_dir}/inpaint'

zip_name = '/kaggle/working/zipped_output'
zipped_output = zip_name + '.zip'


def set_output_dir(dir_path):
	global output_dir, text2image_dir, image2image_dir, inpaint_dir
	
	output_dir = dir_path
	text2image_dir = f'{output_dir}/text2image'
	image2image_dir = f'{output_dir}/image2image'
	inpaint_dir = f'{output_dir}/inpaint'

def create_dir(dir_path):
	if not os.path.exists(dir_path): 
		os.makedirs(dir_path)

def create_output_dir():
    create_dir(text2image_dir)
    create_dir(image2image_dir)
    create_dir(inpaint_dir)

def zip_output():
	if not os.path.exists(output_dir): 
		print('No Output Dir')
		return
	print('Zipping...')
	shutil.make_archive(zip_name, 'zip', output_dir)
	print('Successfully Zipped')
	
def delete_output():
	print('Clearing Output...')
	shutil.rmtree(output_dir, ignore_errors=True, onerror=None)
	print('Output Cleared')
	create_output_dir()

def remove_zip():
	print('Removing Zip...')
	os.remove(zipped_output)
	print('Zip Removed')


def get_webp_bytes(pil_image):
	buffered = BytesIO()
	pil_image.save(buffered, format='webp')
	return buffered.getvalue()

def save_t2i(image):
	output_image = f'{text2image_dir}/image-{round(time.time())}.png'
	image.save(output_image)

def save_t2i_webp(image):
	output_image = f'{text2image_dir}/image-{round(time.time())}.webp'
	image.save(output_image, format="webp")


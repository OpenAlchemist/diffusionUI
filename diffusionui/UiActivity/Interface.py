import ipywidgets as widgets


def build_interface():
	global checkpoint_et, custom_pipeline_et, load_checkpoint_btn
	global lora_et, load_lora_btn, textual_inversion_et, hypernetwork_et
	global prompt_et, neg_prompt_et
	global steps_slider, width_slider, height_slider, cfg_slider, seed_et, rand_seed_cb
	global generate_btn, progress_output, generated_iv , ui_container, log_output

	checkpoint_et = widgets.Text(description="Checkpoint:", value='/kaggle/input/photovision-xl')
	custom_pipeline_et = widgets.Text(description="Pipeline:", value='lpw_stable_diffusion', placeholder='Custom Community Pipeline')
	load_checkpoint_btn = widgets.Button(description="Load")

	lora_et = widgets.Text(description="Lora:")
	load_lora_btn = widgets.Button(description="Load")

	textual_inversion_et = widgets.Text(description="Textual Inversion:")
	hypernetwork_et = widgets.Text(description="Hypernetwork:")


	models_container = widgets.Tab()
	models_container.children = [
		widgets.VBox([
			checkpoint_et,
			#custom_pipeline_et, 
			load_checkpoint_btn
		]), 
		textual_inversion_et, hypernetwork_et,
		widgets.VBox([
			lora_et,
			load_lora_btn
		])
	]
	models_container.set_title(0, "Checkpoint")
	models_container.set_title(1, "Textual Inversion")
	models_container.set_title(2, "Hypernetworks")
	models_container.set_title(3, "Lora")

	setup_models_container = widgets.Accordion(children=[models_container])
	setup_models_container.set_title(0, 'Setup Models')
	setup_models_container.selected_index = 0


	# Create prompt_container
	prompt_et = widgets.Text(value="a beautiful girl,full body, Realism, highly detailed, 8k wallpaper", description="Propmt:")
	neg_prompt_et = widgets.Text(value="cartoon, anime, sketch, ugly, blurry, bad face, bad anatomy, painting", description="Negative Prompt:")


	steps_slider = widgets.IntSlider(value=10, min=5, max=35, description="Steps:")
	width_slider = widgets.IntSlider(value=720, min=512, max=1280, description="Width:")
	height_slider = widgets.IntSlider(value=1280, min=512, max=1280, description="Height:")
	cfg_slider = widgets.FloatSlider(value=7.5, min=1, max=15, step=0.5, description="CFG Scale:")
	seed_et = widgets.IntText(value=0, description="Seed:")

	rand_seed_cb = widgets.Checkbox(
		value = True,
		description = 'Use Random Seed',
		indent = False
	)


	generate_btn = widgets.Button(description="Generate")
	progress_output = widgets.Output()
	log_output = widgets.Output()

	
	# Create an image widget
	generated_iv = widgets.Image(format='webp', width=300, height=300)


	tab1_Text2Image = widgets.VBox([
		widgets.HTML("Used for Text2Image")
	])
	tab2_Image2Image = widgets.VBox([
		widgets.HTML("Used for Image2Image")
	])

	tab_container = widgets.Tab()
	tab_container.children = [tab1_Text2Image, tab2_Image2Image]
	tab_container.set_title(0, "Text2Image")
	tab_container.set_title(1, "Image2Image")

	ui_container = widgets.VBox([
		setup_models_container,
		tab_container,
		widgets.HBox([
			widgets.VBox([
				prompt_et,
				neg_prompt_et,
				steps_slider, 
				width_slider, 
				height_slider,
				cfg_slider,
				widgets.HBox([seed_et, rand_seed_cb])
			]),
			widgets.VBox([
				generate_btn, 
				progress_output,
				generated_iv
			])
		]),
		log_output
	])



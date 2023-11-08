from IPython.display import display, clear_output
import threading
import time
import sys
import os

import FileOperations as fileos
import MyDiffusers as diffuser
import Interface
import ClickListener


is_init = False

def init():
	global is_init
	if is_init: return
	diffuser.init()
	fileos.create_output_dir()
	Interface.build_interface()
	ClickListener.setup_listeners()
	is_init = True


def display_ui():
	display(Interface.ui_container)



#image
#fileos.zip_output()
#fileos.delete_output()
#fileos.remove_zip()


import subprocess
from pathlib import Path
import os
import sys
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.helpers import ASCIIColors, trace_exception
from lollms.personality import APScript, AIPersonality
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class Processor(APScript):
    """
    A class that processes model inputs and outputs.

    Inherits from APScript.
    """
    def __init__(
                 self, 
                 personality: AIPersonality,
                 callback = None,
                ) -> None:
        # Get the current directory
        root_dir = personality.lollms_paths.personal_path
        # We put this in the shared folder in order as this can be used by other personalities.
        shared_folder = root_dir/"shared"
        self.sd_folder = shared_folder / "auto_sd"
        
        self.callback = None
        self.sd = None
        self.previous_sd_positive_prompt = None
        self.sd_negative_prompt = None

        personality_config_template = ConfigTemplate(
            [
                {"name":"device","type":"str","value":"cuda" if personality.config.enable_gpu else "cpu",'options':['cpu','cuda'],"help":"Imagine the images"},
            ]
            )
        personality_config_vals = BaseConfig.from_template(personality_config_template)

        personality_config = TypedConfig(
            personality_config_template,
            personality_config_vals
        )
        super().__init__(
                            personality,
                            personality_config,
                            callback=callback
                        )
        self.model = None

    def install(self):
        super().install()
        try:
            print("Checking pytorch")
            import torch
            import torchvision
            if torch.cuda.is_available():
                print("CUDA is supported.")
            else:
                print("CUDA is not supported. Reinstalling PyTorch with CUDA support.")
                self.reinstall_pytorch_with_cuda()
        except Exception as ex:
            self.reinstall_pytorch_with_cuda()
        
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")

    def prepare(self):
        if self.model is None:
            self.new_message("",MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI)
            self.step_start("Loading Blip")
            self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.personality_config.device)
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.personality_config.device)
            self.step_end("Loading Blip")
            self.finished_message()


    def add_file(self, path, callback=None):
        self.prepare()
        if callback is None and self.callback is not None:
            callback = self.callback
        try:
            self.new_message("", MSG_TYPE.MSG_TYPE_CHUNK)
            pth = str(path).replace("\\","/").split('/')
            idx = pth.index("uploads")
            pth = "/".join(pth[idx:])
            file_path = f"![](/{pth})\n"


            # only one path is required
            self.raw_image = Image.open(path).convert('RGB')
            self.files = [path]
            inputs = self.processor(self.raw_image, return_tensors="pt").to(self.personality_config.device) #"cuda")
            def local_callback(output):
                token = output.argmax(dim=-1)
                token_str = self.processor.decode(token)
                self.full(token_str, callback=callback)
                print(token_str, end='')
            print("Processing...")
            output = self.processor.decode(self.model.generate(**inputs, max_new_tokens=self.personality.model_n_predicts)[0], skip_special_tokens=True, callback=local_callback)
            print("Image description: "+output)
            self.full(f"File added successfully\nImage description :\n{output}\nImage:\n!{file_path}", callback=callback)
            self.finished_message()
            return True
        except Exception as ex:
            trace_exception(ex)
            print("Couldn't load file. PLease check the profided path.")
            return False

    def remove_file(self, path):
        # only one path is required
        self.files = []

    def remove_text_from_string(self, string, text_to_find):
        """
        Removes everything from the first occurrence of the specified text in the string (case-insensitive).

        Parameters:
        string (str): The original string.
        text_to_find (str): The text to find in the string.

        Returns:
        str: The updated string.
        """
        index = string.lower().find(text_to_find.lower())

        if index != -1:
            string = string[:index]

        return string
    
    def process(self, text):
        bot_says = self.bot_says + text
        antiprompt = self.personality.detect_antiprompt(bot_says)
        if antiprompt:
            self.bot_says = self.remove_text_from_string(bot_says,antiprompt)
            print("Detected hallucination")
            return False
        else:
            self.bot_says = bot_says
            return True

    def generate(self, prompt, max_size):
        self.bot_says = ""
        return self.personality.model.generate(
                                prompt, 
                                max_size, 
                                self.process,
                                temperature=self.personality.model_temperature,
                                top_k=self.personality.model_top_k,
                                top_p=self.personality.model_top_p,
                                repeat_penalty=self.personality.model_repeat_penalty,
                                ).strip()    
        

    def run_workflow(self, prompt, previous_discussion_text="", callback=None):
        """
        Runs the workflow for processing the model input and output.

        This method should be called to execute the processing workflow.

        Args:
            generate_fn (function): A function that generates model output based on the input prompt.
                The function should take a single argument (prompt) and return the generated text.
            prompt (str): The input prompt for the model.
            previous_discussion_text (str, optional): The text of the previous discussion. Default is an empty string.
            callback a callback function that gets called each time a new token is received
        Returns:
            None
        """
        self.callback = callback
        self.prepare()
        try:
            inputs = self.processor(self.raw_image, f"{previous_discussion_text}{self.personality.link_text}{self.personality.ai_message_prefix}", return_tensors="pt").to(self.personality_config.device) #"cuda")
            def local_callback(output):
                token = output.argmax(dim=-1)
                token_str = self.processor.decode(token)
                if callback is not None:
                    callback(token_str, MSG_TYPE.MSG_TYPE_CHUNK)
                else:
                    print(token_str, end='')


            output = self.processor.decode(self.model.generate(**inputs, max_new_tokens=self.personality.model_n_predicts)[0], skip_special_tokens=True, callback=local_callback)
        except Exception as ex:
            print(ex)
            trace_exception(ex)
            output = "There seems to be a problem with your image, please upload a valid image to talk about"
        
        self.full(output)
        return output



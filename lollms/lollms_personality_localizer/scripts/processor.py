import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PromptReshaper, git_pull
import re
import importlib
import requests
from tqdm import tqdm
import shutil
import yaml
import urllib.parse
from typing import Callable

# Flask is needed for ui functionalities
from flask import request, jsonify

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
        self.word_callback = None
        self.sd = None
        if personality.app is not None:
            options = [p.name for p in personality.app.mounted_personalities]
        else:
            options = []
        personality_config_template = ConfigTemplate(
            [
                {"name":"model_name","type":"str","value":"DreamShaper_5_beta2_noVae_half_pruned.ckpt", "help":"Name of the model to be loaded for stable diffusion generation"},
                {"name":"sampler_name","type":"str","value":"Euler a", "options":["Euler a","Euler","LMS","Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM++ 2M SDE", "DPM fast", "DPM adaptive", "DPM Karras", "DPM2 Karras", "DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DPM++ 2M SDE Karras" ,"DDIM", "PLMS","UniPC"], "help":"Select the sampler to be used for the diffusion operation. Supported samplers ddim, dpms, plms"},                
                {"name":"ddim_steps","type":"int","value":50, "min":10, "max":1024},
                {"name":"scale","type":"float","value":7.5, "min":0.1, "max":100.0},
                {"name":"steps","type":"int","value":50, "min":10, "max":1024},                
                {"name":"W","type":"int","value":512, "min":10, "max":2048},
                {"name":"H","type":"int","value":512, "min":10, "max":2048},
                {"name":"skip_grid","type":"bool","value":True,"help":"Skip building a grid of generated images"},
                {"name":"img2img_denoising_strength","type":"float","value":7.5, "min":0.01, "max":1.0, "help":"The image to image denoising strength"},
                {"name":"batch_size","type":"int","value":1, "min":1, "max":100,"help":"Number of images per batch (requires more memory)"},
                {"name":"num_images","type":"int","value":1, "min":1, "max":100,"help":"Number of batch of images to generate (to speed up put a batch of n and a single num images, to save vram, put a batch of 1 and num_img of n)"},
                {"name":"seed","type":"int","value":-1},
                {"name":"max_generation_prompt_size","type":"int","value":512, "min":10, "max":personality.config["ctx_size"]},
                
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
        self.sd = None
        self.assets_path = None

    def install(self):
        super().install()
        
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      

        ASCIIColors.success("Installed successfully")



    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, context_details:dict=None):
        """
        This function generates code based on the given parameters.

        Args:
            full_prompt (str): The full prompt for code generation.
            prompt (str): The prompt for code generation.
            context_details (dict): A dictionary containing the following context details for code generation:
                - conditionning (str): The conditioning information.
                - documentation (str): The documentation information.
                - knowledge (str): The knowledge information.
                - user_description (str): The user description information.
                - discussion_messages (str): The discussion messages information.
                - positive_boost (str): The positive boost information.
                - negative_boost (str): The negative boost information.
                - force_language (str): The force language information.
                - fun_mode (str): The fun mode conditionning text
                - ai_prefix (str): The AI prefix information.
            n_predict (int): The number of predictions to generate.
            client_id: The client ID for code generation.
            callback (function, optional): The callback function for code generation.

        Returns:
            None
        """

        self.callback = callback
        output_path:Path = self.personality.lollms_paths.personal_outputs_path / self.personality.personality_folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        # First we create the yaml file
        # ----------------------------------------------------------------
        self.step_start("Fetching personality")
        options = [p.personality_folder_name.lower() for p in self.personality.app.mounted_personalities]
        choice_index = self.multichoice_question(f"What is the name of the personality requested by the user in english out of the provided ones?", options, f"!@>user request:{prompt}")
        name = self.personality.app.mounted_personalities[choice_index].personality_folder_name.lower()
        self.step_end("Fetching personality")
        name = re.sub(r'[\\/:*?"<>|]', '', name)
        ASCIIColors.yellow(f"Name:{name}")

        if name.lower() in options:
            personality:AIPersonality = self.personality.app.mounted_personalities[options.index(name)]
        else:
            self.full("Please provide me with a personality that was already mounted or mount the personality and try again.")
            return
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        self.step_start("Fetching language")
        language = self.generate(f"""{self.personality.personality_conditioning}
!@>user request:{prompt}
!@>task: What is the language of the personality requested by the user?
!@>{self.personality.ai_message_prefix}: The requested language to translate to is """,50,0.1,10,0.98).strip().split("\n")[0]
        self.step_end("Fetching language")
        language = re.sub(r'[\\/:*?"<>|]', '', language)
        language = language.lower().replace(".","").replace(",","")
        ASCIIColors.yellow(f"language:{language}")
        # ----------------------------------------------------------------        
        author = personality.author
        
        # ----------------------------------------------------------------
        version = personality.version
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        category = personality.category
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        self.step_start("Translating description")
        description = self.generate(f"""{self.personality.personality_conditioning}
!@>request:{prompt}
!@>personality name:{name}
!@>task: Translate the personality description to {language}
description: {personality.personality_description}
{self.personality.ai_message_prefix}
description translation:""",256,0.1,10,0.98).strip() 
        self.step_end("Translating description")
        ASCIIColors.yellow(f"Description:{description}")
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        if personality.disclaimer!="":
            self.step_start("Translating disclaimer")
            disclaimer = self.generate(f"""{self.personality.personality_conditioning}
!@>request:{prompt}
!@>personality name:{name}
!@>task: Translate the disclaimer to {language}
disclaimer: {personality.disclaimer}
{self.personality.ai_message_prefix}
disclaimer translation:""",256,0.1,10,0.98).strip()  
            self.step_end("Translating disclaimer")
            ASCIIColors.yellow(f"Disclaimer:{disclaimer}")
        else:
            disclaimer = ""
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        self.step_start("Translate the conditionning")
        conditionning_translation_prompt = f"""!@>request:{prompt}
!@>personality name:{name}
!@>task: Translate the conditionning of the AI to {language}.
{self.personality.ai_message_prefix}
conditionning:{personality.personality_conditioning}
!@>lollms_personality_maker:
conditionning translation: !@>instruction:"""
        ASCIIColors.yellow(conditionning_translation_prompt)
        conditioning = self.generate(conditionning_translation_prompt,256,0.1,10,0.98).strip()
        self.step_end("Translate the conditionning")
        ASCIIColors.yellow(f"Conditioning:{conditioning}")
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        self.step_start("Translating welcome message")
        welcome_message = self.generate(f"""{self.personality.personality_conditioning}
!@>personality name:{name}
!@>task: Translate the welcome message to {language}.
!@>welcome message: {personality.welcome_message}
!@>{self.personality.ai_message_prefix}:
translated welcome message:
""",256,0.1,10,0.98).strip()          
        self.step_end("Translating welcome message")
        ASCIIColors.yellow(f"Welcome message:{welcome_message}")
        # ----------------------------------------------------------------
                         
        # ----------------------------------------------------------------
        self.step_start("Building the yaml file")
        cmt_desc = "\n## ".join(description.split("\n"))
        desc = "\n    ".join(description.split("\n"))
        disclaimer = "\n    ".join(disclaimer.split("\n"))
        conditioning =  "\n    ".join(conditioning.split("\n"))
        welcome_message =  "\n    ".join(welcome_message.split("\n"))
        yaml_data=f"""## {name} Chatbot conditionning file
## Author: {author}
## Version: {version}
## Description:
## {cmt_desc}
## talking to.

# Credits
author: {author}
version: {version}
category: {category}
language: {language}
name: {name}
personality_description: |
    {desc}
disclaimer: |
    {disclaimer}

# Actual useful stuff
personality_conditioning: |
    !@>Instructions: 
    {conditioning}  
user_message_prefix: '!@>User:'
ai_message_prefix: '!@>{name.lower().replace(' ','_')}:'
# A text to put between user and chatbot messages
link_text: '\n'
welcome_message: |
    {welcome_message}
# Here are default model parameters
model_temperature: 0.6 # higher: more creative, lower: more deterministic
model_n_predicts: 8192 # higher: generates more words, lower: generates fewer words
model_top_k: 50
model_top_p: 0.90
model_repeat_penalty: 1.0
model_repeat_last_n: 40

# Recommendations
recommended_binding: ''
recommended_model: ''

# Here is the list of extensions this personality requires
dependencies: []

# A list of texts to be used to detect that the model is hallucinating and stop the generation if any one of these is output by the model
anti_prompts: ["!@>","<|end|>","<|user|>","<|system|>"]
        """



        personality_path:Path = personality.lollms_paths.personalities_zoo_path/personality.category/personality.personality_folder_name/"languages"
        personality_path.mkdir(parents=True, exist_ok=True)
        with open(personality_path/f"{language}.yaml","w", encoding="utf8") as f:
            f.write(yaml_data)

        self.step_end("Building the yaml file")
        # ----------------------------------------------------------------
        
        # Now we generate icon        
        personality_assets_path = personality_path/"assets"
        personality_assets_path.mkdir(parents=True, exist_ok=True)
        self.personality_assets_path = personality_assets_path
        
        self.word_callback = callback
        output = f"```yaml\n{yaml_data}\n```\n"
        
        self.full(output, callback)
        return output



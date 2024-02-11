"""
project: lollms
personality: # Place holder: Personality name 
Author: # Place holder: creator name 
description: # Place holder: personality description
"""
from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, MSG_TYPE
import subprocess
from typing import Callable

# Helper functions
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
        
        self.callback = None
        # Example entries
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        # Supported types:
        # str, int, float, bool, list
        # options can be added using : "options":["option1","option2"...]        
        personality_config_template = ConfigTemplate(
            [
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
                            [
                                {
                                    "name": "idle",
                                    "commands": { # list of commands (don't forget to add these to your config.yaml file)
                                        "scan_and_fix_files":self.scan_and_fix_files,
                                        "help":self.help,
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        
    def install(self):
        super().install()
        
        # requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        # subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")        

    def help(self, prompt="", full_context=""):
        self.full(self.personality.help)
    

    def process_chunk(self, title, chunk, message = ""):
        self.step_start(f"Processing {title}")
        prompt = self.build_prompt([
            "!@>system: Read the code chunk and try to detect any portential vulenerabilities. Point out the error by rewriting the code line where it occures, then propose a fix to it with a small example.",
            "!@>code:\n",
            chunk,
            "!@>analysis:\n"
        ])
        self.step_end(f"Processing {title}")
        analysis = self.fast_gen(prompt)
        message += analysis
        self.full(message)
        return message

    def scan_and_fix_files(self, prompt="", full_context=""):
        self.new_message("")
        if len(self.personality.text_files)==0:
            self.full("Please send me the files you want me to analyze through the add file button in the chat tab of Lollms-webui.")
        else:
            message =""
            for txt_pth in self.personality.text_files:
                with open(txt_pth,"r",encoding="utf-8") as f:
                    txt = f.read()
                tk = self.personality.model.tokenize(txt)
                message +=f"<h2>{txt_pth}</h2>\n"
                if len(tk)<self.personality.config.ctx_size/2:
                    message = self.process_chunk(f"{txt_pth}",txt, message)
                else:
                    self.step_start(f"Chunking file {txt_pth}")
                    cs = int(self.personality.config.ctx_size/2)
                    n = int(len(tk)/(cs))+1
                    last_pos = 0
                    chunk_id = 0
                    while last_pos<len(tk):
                        message +=f"<h3>chunk : {chunk_id+1}</h3>\n"
                        chunk = tk[last_pos:last_pos+cs]
                        last_pos= last_pos+cs
                        message = self.process_chunk(f"{txt_pth} chunk {chunk_id+1}/({n+1})", self.personality.model.detokenize(chunk), message)
                        chunk_id += 1
                    self.step_end(f"Chunking file {txt_pth}")



    def add_file(self, path, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, callback)

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
        self.personality.info("Generating")
        self.callback = callback
        out = self.fast_gen(previous_discussion_text)
        self.full(out)
        return out


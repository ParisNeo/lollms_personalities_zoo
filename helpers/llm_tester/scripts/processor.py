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
import json
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
                {"name":"system_message","type":"text","value":"Act as a helpful AI.", "help":"System message to use for all questions"},
                {"name":"test_file_path","type":"str","value":"", "help":"Path tp the test file to use"},
                {"name":"output_file_path","type":"str","value":"", "help":"Path tp the output file to create"},
                {"name":"models_to_test","type":"str","value":"openai/gpt-4-turbo-preview,openai/gpt-4-turbo-preview", "help":"List of coma separated models to test in format binding_name/model_name"},
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
                                        "help":self.help,
                                        "start_testing":self.start_testing
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
    
    def start_testing(self, prompt="", full_context=""):
        self.new_message("")
        msg =[]
        if self.personality_config.test_file_path=="":
            msg.append("Please set the test file path to be used in my settings first then try again. I need that file to test the AIs")
        if self.personality_config.models_to_test=="":
            msg.append("Please set the list of models to be tested first (in form binding_name/model_name) It is case sensitive so be careful.")
        self.full("\n".join(msg))
        if len(msg)>0:
            return
        models_list = [{"binding": entry.split('/')[0].strip(), "model": entry.split('/')[1].strip()} for entry in self.personality_config.models_to_test.split(",")]
        with open(self.personality_config.test_file_path,"r",encoding="utf-8", errors="ignore") as f:
            prompts = json.load(f)

        for model in models_list:
            self.step_start(f'Started testing model {model["binding"]}/{model["model"]}')
            self.select_model(model["binding"], model["model"])
            for prompt in prompts:
                reworked_prompt = f"!@>system:{self.personality_config.system_message}\n!@>prompt:{prompt['prompt']}\n!@>assistant:"
                answer = self.fast_gen(reworked_prompt, callback=self.sink)
                prompt[f'answer_{model["binding"]}_{model["model"]}']=answer
            self.step_end(f'Started testing model {model["binding"]}/{model["model"]}')

        with open(self.personality_config.output_file_path,"w",encoding="utf-8", errors="ignore") as f:
            json.dump(prompts, f, indent=4)
    
    def add_file(self, path, client, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, client, callback)

    from lollms.client_session import Client
    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, context_details:dict=None, client:Client=None):
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


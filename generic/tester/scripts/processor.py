from fastapi import APIRouter, Request
from typing import Dict, Any
import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import json
from typing import Callable
import shutil
import yaml



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

        personality_config_template = ConfigTemplate(
            [
                {"name":"test_boolean","type":"bool","value":True,"help":"Test a boolean checkbox"},
                {"name":"test_multichoices","type":"str","value":"Euler a", "options":["Choice 1","Choice 2","Choice 3"], "help":"Tests multichoices select"},                
                {"name":"test_float","type":"float","value":7.5, "min":0.01, "max":1.0, "help":"tests float"},
                {"name":"test_int","type":"int","value":50, "min":10, "max":1024, "help":"tests int"},
                {"name":"test_str","type":"str","value":"test", "help":"tests string"},
                
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
                                    "commands": { # list of commands
                                        "help":self.help,
                                        "test_new_message":self.test_new_message,
                                        "test_goto_state1":self.test_goto_state1,
                                    },
                                    "default": self.idle
                                }, 
                                {
                                    "name": "state1",
                                    "commands": { # list of commands
                                        "help":self.help,
                                        "test_goto_idle":self.test_goto_idle,
                                    },
                                    "default": self.state1
                                },                                                           
                            ],
                            callback=callback
                        )
        
    def install(self):
        super().install()
        
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      

        ASCIIColors.success("Installed successfully")



    def help(self, prompt, full_context):
        self.full(self.personality.help)
    
    def test_new_message(self, prompt, full_context):
        self.new_message("Starting fresh :)", MSG_TYPE.MSG_TYPE_FULL)
        
    def test_goto_state1(self, prompt, full_context):
        self.goto_state("state1")
        
    def test_goto_idle(self, prompt, full_context):
        self.goto_state("idle")
        
        
        
    def add_file(self, path, client, callback=None):
        if callback is None and self.callback is not None:
            callback = self.callback
        ASCIIColors.yellow("Testing add file")
        super().add_file(path, client, callback)




    async def handle_request(self, request: Request) -> Dict[str, Any]:
        """
        Handle client requests.

        Args:
            data (dict): A dictionary containing the request data.

        Returns:
            dict: A dictionary containing the response, including at least a "status" key.

        This method should be implemented by a class that inherits from this one.

        Example usage:
        ```
        handler = YourHandlerClass()
        request_data = {"command": "some_command", "parameters": {...}}
        response = await handler.handle_request(request_data)
        ```
        """
        data = (await request.json())
        personality_subpath = data['personality_subpath']
        logo_path = data['logo_path']
        assets_path:Path = self.personality.lollms_paths.personalities_zoo_path / "personal" / personality_subpath / "assets"

        shutil.copy(logo_path, assets_path/"logo.png")
        return {"status":True}


    def make_selectable_photo(self, image_id, image_source, params=""):
        return f"""
        <div class="flex items-center cursor-pointer justify-content: space-around">
            <img id="{image_id}" src="{image_source}" alt="Artbot generated image" class="object-cover cursor-pointer" style="width:300px;height:300px" onclick="console.log('Selected');"""+"""
            fetch('/post_to_personality', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({"""+f"""
                {params}"""+"""})
            })">
        </div>
        """
    # States ================================================
    def idle(self, prompt, full_context):    
         self.full("testing responses and creating new json message")
         self.new_message("json data", MSG_TYPE.MSG_TYPE_JSON_INFOS,{'test':{'value':1,'value2':2},'test2':['v1','v2']})
         file_id = 721
         personality_path:Path = self.personality.lollms_paths.personal_outputs_path / self.personality.personality_folder_name
         personality_path="/".join(str(personality_path).replace('\\','/').split('/')[-2:])
         pth = "outputs/sd/Artbot_721.png"
         self.ui('<img src="outputs/sd/Artbot_721.png">')
         self.new_message(self.make_selectable_photo("721", "outputs/sd/Artbot_721.png", params="param1:0"), MSG_TYPE.MSG_TYPE_UI)
         self.new_message("Testing generation", MSG_TYPE.MSG_TYPE_FULL)
         out = self.generate("explain what is to be human",50,callback = self.callback)
         self.full(out)
        
    def state1(self, prompt, full_context):    
         self.full("testing responses from state 1", callback=self.callback)
    

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
                - current_language (str): The force language information.
                - fun_mode (str): The fun mode conditionning text
                - ai_prefix (str): The AI prefix information.
            n_predict (int): The number of predictions to generate.
            client_id: The client ID for code generation.
            callback (function, optional): The callback function for code generation.

        Returns:
            None
        """

        self.callback = callback
        self.process_state(prompt, previous_discussion_text, callback)

        return ""


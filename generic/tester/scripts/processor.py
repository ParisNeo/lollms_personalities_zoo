import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import json


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
        
        
        
    def add_file(self, path, callback=None):
        if callback is None and self.callback is not None:
            callback = self.callback
        ASCIIColors.yellow("Testing add file")
        super().add_file(path)


    # States ================================================
    def idle(self, prompt, full_context):    
         self.full("testing responses and creating new json message")
         self.new_message("json data", MSG_TYPE.MSG_TYPE_JSON_INFOS,{'test':{'value':1,'value2':2},'test2':['v1','v2']})
        
    def state1(self, prompt, full_context):    
         self.full("testing responses from state 1", callback=self.callback)
    

    def run_workflow(self, prompt, previous_discussion_text="", callback=None):
        """
        Runs the workflow for processing the model input and output.

        This method should be called to execute the processing workflow.

        Args:
            prompt (str): The input prompt for the model.
            previous_discussion_text (str, optional): The text of the previous discussion. Default is an empty string.
            callback a callback function that gets called each time a new token is received
        Returns:
            None
        """
        self.callback = callback
        self.process_state(prompt, previous_discussion_text, callback)

        return ""


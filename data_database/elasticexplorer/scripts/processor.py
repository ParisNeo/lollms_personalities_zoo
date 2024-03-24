"""
project: lollms
personality: # Place holder: Personality name 
Author: # Place holder: creator name 
description: # Place holder: personality description
"""
from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, MSG_TYPE
from lollms.databases.discussions_database import Discussion
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
                {"name":"server","type":"str","value":"https://localhost:9200", "help":"List of addresses of the server in form of ip or host name: port"},
                {"name":"index_name","type":"str","value":"", "help":"The index to be used for querying"},
                {"name":"user","type":"str","value":"", "help":"The user name to connect to the database"},
                {"name":"password","type":"str","value":"", "help":"The password to connect to the elastic search database"},
                {"name":"max_execution_depth","type":"int","value":10, "help":"The maximum execution depth"},
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
                                    },
                                    "default": None
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

    def help(self, prompt="", full_context=""):
        self.full(self.personality.help)
    
    def add_file(self, path, client, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, client, callback)

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
        header_text = f"!@>Extra infos:"
        header_text += f"server:{self.personality_config.server}\n"
        if self.personality_config.user!="" and self.personality_config.password!="":
            header_text += f"user:{self.personality_config.user}\n"
            header_text += f"password:{self.personality_config.password}\n"
            header_text += f'es = Elasticsearch("{self.personality_config.server}", http_auth=("{self.personality_config.user}", "{self.personality_config.password}"), verify_certs=False)'
        else:
            header_text += f'es = Elasticsearch("{self.personality_config.server}", verify_certs=False)'

        execution_output = ""
        repeats=0
        while repeats<self.personality_config.max_execution_depth:
            repeats += 1
            prompt = self.build_prompt(
                [
                    header_text,
                    context_details["conditionning"],
                    context_details["discussion_messages"],
                    "\n!@>ElasticExplorer:",
                    execution_output,
                ],
                2
            )
            out = self.fast_gen(prompt, callback=self.sink)
            self.full(out)
            self.chunk("")
            context_details["discussion_messages"] += "!@>ElasticExplorer:\n"+ out
            code_blocks = self.extract_code_blocks(out)
            execution_output = ""
            if len(code_blocks)>0:
                self.step_start("Executing code")
                for i in range(len(code_blocks)):
                    if code_blocks[i]["type"]=="python":
                        code = code_blocks[i]["content"].replace("\_","_")
                        discussion:Discussion = self.personality.app.session.get_client(context_details["client_id"]).discussion 
                        try:
                            output = self.execute_python(code, discussion.discussion_folder)
                        except Exception as ex:
                            output = ex
                        execution_output += f"Output of script {i}:\n" + output +"\n!@>ElasticExplorer:"
                self.step_end("Executing code")
            else:
                break

        self.full(out)

        return out


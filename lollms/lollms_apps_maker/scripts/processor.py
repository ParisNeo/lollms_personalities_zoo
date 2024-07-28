"""
Project: LoLLMs
Personality: # Placeholder: Personality name (e.g., "Science Enthusiast")
Author: # Placeholder: Creator name (e.g., "ParisNeo")
Description: # Placeholder: Personality description (e.g., "A personality designed for enthusiasts of science and technology, promoting engaging and informative interactions.")
"""

from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, MSG_TYPE
from lollms.client_session import Client
from lollms.functions.generate_image import build_image, build_image_function
from lollms.functions.select_image_file import select_image_file_function
from lollms.functions.take_a_photo import take_a_photo_function

from lollms.utilities import discussion_path_to_url
import subprocess
from typing import Callable
from functools import partial
from ascii_colors import trace_exception
import yaml
from pathlib import Path
import shutil

class Processor(APScript):
    """
    Defines the behavior of a personality in a programmatic manner, inheriting from APScript.
    
    Attributes:
        callback (Callable): Optional function to call after processing.
    """
    
    def __init__(
                 self, 
                 personality: AIPersonality,
                 callback: Callable = None,
                ) -> None:
        """
        Initializes the Processor class with a personality and an optional callback.

        Parameters:
            personality (AIPersonality): The personality instance.
            callback (Callable, optional): A function to call after processing. Defaults to None.
        """
        
        self.callback = callback
        
        # Configuration entry examples and types description:
        # Supported types: int, float, str, string (same as str, for back compatibility), text (multiline str),
        # btn (button for special actions), bool, list, dict.
        # An 'options' entry can be added for types like string, to provide a dropdown of possible values.
        personality_config_template = ConfigTemplate(
            [
                {"name":"app_path", "type":"string", "value":"", "help":"The current app folder. Please make sure this is the same as the one you are editing if you want to edit text"},
                # Boolean configuration for enabling scripted AI
                #{"name":"make_scripted", "type":"bool", "value":False, "help":"Enables a scripted AI that can perform operations using python scripts."},
                
                # String configuration with options
                #{"name":"response_mode", "type":"string", "options":["verbose", "concise"], "value":"concise", "help":"Determines the verbosity of AI responses."},
                
                # Integer configuration example
                #{"name":"max_attempts", "type":"int", "value":3, "help":"Maximum number of attempts for retryable operations."},
                
                # List configuration example
                #{"name":"favorite_topics", "type":"list", "value":["AI", "Robotics", "Space"], "help":"List of favorite topics for personalized responses."}
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
                            states_list=[
                                {
                                    "name": "idle",
                                    "commands": {
                                        "help": self.help, # Command triggering the help method
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        self.app_path = None

    def mounted(self):
        """
        triggered when mounted
        """
        pass


    def selected(self):
        """
        triggered when selected
        """
        pass
        # self.play_mp3(Path(__file__).parent.parent/"assets"/"borg_threat.mp3")


    # Note: Remember to add command implementations and additional states as needed.

    def install(self):
        """
        Install the necessary dependencies for the personality.

        This method is responsible for setting up any dependencies or environment requirements
        that the personality needs to operate correctly. It can involve installing packages from
        a requirements.txt file, setting up virtual environments, or performing initial setup tasks.
        
        The method demonstrates how to print a success message using the ASCIIColors helper class
        upon successful installation of dependencies. This step can be expanded to include error
        handling and logging for more robust installation processes.

        Example Usage:
            processor = Processor(personality)
            processor.install()
        
        Returns:
            None
        """        
        super().install()
        # Example of implementing installation logic. Uncomment and modify as needed.
        # requirements_file = self.personality.personality_package_path / "requirements.txt"
        # subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")

    def help(self, prompt="", full_context=""):
        """
        Displays help information about the personality and its available commands.

        This method provides users with guidance on how to interact with the personality,
        detailing the commands that can be executed and any additional help text associated
        with those commands. It's an essential feature for enhancing user experience and
        ensuring users can effectively utilize the personality's capabilities.

        Args:
            prompt (str, optional): A specific prompt or command for which help is requested.
                                    If empty, general help for the personality is provided.
            full_context (str, optional): Additional context information that might influence
                                          the help response. This can include user preferences,
                                          historical interaction data, or any other relevant context.

        Example Usage:
            processor = Processor(personality)
            processor.help("How do I use the 'add_file' command?")
        
        Returns:
            None
        """
        # Example implementation that simply calls a method on the personality to get help information.
        # This can be expanded to dynamically generate help text based on the current state,
        # available commands, and user context.
        self.full(self.personality.help)


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
        choices = self.multichoice_question("select the best suited option", [
                "The user is discussing",
                "The user is asking to build the webapp",
                "The user is asking for a modification or reporting a bug in the weapp",
                "The user is asking for a modification of the informaiton (description, author, vertion etc)",
        ], context_details["discussion_messages"])
        if choices ==0:
            self.answer(context_details)
        elif choices ==1:
            out = ""
            # ----------------------------------------------------------------
            self.step_start("Building description.yaml")
            crafted_prompt = self.build_prompt(
                [
                    self.system_full_header,
                    "you are application description file maker. Your objective is to build the description.yaml file for a specific lollms application.",
                    "The user describes a web application and the ai should build the yaml file and return it inside a yaml markdown tag",
                    f"""
name: Give a name to the application using the user provided information
description: Here you can make a detailed description of the application
version: 1.0
author: make the user the author
model: {self.personality.model.model_name}
disclaimer: If needed, write a disclaimer. else return an empty text
""",
                    "If the user explicitely proposed a name, respond with that name",
                    self.system_custom_header("context"),
                    context_details["discussion_messages"],
                    self.system_custom_header("description file")
                ],6
            )
            description_file_name = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink)
            codes = self.extract_code_blocks(description_file_name)
            if len(codes)>0:
                infos = yaml.safe_load(codes[0]["content"])
                app_path:Path = self.personality.lollms_paths.apps_zoo_path/infos["name"].replace(" ","_")
                app_path.mkdir(parents=True, exist_ok=True)
                with open(app_path/"description.yaml","w") as f:
                    yaml.safe_dump(infos,f)
                out += f"description file:\n```yaml\n{codes[0]['content']}"+"\n```\n"
            self.full_invisible_to_ai(out)
            self.step_end("Building description.yaml")

            self.step_start("Building index.html")
            crafted_prompt = self.build_prompt(
                [
                    self.system_full_header,
                    "you are web application maker. Your objective is to build the index.html file for a specific lollms application.",
                    "The user describes a web application and the ai should build a single index.html file for the application",
                    "Make sure the application is visually appealing and try to use reactive design with tailwindcss",
                    "The output must be in a html markdown code tag",
                    self.system_custom_header("context"),
                    context_details["discussion_messages"],
                    self.system_custom_header("application description file maker")
                ],6
            )
            name = self.generate(crafted_prompt,temperature=0.1, top_k=10, top_p=0.98, debug=True, callback=self.sink)
            codes = self.extract_code_blocks(name)
            if len(codes)>0:
                code = codes[0]["content"]
                with open(app_path/"index.html","w", encoding="utf8") as f:
                    f.write(code)
                out += f"index file:\n```html\n{code}"+"\n```\n"

            self.step_end("Building index.html")
            shutil.copy(Path(__file__).parent.parent/"assets"/"icon.png", app_path/"icon.png")
            self.full_invisible_to_ai(out)
        else:
            self.answer("Info: Editing is not yet possible. It will be possible in next versions"+context_details)
            

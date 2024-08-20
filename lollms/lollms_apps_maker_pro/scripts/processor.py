"""
Project: LoLLMs
Personality: # Placeholder: Personality name (e.g., "Science Enthusiast")
Author: # Placeholder: Creator name (e.g., "ParisNeo")
Description: # Placeholder: Personality description (e.g., "A personality designed for enthusiasts of science and technology, promoting engaging and informative interactions.")
"""

from lollms.types import MSG_OPERATION_TYPE
from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from lollms.client_session import Client
from lollms.functions.generate_image import build_image_from_simple_prompt
from lollms.functions.select_image_file import select_image_file_function
from lollms.functions.take_a_photo import take_a_photo_function

from lollms.utilities import discussion_path_to_url, app_path_to_url
import subprocess
from typing import Callable, Any
from functools import partial
from ascii_colors import trace_exception
import yaml
from datetime import datetime
from pathlib import Path
import shutil
import os
import shutil
import yaml
import git
import json

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
                {"name":"update_mode", "type":"str", "value":"rewrite", "options":["rewrite","edit"], "help":"The update mode specifies if the AI needs to rewrite the whole code which is a good idea if the code is not long or just update parts of the code which is more suitable for long codes."},
                {"name":"create_a_plan", "type":"bool", "value":False, "help":"Create a plan for the app before starting."},
                {"name":"generate_icon", "type":"bool", "value":False, "help":"Generate an icon for the application (requires tti to be active)."},
                {"name":"use_lollms_library", "type":"bool", "value":False, "help":"Activate this if the application requires interaction with lollms."},
                {"name":"use_lollms_tasks_library", "type":"bool", "value":False, "help":"Activate this if the application needs to use text code extraction, text summary, yes no question answering, multi choice question answering etc."},
                {"name":"use_lollms_rag_library", "type":"bool", "value":False, "help":"(not ready yet) Activate this if the application needs to use text code extraction, text summary, yes no question answering, multi choice question answering etc."},
                {"name":"use_lollms_image_gen_library", "type":"bool", "value":False, "help":"(not ready yet) Activate this if the application requires image generation."},
                {"name":"use_lollms_audio_gen_library", "type":"bool", "value":False, "help":"(not ready yet) Activate this if the application requires audio manipulation."},

                {"name":"use_lollms_localization_library", "type":"bool", "value":False, "help":"Activate this library if you want to automatically localize your application into multiple languages."},
                {"name":"use_lollms_flow_library", "type":"bool", "value":False, "help":"Activate this library if you want to use lollms flow library in your application into multiple languages."},

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
        self.application_categories = [
            "Productivity",
            "Games",
            "Communication",
            "Entertainment",
            "Finance",
            "Health & Fitness",
            "Education",
            "Travel & Navigation",
            "Utilities",
            "Creative",
            "E-commerce"
        ]
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
        self.set_message_content(self.personality.help)

    
    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str | list | None, MSG_OPERATION_TYPE, str, AIPersonality| None], bool]=None, context_details:dict=None, client:Client=None):
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
        metadata = client.discussion.get_metadata()
        if self.personality_config.use_lollms_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_client_js_info.md","r", errors="ignore") as f:
                lollms_infos = f.read()
        else:
            lollms_infos = ""
        if self.personality_config.use_lollms_rag_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_rag_info.md","r", errors="ignore") as f:
                lollms_infos += f.read()


        if self.personality_config.use_lollms_localization_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_auto_localizer.md","r", errors="ignore") as f:
                lollms_infos += f.read()

        if self.personality_config.use_lollms_flow_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_flow.md","r", errors="ignore") as f:
                lollms_infos += f.read()

        self.answer(context_details, custom_entries="Libraries infos:"+lollms_infos)

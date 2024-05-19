"""
Project: LoLLMs
Personality: # Placeholder: Personality name (e.g., "Science Enthusiast")
Author: # Placeholder: Creator name (e.g., "ParisNeo")
Description: # Placeholder: Personality description (e.g., "A personality designed for enthusiasts of science and technology, promoting engaging and informative interactions.")
"""

from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, MSG_TYPE
from lollms.utilities import discussion_path_to_url, PackageManager, find_first_available_file_index
from lollms.client_session import Client
from lollms.functions.take_screen_shot import take_screenshot
from lollms.functions.take_a_photo import take_photo
from lollms.functions.generate_image import build_image_function
import subprocess
import math
import time

from typing import Callable
from functools import partial
from ascii_colors import trace_exception

if not PackageManager.check_package_installed("pyautogui"):
    PackageManager.install_package("pyautogui")
if not PackageManager.check_package_installed("cv2"):
    PackageManager.install_package("opencv-python")

import cv2
import pyautogui


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
                # Boolean configuration for enabling scripted AI
                {"name":"image_generation_engine", "type":"string", "value":"autosd", "options":["autosd","dall-e-2","dall-e-3"], "help":"The profile name of the user. Used to store progress data."},
                {"name":"show_screenshot_ui", "type":"bool", "value":True, "help":"When taking a screenshot, if this is true then a ui will be show when the screenshot function is called"},
                {"name":"take_photo_ui", "type":"bool", "value":True, "help":"When taking a screenshot, if this is true then a ui will be show when the take photo function is called"},
                
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



    def move_mouse_to(self, x, y):
        pyautogui.moveTo(x, y)

    def click_mouse(self, x, y):
        pyautogui.click(x, y)

    def calculator_function(self, expression: str) -> float:
        try:
            # Add the math module functions to the local namespace
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            
            # Evaluate the expression safely using the allowed names
            result = eval(expression, {"__builtins__": None}, allowed_names)
            return result
        except Exception as e:
            return str(e)


        
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
        # self.process_state(prompt, previous_discussion_text, callback, context_details, client)
        prompt = self.build_prompt_from_context_details(context_details)
        # TODO: add more functions to call
        function_definitions = [
            build_image_function(self, client),
            {
                "function_name": "calculator_function",
                "function": self.calculator_function,
                "function_description": "Whenever you need to perform mathematic computations, you can call this function with the math expression and you will get the answer.",
                "function_parameters": [{"name": "expression", "type": "str"}]                
            },
            {
                "function_name": "take_screenshot",
                "function": partial(take_screenshot, use_ui=self.personality_config.show_screenshot_ui, client=client),
                "function_description": "Takes a screenshot and adds it to the discussion.",
                "function_parameters": []                
            },
            {
                "function_name": "take_photo",
                "function": partial(take_photo, use_ui=self.personality_config.take_photo_ui, client=client),
                "function_description": "Takes a photo using the webcam.",
                "function_parameters": []                
            },
            
        ]
        continue_generation = True
        while continue_generation:
            if len(self.personality.image_files)>0:
                ai_response, function_calls = self.generate_with_function_calls_and_images(prompt, self.personality.image_files, function_definitions)
            else:
                ai_response, function_calls = self.generate_with_function_calls(prompt, function_definitions)
            if len(function_calls)>0:
                outputs = self.execute_function_calls(function_calls,function_definitions)
                out = ai_response + "\n" + "\n".join([str(o) for o in outputs]) +"\n"
                self.full(out)
                if "@<NEXT>@" in ai_response: # The AI needs to get the output and regenerate
                    continue_generation = True
                    prompt += ai_response + "!@>function outputs:\n" +"\n".join([str(o) for o in outputs]) + "\n!@>"+context_details["ai_prefix"].replace("!@>","").replace(":","")+":"
                else:
                    continue_generation=False
            else:
                out = ai_response
                continue_generation=False
            self.full(out)


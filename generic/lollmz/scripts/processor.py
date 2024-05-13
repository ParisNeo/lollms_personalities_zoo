"""
Project: LoLLMs
Personality: # Placeholder: Personality name (e.g., "Science Enthusiast")
Author: # Placeholder: Creator name (e.g., "ParisNeo")
Description: # Placeholder: Personality description (e.g., "A personality designed for enthusiasts of science and technology, promoting engaging and informative interactions.")
"""

from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, MSG_TYPE
from lollms.utilities import file_path_to_url
import subprocess
from typing import Callable
from functools import partial
from ascii_colors import trace_exception
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
                {"name":"image_generation_engine", "type":"string", "value":"autosd", "options":["autosd","dall-e-2","dall-e-3"], "help":"The profile name of the user. Used to store progress data."},
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


    def build_image(self, prompt, width, height):
        try:
            if self.personality_config.image_generation_engine=="autosd":
                if not hasattr(self, "sd"):
                    from lollms.services.sd.lollms_sd import LollmsSD
                    self.step_start("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service")
                    self.sd = LollmsSD(self.personality.app, self.personality.name, max_retries=-1,auto_sd_base_url=self.personality.config.sd_base_url)
                    self.step_end("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service")
                file, infos = self.sd.paint(
                                prompt, 
                                "",
                                self.personality.image_files,
                                width = width,
                                height = height
                            )
            elif self.personality_config.image_generation_engine in ["dall-e-2", "dall-e-3"]:
                if not hasattr(self, "dalle"):
                    from lollms.services.dalle.lollms_dalle import LollmsDalle
                    self.step_start("Loading dalle service")
                    self.dalle = LollmsDalle(self.personality.app, self.personality.config.dall_e_key, self.personality_config.image_generation_engine)
                    self.step_end("Loading dalle service")
                file = self.dalle.paint(
                                prompt, 
                                width = width,
                                height = height
                            )

            file = str(file)
            escaped_url =  file_path_to_url(file)
            return f'\n![]({escaped_url})'
        except Exception as ex:
            trace_exception(ex)
            return "Couldn't generate image. Make sure Auto1111's stable diffusion service is installed"
        
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
        prompt = self.build_prompt([
            context_details["conditionning"] if context_details["conditionning"] else "",
            "!@>documentation:\n"+context_details["documentation"] if context_details["documentation"] else "",
            "!@>knowledge:\n"+context_details["knowledge"] if context_details["knowledge"] else "",
            context_details["user_description"] if context_details["user_description"] else "",
            "!@>positive_boost:\n"+context_details["positive_boost"] if context_details["positive_boost"] else "",
            "!@>negative_boost:\n"+context_details["negative_boost"] if context_details["negative_boost"] else "",
            "!@>current_language:\n"+context_details["current_language"] if context_details["current_language"] else "",
            "!@>fun_mode:\n"+context_details["fun_mode"] if context_details["fun_mode"] else "",
            "!@>discussion_window:\n"+context_details["discussion_messages"] if context_details["discussion_messages"] else "",
            "!@>"+context_details["ai_prefix"].replace("!@>","").replace(":","")+":"
        ], 
        8)
        # TODO: add more functions to call
        function_definitions = [
            {
                "function_name": "build_image",
                "function": self.build_image,
                "function_description": "Builds and shows an image from a prompt and width and height parameters. A square 1024x1024, a portrait woudl be 1024x1820 or landscape 1820x1024.",
                "function_parameters": [{"name": "prompt", "type": "str"}, {"name": "width", "type": "int"}, {"name": "height", "type": "int"}]                
            },
        ]
        if len(self.personality.image_files)>0:
            out, function_calls = self.generate_with_function_calls_and_images(prompt, self.personality.image_files, function_definitions)
        else:
            out, function_calls = self.generate_with_function_calls(prompt, function_definitions)
        if len(function_calls)>0:
            outputs = self.execute_function_calls(function_calls,function_definitions)
            out += "\n" + "\n".join([str(o) for o in outputs])
        self.full(out)


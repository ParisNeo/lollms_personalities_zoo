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
                {"name":"use_lollms_library", "type":"bool", "value":False, "help":"Activate this if the application requires interaction with lollms."},
                {"name":"generate_icon", "type":"bool", "value":False, "help":"Generate an icon for the application (requires tti to be active)."},

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
                    "you are Lollms Apps Maker. Your objective is to build the description.yaml file for a specific lollms application.",
                    "The user describes a web application and the ai should build the yaml file and return it inside a yaml markdown tag",
                    f"""
```yaml
name: Give a name to the application using the user provided information
description: Here you can make a detailed description of the application
version: 1.0
author: make the user the author
category: give a suitable category name from {self.application_categories}
model: {self.personality.model.model_name}
disclaimer: If needed, write a disclaimer. else null
```
""",
                    "If the user explicitely proposed a name, use that name",
                    "Your sole objective is to build the description.yaml file. Do not ask the user for any extra information and only respond with the yaml content in a yaml markdown tag.",
                    self.system_custom_header("context"),
                    context_details["discussion_messages"],
                    self.system_custom_header("Lollms Apps Maker")
                ],6
            )
            
            description_file_name = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink)
            codes = self.extract_code_blocks(description_file_name)
            if len(codes)>0:
                infos = yaml.safe_load(codes[0]["content"].encode('utf-8').decode('ascii', 'ignore'))
                if self.config.debug:
                    ASCIIColors.yellow("--- Description file ---")
                    ASCIIColors.yellow(infos)
                app_path:Path = self.personality.lollms_paths.apps_zoo_path/infos["name"].replace(" ","_")
                app_path.mkdir(parents=True, exist_ok=True)
                with open(app_path/"description.yaml","w") as f:
                    yaml.safe_dump(infos,f)
                out += f"description file:\n```yaml\n{codes[0]['content']}"+"\n```\n"
                self.full_invisible_to_ai(out)
                self.step_end("Building description.yaml")

                self.step_start("Building index.html")
                if self.personality_config.use_lollms_library:
                    with open(Path(__file__).parent/"lollms_client_js_info.md","r", errors="ignore") as f:
                        lollms_infos = f.read()
                else:
                    lollms_infos = ""
                crafted_prompt = self.build_prompt(
                    [
                        self.system_full_header,
                        "you are Lollms Apps Maker. Your objective is to build the index.html file for a specific lollms application.",
                        "The user describes a web application and the ai should build a single html code to fullfill the application requirements.",
                        "Make sure the application is visually appealing and try to use reactive design with tailwindcss",
                        "The output must be in a html markdown code tag",
                        "Try to write the app with sections so that updating it can be easier",
                        "in html use a html comment tag at start and end of each section",
                        "in javascript use a //section_start: and //section_end: comment tags at start and end of each section",
                        "Your sole objective is to build the index.yaml file that does what the user is asking for.",
                        "Do not ask the user for any extra information and only respond with the html content in a html markdown tag.",
                        self.system_custom_header("context"),
                        context_details["discussion_messages"],
                        "```yaml",
                        str(infos),
                        "```",                        
                        lollms_infos,
                        self.system_custom_header("Lollms Apps Maker")
                    ],6
                )
                code_content = self.generate(crafted_prompt,temperature=0.1, top_k=10, top_p=0.98, debug=True, callback=self.sink)
                if self.config.debug:
                    ASCIIColors.yellow("--- Code file ---")
                    ASCIIColors.yellow(code_content)
                codes = self.extract_code_blocks(code_content)
                if len(codes)>0:
                    code = codes[0]["content"]
                    with open(app_path/"index.html","w", encoding="utf8") as f:
                        f.write(code)
                    out += f"index file:\n```html\n{code}"+"\n```\n"
                    self.step_end("Building index.html")
                    shutil.copy(Path(__file__).parent.parent/"assets"/"icon.png", app_path/"icon.png")
                    self.full_invisible_to_ai(out)
                    self.personality_config.app_path = app_path
                    self.personality_config.save()
                else:
                    self.step_end("Building index.html", False)
                    self.full("The model you are using failed to build the index.html file. Change the prompt a bit and try again.")
            else:
                self.step_end("Building description.yaml", False)
                self.full("The model you are using failed to build the description.yaml file. Change the prompt a bit and try again.")
        elif choices ==2:
            out = ""
            self.step_start("Updating index.html")
            if self.personality_config.use_lollms_library:
                with open(Path(__file__).parent/"lollms_client_js_info.md","r", errors="ignore") as f:
                    lollms_infos = f.read()
            else:
                lollms_infos = ""

            with open(Path(self.personality_config.app_path)/"index.html","r", encoding="utf8") as f:
                code = f.read()
            crafted_prompt = self.build_prompt(
                [
                    self.system_full_header,
                    "you are Lollms Apps Maker. Your objective is to a update the html code for a specific lollms application.",
                    "The user describes a web application and gives the code the ai should update the single index.html file for the application",
                    "Make sure the application is visually appealing and try to use reactive design with tailwindcss",
                    "The output must be in a html markdown code tag",
                    "Update the code from the user suggestion",
                    "The code uses sections to help you do updates",
                    "in html a html comment tag is used at start and end of each section",
                    "in javascript a //section_start: and //section_end: comment tags are used at start and end of each section",
                    "in your update, you need to use the following syntax:",
                    "First write the section name to be changed inside a <update_section></update_section> tag, then in a code tag write the replacement code",
                    "To add a new section, rewrite the section that preceids it and then add the new section as a code",
                    "Do not rewrite the whole code. Just the needed updates",
                    self.system_custom_header("context"),
                    context_details["discussion_messages"],
                    lollms_infos,
                    self.system_custom_header("Code"),
                    "```html",
                    code,
                    "```",
                    self.system_custom_header("Lollms Apps Maker")
                ],7
            )
            name = self.generate(crafted_prompt,temperature=0.1, top_k=10, top_p=0.98, debug=True, callback=self.sink)
            
            codes = self.extract_code_blocks(name)
            if len(codes)>0:
                code = codes[0]["content"]
                with open(Path(self.personality_config.app_path)/"index.html","w", encoding="utf8") as f:
                    f.write(code)
                out += f"index file:\n```html\n{code}"+"\n```\n"

            self.step_end("Updating index.html")
            shutil.copy(Path(__file__).parent.parent/"assets"/"icon.png", Path(self.personality_config.app_path)/"icon.png")
            self.full_invisible_to_ai(out)            
            

from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PackageManager
from lollms.types import MSG_OPERATION_TYPE
from typing import Callable, Any

from pathlib import Path
from typing import List
if PackageManager.check_package_installed("pygame"):
    import pygame
else:
    PackageManager.install_package("pygame")
    import pygame

pygame.mixer.init()
import subprocess

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
        # Example entry
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        # Supported types:
        # str, int, float, bool, list
        # options can be added using : "options":["option1","option2"...]        
        personality_config_template = ConfigTemplate(
            [
                
                {"name":"nb_attempts","type":"int","value":5, "help":"Maximum number of attempts to summon a drone"},
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

    def mounted(self):
        """
        triggered when mounted
        """
        pass

    def play_mp3(self, file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def selected(self):
        """
        triggered when mounted
        """
        self.play_mp3(Path(__file__).parent.parent/"assets"/"borg_threat.mp3")

    def help(self, prompt="", full_context=""):
        self.set_message_content(self.personality.help)
    
    def add_file(self, path, client, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, client, callback)

    from lollms.client_session import Client
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
        ASCIIColors.info("Generating")
        collective:List[AIPersonality] = self.personality.app.mounted_personalities


        q_prompt = f"{self.config.start_header_id_template}You are the queen of borg.\nYou have access to the following assimilated drones:\n"
        collective_infos = ""
        for i,drone in enumerate(collective):
            collective_infos +=  f"drone id: {i}\n"
            collective_infos +=  f"drone name: {drone.name}\n"
            collective_infos +=  f"drone description: {drone.personality_description[:126]}...\n"
        q_prompt += collective_infos
        answer = ""
        q_prompt += f"You are a great leader and you know which drone is most suitable to answer the user request.\n"
        q_prompt += f"{self.config.start_header_id_template}user:{prompt}\n"
        q_prompt += f"{self.config.start_header_id_template}Queen of borg: To answer the user I summon the drone with id "
        attempts = 0



        self.step_start("Summoning collective")
        while attempts<self.personality_config.nb_attempts:
            try:
                selection = int(self.fast_gen(q_prompt, 3, show_progress=True, callback=self.sink).split()[0].split(",")[0])
                q_prompt += f"{selection}\n"
                self.step_end("Summoning collective")
                self.step(f"Selected drone {collective[selection]}")
                collective[selection].callback=callback

                if collective[selection].processor and collective[selection].name!="Queen of the Borg":
                    q_prompt += f"{self.config.start_header_id_template}sytsem:Reformulate the question for the drone.{self.config.separator_template}{self.config.start_header_id_template}Queen of borg: {collective[selection].name},"
                    reformulated_request=self.fast_gen(q_prompt, show_progress=True)
                    self.set_message_content(f"{collective[selection].name}, {reformulated_request}")
                    previous_discussion_text= previous_discussion_text.replace(prompt,reformulated_request)
                    collective[selection].new_message("")
                    collective[selection].set_message_content(f"At your service my queen.\n")
                    collective[selection].processor.text_files = self.personality.text_files
                    collective[selection].processor.image_files = self.personality.image_files
                    collective[selection].processor.run_workflow(reformulated_request, previous_discussion_text, callback, context_details, client)
                else:
                    if collective[selection].name!="Queen of the Borg":
                        q_prompt += f"{self.config.start_header_id_template}{self.config.system_message_template}{self.config.end_header_id_template}Reformulate the question for the drone.{self.config.separator_template}{self.config.start_header_id_template}Queen of borg: {collective[selection].name},"
                        reformulated_request=self.fast_gen(q_prompt, show_progress=True)
                        self.set_message_content(f"{collective[selection].name}, {reformulated_request}")
                        previous_discussion_text= previous_discussion_text.replace(prompt,reformulated_request)
                        collective[selection].new_message("")
                        collective[selection].set_message_content(f"At your service my queen.\n")
                    collective[selection].generate(previous_discussion_text,self.personality.config.ctx_size-len(self.personality.model.tokenize(previous_discussion_text)),callback=callback)
                break
            except Exception as ex:
                self.step_end("Summoning collective", False)
                attempts += 1
        return answer


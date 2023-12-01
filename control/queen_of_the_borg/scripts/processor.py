from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PackageManager
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
        self.full(self.personality.help)
    
    def add_file(self, path, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, callback)

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
        ASCIIColors.info("Generating")
        collective:List[AIPersonality] = self.personality.app.mounted_personalities


        q_prompt = "!@>You are the queen of borg.\nYou have access to the following assimilated drones:\n"
        collective_infos = ""
        for i,drone in enumerate(collective):
            collective_infos +=  f"drone id: {i}\n"
            collective_infos +=  f"drone name: {drone.name}\n"
            collective_infos +=  f"drone description: {drone.personality_description[:126]}...\n"
        q_prompt += collective_infos
        answer = ""
        q_prompt += f"You are a great leader and you know which drone is most suitable to answer the user request.\n"
        q_prompt += f"!@>user:{prompt}\n"
        q_prompt += "!@>Queen of borg: To answer the user I summon the drone with id "
        attempts = 0
        self.step_start("Summoning collective")
        while attempts<self.personality_config.nb_attempts:
            try:
                selection = int(self.fast_gen(q_prompt, 3,show_progress=True).split()[0].split(",")[0])
                self.step_end("Summoning collective")
                self.step(f"Selected drone {collective[selection]}")
                collective[selection].callback=callback
                collective[selection].new_message("")

                if collective[selection].processor:
                    collective[selection].processor.text_files = self.text_files
                    collective[selection].processor.image_files = self.image_files
                    collective[selection].processor.run_workflow(prompt, previous_discussion_text, callback)
                else:
                    collective[selection].generate(previous_discussion_text,self.personality.config.ctx_size-len(self.personality.model.tokenize(previous_discussion_text)),callback=callback)
                break
            except:
                self.step_end("Summoning collective", False)
                attempts += 1
        return answer


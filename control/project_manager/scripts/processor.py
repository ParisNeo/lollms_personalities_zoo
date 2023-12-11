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
        self.personality.info("Generating")
        members:List[AIPersonality] = self.personality.app.mounted_personalities


        q_prompt = "!@instructions>Act as a highly efficient and organized AI that excels in problem-solving and task management. It possesses excellent communication skills and can effectively coordinate and delegate tasks to a team of mounted AIs. It is proactive, adaptable, and results-oriented, ensuring that projects are executed smoothly and efficiently.\nTeam members:\n"
        collective_infos = ""
        for i,drone in enumerate(members):
            collective_infos +=  f"member id: {i}\n"
            collective_infos +=  f"member name: {drone.name}\n"
            collective_infos +=  f"member description: {drone.personality_description[:126]}...\n"
        q_prompt += collective_infos
        answer = ""
        q_prompt += f"Using the skills of some members, elaborate a plan to solve the problem exposed by the user:\n"
        q_prompt += f"!@>user:{prompt}\n"
        q_prompt += "!@>project_manager_ai: To reach the objective set by the user, here is a comprehensive plan using some of the available project members.\n1. Member id "
        attempts = 0
        self.step_start("Making plan")
        answer = self.fast_gen(q_prompt, 1024, show_progress=True)
        q_prompt += answer
        answer = "1. Member id " +answer
        self.full(answer)

        plan_steps = answer.split("\n")
        self.step_end("Making plan")
        for step in plan_steps:
            self.step_start(step)
            while attempts<self.personality_config.nb_attempts:
                val = step[step.index("id")+2:].strip()
                val = val.split(" ")[0].replace(".","").replace(",","")
                try:
                    selection = int(val)
                except Exception as ex:
                    self.error(f"{ex}")
                members[selection].callback=callback
                if members[selection].processor and members[selection].name!="project_manager":
                    q_prompt += f"!@>sytsem:Starting task by {members[selection].name}, provide details\n!@>project_manager_ai: "
                    reformulated_request=self.fast_gen(q_prompt, show_progress=True)
                    members[selection].new_message("")
                    self.full(f"{members[selection].name}, {reformulated_request}")
                    previous_discussion_text= previous_discussion_text.replace(prompt,reformulated_request)
                    members[selection].new_message("")
                    members[selection].processor.text_files = self.text_files
                    members[selection].processor.image_files = self.image_files
                    members[selection].processor.run_workflow(reformulated_request, previous_discussion_text, callback)
                else:
                    if members[selection].name!="project_manager":
                        q_prompt += f"!@>system: Reformulate the question for the member.\n!@>project_manager: {members[selection].name},"
                        reformulated_request=self.fast_gen(q_prompt, show_progress=True)
                        self.full(f"{members[selection].name}, {reformulated_request}")
                        previous_discussion_text= previous_discussion_text.replace(prompt,reformulated_request)
                        members[selection].new_message("")
                        members[selection].full(f"At your service my queen.\n")
                    members[selection].generate(previous_discussion_text,self.personality.config.ctx_size-len(self.personality.model.tokenize(previous_discussion_text)),callback=callback)
                break
            self.step_end(step)

        return answer


import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import re
import importlib
import requests
from tqdm import tqdm
import shutil
from lollms.types import GenerationPresets

class Processor(APScript):
    """
    A class that processes model inputs and outputs.

    Inherits from APScript.
    """


    def __init__(
                 self, 
                 personality: AIPersonality
                ) -> None:
        self.word_callback = None
        personality_config_template = ConfigTemplate(
            [
                {"name":"layout_max_size","type":"int","value":512, "min":10, "max":personality.config["ctx_size"]},                
            ]
            )
        personality_config_vals = BaseConfig.from_template(personality_config_template)

        personality_config = TypedConfig(
            personality_config_template,
            personality_config_vals
        )
        super().__init__(
                            personality,
                            personality_config
                        )
        self.previous_versions = []
        
    def install(self):
        super().install()
        # Get the current directory
        root_dir = self.personality.lollms_paths.personal_path
        # We put this in the shared folder in order as this can be used by other personalities.
        shared_folder = root_dir/"shared"

        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Step 2: Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])            
        ASCIIColors.success("Installed successfully")


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
        output_path = self.personality.lollms_paths.personal_outputs_path
        
        # First we create the yaml file
        # ----------------------------------------------------------------
        self.step_start("Building the title...", callback)
        title = self.generate(f"""!@>project_information:\n{prompt}
!@>task: Using the project information, Create a title for the document.
!@>title:""",512,**GenerationPresets.deterministic_preset()).strip().split("\n")[0]
        self.step_end("Building the title...", callback)
        ASCIIColors.yellow(f"title:{title}")
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        self.step_start("Building the layout...", callback)
        layout = "1. Introduction\n"+self.generate(f"""!@>project_information:\n{prompt}
!@>task: Using the project information, Let's build a layout structure for our documentation of this project.
Only write the layout, don't put any details.
Use all information from the peoject information set to elaborate a comprehensive and well organized structure.
!@>structure:
1. Introduction""",512,**GenerationPresets.deterministic_preset())
        self.step_end("Building the layout...", callback)
        ASCIIColors.yellow(f"structure:\n{layout}")
        # ----------------------------------------------------------------
        sections = [{"name": section} for section in layout.split("\n")]
        for section in sections:
            # ----------------------------------------------------------------
            self.step_start(f"Building section {section['name']}...", callback)
            section["content"] = self.generate(f"""!@>project information:
{prompt}
!@>task: Using the project information, populate the content of the section {section['name']}.
!@>instructions:
Act as a professional documentation builder and make this section content from the project information.
Use multiple lines and make a text that fits both the project information and the section title.
You must give details and be clear to make sure the reader understands the section.
!@>section title: {section['name']}
!@>section content:""",1024,**GenerationPresets.deterministic_preset())
            self.step_end(f"Building section {section['name']}...", callback)
            ASCIIColors.yellow(f"{section}\n")
            # ----------------------------------------------------------------
        
        output = f"```markdown\n# {title}\n\n"   
        output += "\n".join([f"{s['name']}\n{s['content']}\n" for s in sections])
        output += "```"
        output += "Now we can update some of the sections using the commands.(This is work in progress)"
        self.previous_versions.append(output)
        self.full(output, callback)
        
        return output



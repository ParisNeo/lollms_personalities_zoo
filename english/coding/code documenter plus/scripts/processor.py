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

    def convert_string_to_sections(self, string):
        table_of_content = ""
        lines = string.split('\n')  # Split the string into lines
        sections = []
        current_section = None
        for line in lines:
            line = line.strip()
            if line!="":
                table_of_content+=line+"\n"
                if line.startswith('## '):  # Detect section
                    section_title = line.replace('## ', '')
                    current_section = {'title': section_title, 'subsections': [], "content":""}
                    sections.append(current_section)
                elif line.startswith('### '):  # Detect subsection
                    if current_section is not None:
                        subsection_title = line.replace('### ', '')
                        current_section['subsections'].append(subsection_title)
        return sections, table_of_content

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
        title = self.generate(f"""project_information:\n{prompt}
User: Create a title that reflects the idea of this project. The title should contain at least 2 words.
Documentation builder ai:
suggested title:""",512,**GenerationPresets.deterministic_preset()).strip().split("\n")[0]
        self.step_end("Building the title...", callback)
        ASCIIColors.yellow(f"title:{title}")
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        self.step_start("Building the layout...", callback)
        layout = "## Introduction\n##"+self.generate(f"""Documentation builder aiis a tool that can understand project information and convert it to a documentation table of content.
project_information:{prompt}
User: Write a table of content for this project
Documentation builder ai: Here is the table of contents for the project documentation in markdown format:
```markdown 
# {title}
## Introduction
##""",512,**GenerationPresets.deterministic_preset())
        self.step_end("Building the layout...", callback)
        ASCIIColors.yellow(f"structure:\n{layout}")
        # ----------------------------------------------------------------
        sections, table_of_content = self.convert_string_to_sections(layout) #[{"name": section} for section in layout.split("\n")]
        for section in sections:
            # ----------------------------------------------------------------
            self.step_start(f"Building section {section['name']}...", callback)
            section["content"] = self.generate(f"""project information:
{prompt}
Table of content:

User: Using the project information, populate the content of the section {section['name']}.

!@>section title: {section['name']}
!@>section content:""",1024,**GenerationPresets.deterministic_preset())
            self.step_end(f"Building section {section['name']}...", callback)
            ASCIIColors.yellow(f"{section}\n")
            # ----------------------------------------------------------------
        
        output = f"```markdown\n# {title}\n\n"   
        output += "\n".join([f"{s['name']}\n{s['content']}\n" for s in sections])
        output += "```\n"
        output += "Now we can update some of the sections using the commands.(This is work in progress)"
        self.previous_versions.append(output)
        self.full(output, callback)
        
        return output



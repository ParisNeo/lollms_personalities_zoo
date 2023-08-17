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
                 personality: AIPersonality,
                 callback = None,
                ) -> None:
        self.word_callback = None
        personality_config_template = ConfigTemplate(
            [
                {"name":"layout_max_size","type":"int","value":512, "min":10, "max":personality.config["ctx_size"]},                
                {"name":"is_debug","type":"bool","value":False, "help":"Activates debug mode where all prompts are shown in the console"},                                
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
                            callback=callback
                            
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
        lines = string.split('\n')  # Split the string into lines
        sections = []
        current_section = None
        for line in lines:
            if line.startswith('## '):  # Detect section
                section_title = line.replace('## ', '')
                current_section = {'title': section_title, 'subsections': [], 'content':''}
                sections.append(current_section)
            elif line.startswith('### '):  # Detect subsection
                if current_section is not None:
                    subsection_title = line.replace('### ', '')
                    current_section['subsections'].append({'title':subsection_title, 'content':''})
        return sections


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
        gen_prompt = f"""Act as document title builder assistant. Infer a document title out of the project information.
project_information:
{prompt}
!@>User: Using the project information, Create a title for the document.
!@>Assistant:
Here is a suitable title for this project:"""
        if self.personality_config.is_debug:
            ASCIIColors.info(gen_prompt)

        title = self.generate(gen_prompt,512,**GenerationPresets.deterministic_preset()).strip().split("\n")[0]
        if title.startswith('"'):
            title = title[1:]
        self.step_end("Building the title...", callback)
        ASCIIColors.yellow(f"title:{title}")
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        self.step_start("Building the table of contents...", callback)
        gen_prompt = f"""Act as document table of contents builder assistant. Infer a document structure out of the project information and do not populate its content.
Do not write the sections contents. All you are asked to do is to make the table of contents which is the title, the sections and the subsections.
The table of content should be formatted in markdown format with # for main title, ## for sections and ### for subsections.
project information:
{prompt}
@!>User: Using the project information, Let's build a table of contents for our documentation of this project.
@!>Assistant:
Here is the table of contents in markdown format:
# {title}                                                  
## Introduction"""
        if self.personality_config.is_debug:
            ASCIIColors.info(gen_prompt)
        layout = "## Introduction\n"+self.generate(gen_prompt,512,**GenerationPresets.deterministic_preset())
        self.step_end("Building the table of contents...", callback)
        ASCIIColors.yellow(f"structure:\n{layout}")
        sections = self.convert_string_to_sections(layout)
        # ----------------------------------------------------------------
        
        
        
        document_text=f"# {title}\n"        
        
        
        
        for i,section in enumerate(sections):
            # ----------------------------------------------------------------
            self.step_start(f"Building section {section['title']}...", callback)
            if len(section['subsections'])>0:
                document_text += f"## {section['title']}\n"
                for subsection in section['subsections']:
                    gen_prompt = f"""Act as document section filler assistant. Infer the data for the next section of the document from the project information.
project information:
{prompt}
!@>User: Using the project information, populate the content of the section {section['title']}. Don't repeat information already stated in previous text. Only write information that is relevant to the section.
!@>previous chunk of text preview:
{document_text[-500:]}
!@>Assistant:
Here is the subsection content:
### {subsection['title']}"""
                    if self.personality_config.is_debug:
                        ASCIIColors.info(gen_prompt)
                    
                    subsection["content"] = self.generate(gen_prompt,1024,**GenerationPresets.deterministic_preset()).strip()
                    document_text += f"### {subsection['title']}\n"
                    document_text += f"{subsection['content']}\n"
            else:
                gen_prompt = f"""Act as document section filler assistant. Infer the data for the next section of the document from the project information.
project information:
{prompt}
!@>User: Using the project information, populate the content of the section {section['title']}. Don't repeat information already stated in previous text. Only write information that is relevant to the section.
!@>previous chunk of text preview:
{document_text[-500:]}
!@>Assistant:
Here is the section content:
## {section['title']}"""
                if self.personality_config.is_debug:
                    ASCIIColors.info(gen_prompt)
                
                section["content"] = self.generate(gen_prompt,1024,**GenerationPresets.deterministic_preset()).strip()
                document_text += f"## {section['title']}\n"
                document_text += f"{section['content']}\n"

            self.step_end(f"Building section {section['title']}...", callback)
            ASCIIColors.yellow(f"{section}\n")
            # ----------------------------------------------------------------
        
        output = f"```markdown\n"   
        output += document_text
        output += "\n```\n"
        output += "Now we can update some of the sections using the commands.\nYou can use the command update_section followed by the section name to add information about the section"
        
        self.previous_versions.append(output)
        if callback:
            self.full(output, callback)
        
        self.current_document = sections
        
        if callback:
            self.json(sections, callback)
        
        
        return output



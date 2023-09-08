import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PromptReshaper, git_pull
import re
import importlib
import requests
from tqdm import tqdm
import webbrowser
try:
    import pytesseract
    from PIL import Image
except:
     pass
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
        # Get the current directory
        root_dir = personality.lollms_paths.personal_path
        # We put this in the shared folder in order as this can be used by other personalities.
        
        
        self.callback = callback

        personality_config_template = ConfigTemplate(
            [
                
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
                                    },
                                    "default": self.main_process
                                },                           
                            ],
                            callback=callback
                        )

    def install(self):
        super().install()
        self.info("Please install [tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add it to the path.")
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        try:
            ASCIIColors.info("Loading pytesseract and PIL")
            import pytesseract
            from PIL import Image
        except:
            pass

    def add_file(self, path, callback=None):
        # Load an image using PIL (Python Imaging Library)
        if callback is None and self.callback is not None:
            callback = self.callback
        super().add_file(path)
        image = Image.open(self.files[-1])
        url = str(self.files[-1]).replace("\\","/").split("uploads")[-1]
        self.new_message(f'<img src="/uploads{url}">', MSG_TYPE.MSG_TYPE_UI)
        try:
            # Load an image using PIL (Python Imaging Library)
            image = Image.open(self.files[-1])

            # Use pytesseract to extract text from the image
            text = pytesseract.image_to_string(image)
            self.full("<h3>Extracted text:</h3>\n\n"+text)
        except Exception as ex:
            self.full(f"<h3>Looks like you didn't install tesseract correctly</h3><br>\n\nPlease install [tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add it to the path.\n\nException:{ex}")
        return True
    
    
    def main_process(self, initial_prompt, full_context):
        if len(self.files)==0:
            self.full("<h3>Please send an image file first</h3>")
        else:
            text = self.generate(full_context+initial_prompt,1024, callback=self.callback)
            self.full(text)
            
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
        self.process_state(prompt, previous_discussion_text, callback)

        return ""


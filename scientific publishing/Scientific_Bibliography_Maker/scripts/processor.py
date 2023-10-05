import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors
from lollms.utilities import PackageManager
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import re
import importlib
import requests
from tqdm import tqdm
import shutil
from lollms.types import GenerationPresets
import json
from functools import partial
import os

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
        self.arxiv = None
        personality_config_template = ConfigTemplate(
            [
                {"name":"project_path","type":"str","value":'', "help":"The path to the project to document"},
                {"name":"layout_max_size","type":"int","value":2048, "min":10, "max":personality.config["ctx_size"]},
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
                            [],
                            callback=callback
                        )
        self.previous_versions = []
        self.code=[]
        
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
        
        
    
    def prepare(self):
        if self.arxiv is None:
            import arxiv
            self.arxiv = arxiv

    def run_workflow(self, prompt, full_context="", callback=None):
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
        self.prepare()

        
        # Define your search query
        query = prompt

        # Specify the number of results you want
        num_results = 5

        # Specify the folder where you want to save the articles
        download_folder = "arxiv_articles"

        # Create the download folder if it doesn't exist
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        # Search for articles
        search_results = self.arxiv.query(query=query, max_results=num_results)

        # Download and save articles
        for result in search_results:
            pdf_url = result.pdf_url
            if pdf_url:
                # Get the PDF content
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    # Create the filename for the downloaded article
                    filename = os.path.join(download_folder, f"{result.id}.pdf")
                    
                    # Save the PDF to the specified folder
                    with open(filename, "wb") as file:
                        file.write(response.content)
                    print(f"Downloaded {result.title} to {filename}")
                else:
                    print(f"Failed to download {result.title}")
        



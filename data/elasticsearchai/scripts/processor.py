from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PackageManager
import subprocess

if not PackageManager.check_package_installed("elasticsearch"):
    PackageManager.install_package("elasticsearch")

from elasticsearch import Elasticsearch
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
                {"name":"server_id","type":"str","value":"['localhost:9200']", "help":"List of addresses of the server in form of ip or host name: port"},
                {"name":"search_index","type":"str","value":"your_index", "help":"The index to be used for querying"},
                
                # Specify the host and port of the Elasticsearch server
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
        es = Elasticsearch(eval(self.personality_config.server_id))
        query = self.fast_gen(previous_discussion_text+"!@>system: make an elastic search query to answer the user.\nelasticsearch_ai:\n")
        # Perform the search query
        res = es.search(index=self.personality_config.search_index, body=eval(query))

        # Process the search results
        for hit in res['hits']['hits']:
            print(hit['_source'])
        self.personality.info("Generating")
        self.callback = callback
        out = self.fast_gen(previous_discussion_text)
        self.full(out)
        return ""


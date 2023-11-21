from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from safe_store.generic_data_loader import GenericDataLoader
from safe_store.document_decomposer import DocumentDecomposer
import subprocess
from pathlib import Path
import json


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
                {"name":"logs_path","type":"str","value":"", "help":"The path to a folder containing the logs"},
                {"name":"output_file_path","type":"str","value":"", "help":"The path to a text file that will contain the final report of the AI"},
                {"name":"file_types","type":"str","value":"nips,xml,pcap,json", "help":"The extensions of files to read"},
                {"name":"chunk_size","type":"int","value":3072, "help":"The size of the chunk to read each time"},
                {"name":"chunk_overlap","type":"int","value":256, "help":"The overlap between blocs"},
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
                                        "read_all_logs":self.read_all_logs,
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

    def read_all_logs(self, prompt="", full_context=""):
        if self.personality_config.output_file_path=="":
            self.notify("Please setup output file path first")
            return
        if self.personality_config.logs_path=="":
            self.notify("Please setup logs folder path first")
            return
        self.new_message("")
        self.process_logs(
                            self.personality_config.logs_path, 
                            self.personality_config.file_types
                        )

    def process_logs(self, folder_path, extensions):
        folder = Path(folder_path)
        files = folder.glob('*')
        extension_list = [v.strip() for v in extensions.split(',')]

        output_file = open(self.personality_config.output_file_path,"w")

        for file in files:
            if file.is_file() and file.suffix[1:] in extension_list:
                self.step_start(f"Processing {file.name}")
                data = GenericDataLoader.read_file(file)
                dd = DocumentDecomposer()
                chunks = dd.decompose_document(
                                            data,
                                            self.personality_config.chunk_size,
                                            self.personality_config.chunk_overlap
                                    )
                n_chunks = len(chunks)
                for i, chunk in enumerate(chunks):
                    self.step_start(f"Processing {file.name} chunk {i+1}/{n_chunks}")
                    output = "[" + self.fast_gen(
                        f"""!@>log chunk:
{chunk}
"""+"""
!@>instructions:
Act as cyber_sentinel_AI an AI that analyzes logs and detect security breaches from the content of the log chunk.
Analyze the provided chunk of data and extract all potential security breach attempts from the chunk.
Identify any suspicious patterns or anomalies that may indicate a security breach.
Provide details about the breach attempt. If applicable mention the timestamp or ID of the log entry that triggers the detection and explain it.
Be specific and provide explanations.
If no breach detected, return an empty list []. Only report breaches and suspecious activities.
Your analysis should be detailed and provide clear evidence to support your conclusion. Remember to consider both known and unknown security threats.
Be attentive to the content of the logs and do not miss important information.
Answer in valid json format.


!@>JSON format:
[
    {
        "severity": "high","medium" or "low",
        "breach_timestamp": the timestamp of the suspicious entry,
        "breach_description": "a description of the breach"
    }
]
!@>cyber_sentinel_AI:
Here is my report as a valid json:
["""
                    )
                    self.full(output)
                    try:
                        output = output.replace('\n', '').replace('\r', '').strip()
                        if not output.endswith(']'):
                              output +="]"
                        json_output = json.loads(output)
                        for entry in json_output:
                            output_file.write(f"## A {entry['severity']} breach detected chunk {i+1} of file {file}\n")
                            output_file.write(f"### breach_timestamp:\n")
                            output_file.write(f"{entry['breach_timestamp']}\n")
                            output_file.write(f"### description:\n")
                            output_file.write(f"{entry['breach_description']}\n")
                    except Exception as ex:
                        ASCIIColors.error(ex)
                    self.step_end(f"Processing {file.name} chunk {i+1}/{n_chunks}")

                self.step_end(f"Processing {file.name}")

    
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
        ASCIIColors.info("Generating")
        self.callback = callback
        if self.personality_config.output_file_path=="":
            out = self.fast_gen(previous_discussion_text + "Please set the output file path in my settings page.")
            self.full(out)
        if self.personality_config.logs_path=="":
            out = self.fast_gen(previous_discussion_text + "Please set the logs folder path in my settings page.")
            self.full(out)

        self.process_logs(
                            self.personality_config.logs_path, 
                            self.personality_config.file_types
                        )


        return ""


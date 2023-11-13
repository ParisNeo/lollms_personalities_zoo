from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from safe_store.generic_data_loader import GenericDataLoader
from safe_store.document_decomposer import DocumentDecomposer
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
                {"name":"candidate_cv","type":"str","value":"", "help":"Path to the candidate CV"},
                {"name":"subject_text","type":"str","value":"", "help":"Path to the job position description"},
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
                                        "start":self.start
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        self.cv = None
        self.position = None

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


    def remove_backticks(self, text):
        if text.startswith("```") and text.endswith("```"):
            split_text = text.split()
            text = " ".join(split_text[1:])
            return text.replace("```", "")
        else:
            return text

    def summerize(self, chunks, summary_instruction="summerize"):
        summeries = []
        for i, chunk in enumerate(chunks):
            self.step_start(f"Processing chunk : {i}")
            summery = self.remove_backticks("```markdown\n"+ self.fast_gen(f"!@>instructuion: {summary_instruction}\nCV chunk {chunk}\n!@>summary:\n```markdown\n"))
            summeries.append(summery)
            self.step_end(f"Processing chunk : {i}")
        return "\n".join(summeries)

    def process_cv(self):
        output = ""
        self.step_start("Reading data")
        cv_data = GenericDataLoader.read_file(self.personality_config.candidate_cv)
        subject_text = GenericDataLoader.read_file(self.personality_config.subject_text)
        self.step_end("Reading data")

        self.step_start("Processing cv")
        cv_chunks = DocumentDecomposer.decompose_document(cv_data,self.personality.config.ctx_size//2,0, self.personality.model.tokenize, self.personality.model.detokenize)
        subject_chunks = DocumentDecomposer.decompose_document(subject_text,self.personality.config.ctx_size//2,0, self.personality.model.tokenize, self.personality.model.detokenize)
        output += f"Found `{len(cv_chunks)}` chunks in cv\n"
        output += f"Found `{len(subject_chunks)}` chunks in position description\n"
        self.full(output)
        
        cv_summary = self.summerize(cv_chunks,"Summerize this CV chunk in form of bullet points. Start by giving information about the candidate like his name and address, then his academic record, followed by his professional record if applicable. Finally, list pros and cons. Keep only relevant information about the candidate.")
        output += f"**CV summary**\n{cv_summary}\n"
        self.full(output)
        
        subject_summary = self.summerize(subject_chunks,"Summerize this position description. The objective is to identify the skills required for this position. Only extract the information from the provided chunk. Do not invent anything outside the provided text.")
        output += f"**Position description summary**\n{subject_summary}\n"
        self.full(output)

        answer = "```latex\n"+self.fast_gen("!@>instructions: Given the following position description and cv, build a latex file to help the interviewer planify and prepare the interview with the candidate. The text has the following sections:\n1- Candidate presentation\n2- Interview questions\n3- grades table.!@>interview ai:\nHere is the latex code:\n```latex\n")
        output += answer
        self.full(output)


        self.step_end("Processing cv")





    def start(self, prompt="", full_context=""):
        self.new_message("")
        self.process_cv()


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
        if self.personality_config.candidate_cv!="" and self.personality_config.subject_text!="":
            ASCIIColors.info("Generating")
            self.step_start("Understanding request")
            if self.yes_no("Is the user asking for starting the process?", previous_discussion_text):
                self.step_end("Understanding request")
                self.process_cv()
            else:
                self.step_end("Understanding request")
                self.fast_gen(previous_discussion_text, callback=self.callback)
        else:
            self.step_end("Understanding request")
            self.fast_gen(previous_discussion_text, callback=self.callback)
        return ""



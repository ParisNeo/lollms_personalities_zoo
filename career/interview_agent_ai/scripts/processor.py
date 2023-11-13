from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from safe_store.generic_data_loader import GenericDataLoader
from safe_store.document_decomposer import DocumentDecomposer
import subprocess
from pathlib import Path

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
                {"name":"language","type":"str","value":"", "help":"language to rewrite the document in"},

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
        if text.startswith("```"):
            split_text = text.split("\n")
            text = "\n".join(split_text[1:])
        if text.endswith("```"):
            text= text[:-3]
        return text

    def summerize(self, chunks, summary_instruction="summerize", chunk_name="chunk", answer_start=""):
        summeries = []
        for i, chunk in enumerate(chunks):
            self.step_start(f"Processing chunk : {i+1}")
            summery = self.remove_backticks(f"```markdown\n{answer_start}"+ self.fast_gen(f"!@>instructuion: {summary_instruction}\n{chunk_name}:\n{chunk}\n!@>summary:\n```markdown\n{answer_start}"))
            summeries.append(summery)
            self.step_end(f"Processing chunk : {i+1}")
        return "\n".join(summeries)

    def process_cv(self):
        output = ""
        self.step_start("Reading data")
        cv_data = GenericDataLoader.read_file(self.personality_config.candidate_cv)
        subject_text = GenericDataLoader.read_file(self.personality_config.subject_text)
        self.step_end("Reading data")

        self.step_start("chunking documents")
        cv_chunks = DocumentDecomposer.decompose_document(cv_data,self.personality.config.ctx_size//2,0, self.personality.model.tokenize, self.personality.model.detokenize, True)
        subject_chunks = DocumentDecomposer.decompose_document(subject_text,self.personality.config.ctx_size//2,0, self.personality.model.tokenize, self.personality.model.detokenize, True)
        output += f"Found `{len(cv_chunks)}` chunks in cv\n"
        output += f"Found `{len(subject_chunks)}` chunks in position description\n"
        self.full(output)
        self.step_end("chunking documents")
        
        self.step_start("summerizing cv")
        cv_summary = self.summerize(cv_chunks,"Summerize this CV chunk in form of bullet points separated by new line and do not add anny comments after the summary.\nUse a new line for each summary entry.\nStart by giving information about the candidate like his name and address and any other available information in the cv, then his academic record if applicable, followed by his professional record if applicable.\nKeep only relevant information about the candidate.\nDo not add information that is not in the cv.", "CV chunk", answer_start="- Type: Candidate\n- Name:")
        output += f"**CV summary**\n{cv_summary}\n\n"
        self.full(output)
        cv_path = Path(self.personality_config.candidate_cv)
        summary_path = cv_path.parent/(cv_path.stem+"_summary.md")
        with open(summary_path,"w") as f:
            f.write(cv_summary)
        self.step_end("summerizing cv")

        self.step_start("summerizing position subject")
        subject_summary = self.summerize(subject_chunks,"Summerize this position description and do not add any comments after the summary.\nThe objective is to identify the skills required for this position. Only extract the information from the provided chunk.\nDo not invent anything outside the provided text.","position description chunk")
        output += f"**Position description summary**\n{subject_summary}\n"
        self.full(output)

        subject_text_path = Path(self.personality_config.subject_text)
        subject_text_summary_path = subject_text_path.parent/(subject_text_path.stem+"_summary.md")
        with open(subject_text_summary_path,"w") as f:
            f.write(subject_summary)
        self.step_end("summerizing position subject")

        self.step_start("writing interview document")
        answer = "```latex\n"+"\\begin{document}\n"+self.fast_gen(f"!@>instructions: Given the following position description and candidate cv, build a latex file to help the interviewer planify and prepare the interview with the candidate. At the end, generate a table of grades for each specific point and do not fill it as it should be filled by the interviewer. The text has the following sections:\n1- Candidate presentation\n2- Interview questions\n3- grades table.\nDo not add any comments after the code.\nSubject:{subject_summary}\nCandidate cv:\n{cv_summary}\n!@>interview ai:\nHere is the latex code built from the subject summary and the cv:\n```latex\n"+"\\begin{document}\n")
        if not answer.endswith("```"):
            answer += "\n```\n"

        output += answer
        self.full(output)

        interview_latex_path = cv_path.parent/(cv_path.stem+"_interview.tex")
        with open(interview_latex_path,"w") as f:
            f.write(self.remove_backticks(answer))        
        self.step_end("writing interview document")
        

        if self.personality_config.language!="":
            self.step_start(f"Translating to:{self.personality_config.language}")
            answer = self.remove_backticks("```text\n" + self.fast_gen(f"!@>instructions: Translate the following text to {self.personality_config.language}\n!@>text to translate:\n{answer}!@>translator:\n```text\n"))
            if not answer.startswith("```"):
                answer = "```latex\n"+answer
            if not answer.endswith("```"):
                answer += "\n```\n"
            self.step_end(f"Translating to:{self.personality_config.language}")            
            output += f"## Translated to {self.personality_config.language}\n{answer}"
            interview_latex_path = cv_path.parent/(cv_path.stem+f"_interview_{self.personality_config.language}.tex")
            with open(interview_latex_path,"w") as f:
                f.write(self.remove_backticks(answer))        
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



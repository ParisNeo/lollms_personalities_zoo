from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from safe_store.text_vectorizer import TextVectorizer, VectorizationMethod
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
        self.data_base = TextVectorizer(VectorizationMethod.BERT, self.personality.model)

        
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
        operation = self.multichoice_question(
                                    "Classify the last prompt from the user.",
                                    [
                                        "Not explanation of code, not asking to start building code, not giving update information",
                                        "Explanation about the code",
                                        "Asking to start building code",
                                        "New updates to the previous code"
                                    ],
                                    previous_discussion_text)
        ASCIIColors.yellow(operation)
        if operation == 0: #Generic stuff
            ASCIIColors.info("Generating")
            self.step("Detected a generic ")
            self.callback = callback
            out = self.fast_gen(previous_discussion_text)
            self.full(out)
        elif operation == 1: # Giving information about the software to build
            ASCIIColors.info("Generating")
            self.step("Detected a software information")
            title = self.make_title(prompt)
            self.data_base.add_document(title,prompt, add_to_index=True)
            self.callback = callback
            out = self.fast_gen(previous_discussion_text)
            self.full(out)
        elif operation == 2: #build the software
            ASCIIColors.info("Generating")
            self.step("Detected a software build request")
            self.step_start("Saving the information to long term memory")
            title = self.make_title(prompt)
            self.data_base.add_document(title,prompt, add_to_index=True)
            self.step_end("Saving the information to long term memory")
            self.callback = callback
            self.step_start("Building the code")
            out = self.build_python_code(previous_discussion_text)
            if out!="":
                self.step_end("Building the code")
                self.full("```python\n"+out+"```")
                exec(out)
            else:
                self.step_end("Building the code", False)
            ASCIIColors.yellow(out)
            
        return ""


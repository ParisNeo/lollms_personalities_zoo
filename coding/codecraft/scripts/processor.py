from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
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
        # Example entries
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        # Supported types:
        # str, int, float, bool, list
        # options can be added using : "options":["option1","option2"...]        
        personality_config_template = ConfigTemplate(
            [
                {"name":"template_file","type":"str","value":"", "help":"A template file is a file that contains example of the structure of the code to generate. Just link to a file. The file should contain a structuire and placeholders put as comments. # Placeholder: Here you do this thing. You can place multiple placeholders in your code. Make sure the code is not very long as you may envounter context size problems."},
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
                                    "commands": { # list of commands (don't forget to add these to your config.yaml file)
                                        "help":self.help,
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        self.settings_updated()
    def settings_updated(self):
        """
        
        """
        if self.personality_config.template_file!="":
            with open(self.personality_config.template_file,"r") as f:
                self.template = f.read()
        else:
            self.template=None
        
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

    def run_workflow(self, prompt, previous_discussion_text="", callback=None, context_details=None):
        """
        This function generates code based on the given parameters.

        Args:
            full_prompt (str): The full prompt for code generation.
            prompt (str): The prompt for code generation.
            context_details (dict): A dictionary containing the following context details for code generation:
                - conditionning (str): The conditioning information.
                - documentation (str): The documentation information.
                - knowledge (str): The knowledge information.
                - user_description (str): The user description information.
                - discussion_messages (str): The discussion messages information.
                - positive_boost (str): The positive boost information.
                - negative_boost (str): The negative boost information.
                - force_language (str): The force language information.
                - ai_prefix (str): The AI prefix information.
            n_predict (int): The number of predictions to generate.
            client_id: The client ID for code generation.
            callback (function, optional): The callback function for code generation.

        Returns:
            None
        """
        if context_details is None:
            self.full("<b>The context details is none. This is probably due to the fact that you are using an old version of lollms. Please upgrade lollms to use this persona.</b>")
            return ""
        self.personality.info("Generating")
        self.callback = callback
        if self.template is not None:
            crafted_prompt=self.build_prompt([
                                    context_details["conditionning"],
                                    context_details["user_description"],
                                    "!@>template: \n"+self.template,
                                    context_details["documentation"],
                                    context_details["knowledge"],
                                    context_details["discussion_messages"],
                                    context_details["positive_boost"],
                                    context_details["negative_boost"],
                                    context_details["force_language"],
                                    context_details["ai_prefix"],
            ],5)
        else:
            crafted_prompt=self.build_prompt([
                                    context_details["conditionning"],
                                    context_details["user_description"],
                                    context_details["documentation"],
                                    context_details["knowledge"],
                                    context_details["discussion_messages"],
                                    context_details["positive_boost"],
                                    context_details["negative_boost"],
                                    context_details["force_language"],
                                    context_details["ai_prefix"],
            ],4)

        out = self.fast_gen(crafted_prompt)
        self.full(out)
        return out


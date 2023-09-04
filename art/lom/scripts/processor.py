import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PromptReshaper, git_pull, File_Path_Generator
import re
import importlib
import requests
from tqdm import tqdm
import webbrowser
try:
    import torchaudio
except:
    ASCIIColors.warning("No torch audio found")

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
        
        self.callback = None
        self.music_model = None
        self.previous_mg_prompt = None

        personality_config_template = ConfigTemplate(
            [
                {"name":"model_name","type":"str","value":"facebook/musicgen-small","options":["facebook/musicgen-small","facebook/musicgen-medium","facebook/musicgen-large"],"help":"Select the model to be used to generate the music. Bigger models provide higher quality but consumes more computing power"},
                {"name":"imagine","type":"bool","value":True,"help":"Imagine the images"},
                {"name":"generate","type":"bool","value":True,"help":"Paint the images"},
                {"name":"show_infos","type":"bool","value":True,"help":"Shows generation informations"},
                {"name":"continuous_discussion","type":"bool","value":True,"help":"If true then previous prompts and infos are taken into acount to generate the next image"},
                {"name":"add_style","type":"bool","value":True,"help":"If true then musicbot will choose and add a specific style to the prompt"},
                
                {"name":"activate_discussion_mode","type":"bool","value":True,"help":"If active, the AI will not generate an image until you ask it to, it will just talk to you until you ask it to make an musicwork"},
                
                {"name":"duration","type":"int","value":8, "min":1, "max":2048},

                {"name":"seed","type":"int","value":-1},
                {"name":"max_generation_prompt_size","type":"int","value":512, "min":10, "max":personality.config["ctx_size"]},
                
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
                                        "new_music":self.new_music,
                                        "regenerate":self.regenerate
                                    },
                                    "default": self.main_process
                                },                           
                            ],
                            callback=callback
                        )


    def install(self):
        super().install()
        
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        try:
            import torchaudio
            ASCIIColors.success("Torch audio OK")
        except:
            ASCIIColors.warning("No torch audio found")
        # Clone repository
        self.prepare()
        ASCIIColors.success("Installed successfully")


    def prepare(self):
        if self.music_model is None:
            from audiocraft.models import musicgen
            import torch
            self.step_start("Loading Meta's musicgen")
            self.music_model = musicgen.MusicGen.get_pretrained(self.personality_config.model_name, device='cuda')
            self.step_end("Loading Meta's musicgen")
        

    def help(self, prompt, full_context):
        self.full(self.personality.help)
    
    def new_music(self, prompt, full_context):
        self.files=[]
        self.full("Stmusicing fresh :)")
        

    def regenerate(self, prompt, full_context):
        if self.previous_mg_prompt:
            self.music_model.set_generation_params(duration=self.personality_config.duration)
            res = self.music_model.generate([prompt], progress=True)
            if self.personality_config.show_infos:
                self.new_message("infos", MSG_TYPE.MSG_TYPE_JSON_INFOS,{"prompt":prompt,"duration":self.personality_config.duration})
        else:
            self.full("Please generate an image first then retry")

    

    def get_styles(self, prompt, full_context):
        self.step_start("Selecting style")
        styles=[
            "hard rock",
            "Pop",
            "HipHop",
            "Riggay",
            "Techno",

        ]
        stl=", ".join(styles)
        prompt=f"{full_context}\n!@>user:{prompt}\nSelect what style(s) among those is more suitable for this musicwork: {stl}\n!@>assistant:I select"
        stl = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
        self.step_end("Selecting style")

        selected_style = ",".join([s for s in styles if s.lower() in stl])
        return selected_style


    def main_process(self, initial_prompt, full_context):    
        self.prepare()
        full_context = full_context[:full_context.index(initial_prompt)]
        
        if self.personality_config.imagine:
            if self.personality_config.activate_discussion_mode:
                pr  = PromptReshaper("""!@>discussion:
{{previous_discussion}}{{initial_prompt}}
!@>question: Is the user's message asking to generate a music sequence? 
!@>instruction>musicbot should answer with Yes or No.
!@>musicbot:The answer to the question is""")
                prompt = pr.build({
                        "previous_discussion":full_context,
                        "initial_prompt":initial_prompt
                        }, 
                        self.personality.model.tokenize, 
                        self.personality.model.detokenize, 
                        self.personality.model.config.ctx_size,
                        ["previous_discussion"]
                        )
                is_discussion = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
                ASCIIColors.cyan(is_discussion)
                if "yes" not in is_discussion.lower():
                    pr  = PromptReshaper("""!@>instructions>Lord of Music is a music geneation IA taht discusses with humans about music.
!@>discussion:
{{previous_discussion}}{{initial_prompt}}
!@>musicbot:""")
                    prompt = pr.build({
                            "previous_discussion":full_context,
                            "initial_prompt":initial_prompt
                            }, 
                            self.personality.model.tokenize, 
                            self.personality.model.detokenize, 
                            self.personality.model.config.ctx_size,
                            ["previous_discussion"]
                            )
                    ASCIIColors.yellow(prompt)

                    response = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
                    self.full(response)
                    return


      
            # ====================================================================================
            if self.personality_config.add_style:
                styles = self.get_styles(initial_prompt,full_context)
            else:
                styles = "No specific style selected."
            self.full(f"### Chosen style:\n{styles}")         

            self.step_start("Imagining prompt")
            # 1 first ask the model to formulate a query
            past = "!@>".join(full_context.split("!@>")[:-2])
            pr  = PromptReshaper("""!@>discussion:                                 
{{previous_discussion}}
!@>instructions:
Make a prompt based on the discussion with the user presented below to generate some music in the right style.
Make sure you mention every thing asked by the user's idea.
Do not make a very long text.
Do not use bullet points.
The prompt should be in english.
The generation ai has no access to the previous text so do not do references and justwrite the prompt 
{{initial_prompt}}
!@>style_choice: {{styles}}                                 
!@>music_generation_prompt: Create""")
            prompt = pr.build({
                    "previous_discussion":past if self.personality_config.continuous_discussion else '',
                    "initial_prompt":initial_prompt,
                    "styles":styles
                    }, 
                    self.personality.model.tokenize, 
                    self.personality.model.detokenize, 
                    self.personality.model.config.ctx_size,
                    ["previous_discussion"]
                    )
            
            ASCIIColors.yellow(prompt)
            generation_prompt = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
            self.step_end("Imagining prompt")
            self.full(f"### Chosen style:\n{styles}\n### Prompt:\n{generation_prompt}")         
            # ====================================================================================
            self.full(f"### Chosen style:\n{styles}\n### Prompt:\n{generation_prompt}")         
            # ====================================================================================            
            
        else:
            generation_prompt = initial_prompt
            
        self.previous_mg_prompt = generation_prompt

        output = f"### Prompt :\n{generation_prompt}"
        res = self.music_model.generate([generation_prompt])
        output_folder = self.personality.lollms_paths.personal_outputs_path / "lom"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = File_Path_Generator.generate_unique_file_path(output_folder, "generation","wav")
        torchaudio.save(output_file, res.reshape(1, -1).cpu(), 32000)

        url = "/outputs"+str(output_file).split("outputs")[1].replace("\\","/")
        output += f"""
<audio controls>
    <source src="{url}" type="audio/wav">
    Your browser does not support the audio element.
</audio>
"""
        self.full(output.strip())

        ASCIIColors.success("Generation succeeded")


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


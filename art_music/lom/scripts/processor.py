import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PromptReshaper, git_pull, File_Path_Generator
from lollms.client_session import Client

try:
    import torchaudio
except:
    ASCIIColors.warning("No torch audio found")

from typing import Callable
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
                {"name":"device","type":"str","value":"cuda:0","options":["cuda","cpu","xpu","ipu","hpu","xla","Vulkan"],"help":"Select the model to be used to generate the music. Bigger models provide higher quality but consumes more computing power"},
                {"name":"model_name","type":"str","value":"facebook/musicgen-melody","options":["facebook/musicgen-small","facebook/musicgen-medium","facebook/musicgen-melody","facebook/musicgen-large"],"help":"Select the model to be used to generate the music. Bigger models provide higher quality but consumes more computing power"},
                {"name":"number_of_samples","type":"int","value":1,"help":"The number of samples to generate"},
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
            self.music_model = musicgen.MusicGen.get_pretrained(self.personality_config.model_name, device=self.personality_config.device)
            self.step_end("Loading Meta's musicgen")
        

    def help(self, prompt, full_context):
        self.full(self.personality.help)
    
    def new_music(self, prompt, full_context):
        self.image_files=[]
        self.full("Starting fresh :)")
        

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
            "pop",
            "hiphop",
            "riggay",
            "techno",
            "classic"
        ]
        stl=", ".join(styles)
        prompt=f"{full_context}{self.config.separator_template}{self.config.start_header_id_template}user:{prompt}\nSelect what style(s) among those is more suitable for this musicwork: {stl}{self.config.separator_template}{self.config.start_header_id_template}assistant:The most suitable style among the proposed is"
        stl = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
        self.step_end("Selecting style")

        selected_style = ",".join([s for s in styles if s.lower() in stl])
        return selected_style


    def main_process(self, initial_prompt, full_context, callback, context_state, client):    
        self.prepare()
        full_context = full_context[:full_context.index(initial_prompt)]
        
        if self.personality_config.imagine:
            if self.personality_config.activate_discussion_mode:
                pr  = PromptReshaper("""{self.config.start_header_id_template}discussion:
{{previous_discussion}}{{initial_prompt}}
{self.config.start_header_id_template}question: Is the user's message asking to generate a music sequence? Make a very short answer.
{self.config.start_header_id_template}musicbot:Request answer:<answer>""")
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
                is_discussion = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
                ASCIIColors.cyan(is_discussion)
                if "yes" not in is_discussion.lower():
                    pr  = PromptReshaper("""{self.config.start_header_id_template}instructions>Lord of Music is a music geneation IA that discusses about making music with the user.
{self.config.start_header_id_template}discussion:
{{previous_discussion}}{{initial_prompt}}
{self.config.start_header_id_template}musicbot:""")
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
            past = f"{self.config.start_header_id_template}".join(full_context.split("{self.config.start_header_id_template}")[:-2])
            pr  = PromptReshaper("""{self.config.start_header_id_template}discussion:                                 
{{previous_discussion}}
{self.config.start_header_id_template}instructions:
Make a prompt based on the discussion with the user presented below to generate some music in the right style.
Make sure you mention every thing asked by the user's idea.
Do not make a very long text.
Do not use bullet points.
The prompt should be in english.
The generation ai has no access to the previous text so do not do references and just write the prompt.
{{initial_prompt}}
{self.config.start_header_id_template}style_choice: {{styles}}                                 
{self.config.start_header_id_template}music_generation_prompt: Create""")
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

        self.step_start("Making some music")
        output = f"### Prompt :\n{generation_prompt}"
        self.music_model.set_generation_params(duration=self.personality_config.duration)
        import torch
        torch.cuda.empty_cache()
        for sample in range(self.personality_config.number_of_samples):
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
        self.step_end("Making some music")

        ASCIIColors.success("Generation succeeded")


        
    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, context_details:dict=None, client:Client=None):
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
                - current_language (str): The force language information.
                - fun_mode (str): The fun mode conditionning text
                - ai_prefix (str): The AI prefix information.
            n_predict (int): The number of predictions to generate.
            client_id: The client ID for code generation.
            callback (function, optional): The callback function for code generation.

        Returns:
            None
        """
        self.callback = callback
        self.process_state(prompt, previous_discussion_text, callback)

        return ""


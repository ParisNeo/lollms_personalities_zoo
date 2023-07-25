import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import re
import importlib
import requests
from tqdm import tqdm
import webbrowser

class Processor(APScript):
    """
    A class that processes model inputs and outputs.

    Inherits from APScript.
    """
    def __init__(
                 self, 
                 personality: AIPersonality
                ) -> None:
        # Get the current directory
        root_dir = personality.lollms_paths.personal_path
        # We put this in the shared folder in order as this can be used by other personalities.
        shared_folder = root_dir/"shared"
        self.sd_folder = shared_folder / "auto_sd"
        
        self.callback = None
        self.sd = None
        personality_config_template = ConfigTemplate(
            [
                {"name":"imagine","type":"bool","value":True,"help":"Imagine the images"},
                {"name":"paint","type":"bool","value":True,"help":"Paint the images"},
                {"name":"img2img_denoising_strength","type":"float","value":7.5, "min":0.01, "max":1.0, "help":"The image to image denoising strength"},
                {"name":"restore_faces","type":"bool","value":True,"help":"Restore faces"},
                {"name":"caption_received_files","type":"bool","value":False,"help":"If active, the received file will be captioned"},
                {"name":"sampler_name","type":"str","value":"Euler a", "options":["Euler a","Euler","LMS","Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM++ 2M SDE", "DPM fast", "DPM adaptive", "DPM Karras", "DPM2 Karras", "DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DPM++ 2M SDE Karras" ,"DDIM", "PLMS","UniPC"], "help":"Select the sampler to be used for the diffusion operation. Supported samplers ddim, dpms, plms"},                
                {"name":"steps","type":"int","value":50, "min":10, "max":1024},
                {"name":"scale","type":"float","value":7.5, "min":0.1, "max":100.0},
                {"name":"width","type":"int","value":512, "min":10, "max":2048},
                {"name":"height","type":"int","value":512, "min":10, "max":2048},
                {"name":"skip_grid","type":"bool","value":True,"help":"Skip building a grid of generated images"},
                {"name":"batch_size","type":"int","value":1, "min":1, "max":100,"help":"Number of images per batch (requires more memory)"},
                {"name":"num_images","type":"int","value":1, "min":1, "max":100,"help":"Number of batch of images to generate (to speed up put a batch of n and a single num images, to save vram, put a batch of 1 and num_img of n)"},
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
                                        "new_image":self.new_image,
                                        "show_sd":self.show_sd,
                                    },
                                    "default": self.main_process
                                },                           
                            ]
                        )
        
    def install(self):
        super().install()
        # Step 1: Clone repository
        if not self.sd_folder.exists():
            subprocess.run(["git", "clone", "https://github.com/ParisNeo/stable-diffusion-webui.git", str(self.sd_folder)])
        self.prepare()
        ASCIIColors.success("Installed successfully")

    def prepare(self):
        if self.sd is None:
            self.step_start("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service", self.callback)
            self.sd = self.get_sd().SD(self.personality.lollms_paths, self.personality_config, max_retries=-1)
            self.step_end("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service", self.callback)
        
    def get_sd(self):
        sd_script_path = self.sd_folder / "lollms_sd.py"
        if sd_script_path.exists():
            module_name = sd_script_path.stem  # Remove the ".py" extension
            # use importlib to load the module from the file path
            loader = importlib.machinery.SourceFileLoader(module_name, str(sd_script_path))
            sd_module = loader.load_module()
            return sd_module

    def remove_image_links(self, markdown_text):
        # Regular expression pattern to match image links in Markdown
        image_link_pattern = r"!\[.*?\]\((.*?)\)"

        # Remove image links from the Markdown text
        text_without_image_links = re.sub(image_link_pattern, "", markdown_text)

        return text_without_image_links


    def help(self, prompt, full_context):
        self.full(self.personality.help, self.callback)
    
    def new_image(self, prompt, full_context):
        self.files=[]
        self.full("Starting fresh :)", self.callback)
        
        
    def show_sd(self, prompt, full_context):
        self.prepare()
        webbrowser.open("http://127.0.0.1:7860/?__theme=dark")        
        self.full("Showing Stable diffusion UI", self.callback)
        
    def add_file(self, path):
        self.prepare()
        super().add_file(path)
        if self.personality_config.caption_received_files:
            self.step_start("Understanding the image", self.callback)
            description = self.sd.interrogate(path)
            ASCIIColors.yellow(description)
            self.step_end("Understanding the image", self.callback)
            self.full(f"File added successfully\nImage description :{description}", self.callback)
        else:    
            self.full(f"File added successfully\n", self.callback)
        
    def main_process(self, prompt, full_context):    
        self.prepare()
        
        if self.personality_config.imagine:
            # ====================================================================================
            self.step_start("Imagining positive prompt", self.callback)
            # 1 first ask the model to formulate a query
            prompt = f"""{self.remove_image_links(full_context)}
    !@>Task:
    Generate a prompt based on the idea presented below.
    !@>idea: {prompt}
    !@>artbot:
    prompt:"""
            ASCIIColors.yellow(prompt)
            sd_positive_prompt = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
            self.step_end("Imagining positive prompt", self.callback)
            # ====================================================================================
            # ====================================================================================
            self.step_start("Imagining negative prompt", self.callback)
            # 1 first ask the model to formulate a query
            prompt = f"""{self.remove_image_links(full_context)}
    !@>Task:
    Generate negative prompt based on the idea.
    The negative prompt is a list of keywords that should not be present in our image.
    Try to force the generator not to generate text or extra fingers or deformed faces. 
    example: blurry, deformed, bad, ugly etc.
    !@>idea: {prompt}
    !@>artbot:
    prompt:{sd_positive_prompt}
    negative_prompt:"""
            ASCIIColors.yellow(prompt)
            sd_negative_prompt = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
            self.step_end("Imagining negative prompt", self.callback)
            # ====================================================================================            
            
        else:
            prompt = prompt.split("\n")
            if len(prompt)>1:
                sd_positive_prompt = prompt[0]
                sd_negative_prompt = prompt[1]
            else:
                sd_positive_prompt = prompt[0]
                sd_negative_prompt = ""
            
            
        output = f"# positive_prompt :\n{sd_positive_prompt}\n# negative_prompt :\n{sd_negative_prompt}"
        if self.personality_config.paint:
            files = []
            for i in range(self.personality_config.num_images):
                self.step_start(f"Building image number {i+1}/{self.personality_config.num_images}", self.callback)
                if len(self.files)>0:
                    out = self.sd.img_to_img(
                                self.files,
                                sd_positive_prompt,
                                negative_prompt=sd_negative_prompt, 
                                sampler_name="Euler",
                                seed=self.personality_config.seed,
                                cfg_scale=self.personality_config.scale,
                                steps=self.personality_config.steps,
                                width=self.personality_config.width,
                                height=self.personality_config.height,
                                denoising_strength=self.personality_config.img2img_denoising_strength,
                                tiling=False,
                                restore_faces=self.personality_config.restore_faces,
                                styles=None, 
                                save_folder=None, 
                                script_name="",
                                )
                    if out:
                        files += out["image_paths"]        
                else:
                    files += self.sd.txt_to_img(
                                sd_positive_prompt,
                                negative_prompt=sd_negative_prompt, 
                                sampler_name="Euler",
                                seed=self.personality_config.seed,
                                cfg_scale=self.personality_config.scale,
                                steps=self.personality_config.steps,
                                width=self.personality_config.width,
                                height=self.personality_config.height,
                                tiling=False,
                                restore_faces=self.personality_config.restore_faces,
                                styles=None, 
                                save_folder=None, 
                                script_name="",
                                upscaler_name="",
                                )["image_paths"]
                if len(files)>0:
                    f = str(files[-1]).replace("\\","/")
                    pth = f.split('/')
                    idx = pth.index("outputs")
                    pth = "/".join(pth[idx:])
                    file_path = f"![](/{pth})\n"
                    self.full(file_path, self.callback)
                
                self.step_end(f"Building image number {i+1}/{self.personality_config.num_images}", self.callback)
            
            for i in range(len(files)):
                files[i] = str(files[i]).replace("\\","/")
                pth = files[i].split('/')
                idx = pth.index("outputs")
                pth = "/".join(pth[idx:])
                file_path = f"![](/{pth})\n"
                output += file_path
                ASCIIColors.yellow(f"Generated file in here : {files[i]}")
            
        self.full(output.strip(), self.callback)
        

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


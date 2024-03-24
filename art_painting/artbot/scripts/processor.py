import subprocess
from fastapi import Request
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PromptReshaper, git_pull, file_path_to_url, PackageManager, find_next_available_filename
from lollms.services.sd.lollms_sd import LollmsSD
import re
import importlib
import requests
from tqdm import tqdm
import webbrowser
from typing import Dict, Any, Callable
from pathlib import Path
from PIL import Image
from io import BytesIO

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
        shared_folder = root_dir/"shared"
        self.sd_folder = shared_folder / "auto_sd"
        
        
        self.callback = None
        self.sd = None
        self.previous_sd_positive_prompt = None
        self.sd_negative_prompt = None

        self.sd_models_folder = self.sd_folder/"models"/"Stable-diffusion"
        if self.sd_models_folder.exists():
            self.sd_models = [f.stem for f in self.sd_models_folder.iterdir()]
        else:
            self.sd_models = ["Not installeed"]

        personality_config_template = ConfigTemplate(
            [
                {"name":"generation_engine","type":"str","value":"stable_diffusion", "options":["stable_diffusion", "dall-e-2", "dall-e-3"],"help":"Select the engine to be used to generate the images. Notice, dalle2 requires open ai key"},                
                {"name":"openai_key","type":"str","value":"","help":"A valid open AI key to generate images using open ai api (optional)"},
                {"name":"production_type","type":"str","value":"an artwork", "options":["a photo","an artwork", "a drawing", "a painting", "a hand drawing", "a design", "a presentation asset", "a presentation background", "a game asset", "a game background", "an icon"],"help":"This selects what kind of graphics the AI is supposed to produce"},
                {"name":"sd_model_name","type":"str","value":self.sd_models[0], "options":self.sd_models, "help":"Name of the model to be loaded for stable diffusion generation"},
                {"name":"sd_address","type":"str","value":"http://127.0.0.1:7860","help":"The address to stable diffusion service"},
                {"name":"share_sd","type":"bool","value":False,"help":"If true, the created sd server will be shared on yourt network"},
                {"name":"sampler_name","type":"str","value":"DPM++ 3M SDE", "options":["Euler a","Euler","LMS","Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM++ 2M SDE", "DPM fast", "DPM adaptive", "DPM Karras", "DPM2 Karras", "DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DPM++ 2M SDE Karras" ,"DDIM", "PLMS", "UniPC", "DPM++ 3M SDE", "DPM++ 3M SDE Karras", "DPM++ 3M SDE Exponential"], "help":"Select the sampler to be used for the diffusion operation. Supported samplers ddim, dpms, plms"},                
                {"name":"steps","type":"int","value":40, "min":10, "max":1024},
                {"name":"scale","type":"float","value":5, "min":0.1, "max":100.0},
                
                {"name":"install_sd","type":"btn","value":"Install Stable diffusion","help":"Installs stable diffusion"},

                {"name":"imagine","type":"bool","value":True,"help":"Imagine the images"},
                {"name":"build_title","type":"bool","value":True,"help":"Build a title for the artwork"},
                {"name":"paint","type":"bool","value":True,"help":"Paint the images"},
                {"name":"use_fixed_negative_prompts","type":"bool","value":True,"help":"Uses parisNeo's preferred negative prompts"},
                {"name":"fixed_negative_prompts","type":"str","value":"(((ugly))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), ((extra arms)), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), ((watermark)), ((robot eyes))","help":"which negative prompt to use in case use_fixed_negative_prompts is checked"},                
                {"name":"show_infos","type":"bool","value":True,"help":"Shows generation informations"},
                {"name":"continuous_discussion","type":"bool","value":True,"help":"If true then previous prompts and infos are taken into acount to generate the next image"},
                {"name":"automatic_resolution_selection","type":"bool","value":False,"help":"If true then artbot chooses the resolution of the image to generate"},
                {"name":"add_style","type":"bool","value":False,"help":"If true then artbot will choose and add a specific style to the prompt"},
                
                {"name":"activate_discussion_mode","type":"bool","value":True,"help":f"If active, the AI will not generate an image until you ask it to, it will just talk to you until you ask it to make the graphical output requested"},
                
                {"name":"continue_from_last_image","type":"bool","value":False,"help":"Uses last image as input for next generation"},
                {"name":"img2img_denoising_strength","type":"float","value":7.5, "min":0.01, "max":1.0, "help":"The image to image denoising strength"},
                {"name":"restore_faces","type":"bool","value":True,"help":"Restore faces"},
                {"name":"caption_received_files","type":"bool","value":False,"help":"If active, the received file will be captioned"},

                {"name":"width","type":"int","value":512, "min":10, "max":2048},
                {"name":"height","type":"int","value":512, "min":10, "max":2048},

                {"name":"thumbneil_ratio","type":"int","value":2, "min":1, "max":5},

                {"name":"automatic_image_size","type":"bool","value":False,"help":"If true, artbot will select the image resolution"},
                {"name":"skip_grid","type":"bool","value":True,"help":"Skip building a grid of generated images"},
                {"name":"batch_size","type":"int","value":1, "min":1, "max":100,"help":"Number of images per batch (requires more memory)"},
                {"name":"num_images","type":"int","value":1, "min":1, "max":100,"help":"Number of batch of images to generate (to speed up put a batch of n and a single num images, to save vram, put a batch of 1 and num_img of n)"},
                {"name":"seed","type":"int","value":-1},
                {"name":"max_generation_prompt_size","type":"int","value":512, "min":10, "max":personality.config["ctx_size"]},

                {"name":"quality","type":"str","value":"standard", "options":["standard","hd"],"help":"The quality of Dalle generated files."},                
   
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
                                        "regenerate":self.regenerate,
                                        "show_settings":self.show_settings,
                                    },
                                    "default": self.main_process
                                },                           
                            ],
                            callback=callback
                        )
        self.width=int(self.personality_config.width)
        self.height=int(self.personality_config.height)

    def get_css(self):
        return '<link rel="stylesheet" href="/personalities/art/artbot/assets/tailwind.css">'


    def make_selectable_photo(self, image_id, image_source, image_infos={}):
        with(open(Path(__file__).parent.parent/"assets/photo.html","r") as f):
            str_data = f.read()
        
        reshaper = PromptReshaper(str_data)
        str_data = reshaper.replace({
            "{image_id}":f"{image_id}",
            "{thumbneil_width}":f"{self.personality_config.width/self.personality_config.thumbneil_ratio}",
            "{thumbneil_height}":f"{self.personality_config.height/self.personality_config.thumbneil_ratio}",
            "{image_source}":image_source,
            "{__infos__}":str(image_infos).replace("True","true").replace("False","false").replace("None","null")
        })
        return str_data
    def make_selectable_photos(self, html:str):
        with(open(Path(__file__).parent.parent/"assets/photos_galery.html","r") as f):
            str_data = f.read()
        
        reshaper = PromptReshaper(str_data)
        str_data = reshaper.replace({
            "{{photos}}":html
        })
        return str_data
    def print_prompt(self, title, prompt):
        ASCIIColors.red("*-*-*-*-*-*-*-* ", end="")
        ASCIIColors.red(title, end="")
        ASCIIColors.red(" *-*-*-*-*-*-*-*")
        ASCIIColors.yellow(prompt)
        ASCIIColors.red(" *-*-*-*-*-*-*-*")

    def install(self):
        super().install()        
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      


    def prepare(self):
        if self.sd is None and self.personality_config.generation_engine=="stable_diffusion":
            self.step_start("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service")
            self.sd = LollmsSD(self.personality.app, "Artbot", max_retries=-1,auto_sd_base_url=self.personality_config.sd_address,share = self.personality_config.share_sd)
            self.step_end("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service")
        
        if self.personality_config.generation_engine=="stable_diffusion":
            model = self.sd.util_get_current_model().split(".")[0]
            if model!=self.personality_config.sd_model_name:
                self.step_start(f"Changing the model to {self.personality_config.sd_model_name}")
                self.sd.util_set_model(self.personality_config.sd_model_name,True)
                self.step_end(f"Changing the model to {self.personality_config.sd_model_name}")

    def remove_image_links(self, markdown_text):
        # Regular expression pattern to match image links in Markdown
        image_link_pattern = r"!\[.*?\]\((.*?)\)"

        # Remove image links from the Markdown text
        text_without_image_links = re.sub(image_link_pattern, "", markdown_text)

        return text_without_image_links


    def help(self, prompt="", full_context=""):
        self.personality.InfoMessage(self.personality.help)
    
    def new_image(self, prompt="", full_context=""):
        self.personality.image_files=[]
        self.personality.info("Starting fresh :)")
        
        
    def show_sd(self, prompt="", full_context=""):
        self.prepare()
        
        webbrowser.open(self.personality_config.sd_address+"/?__theme=dark")        
        self.personality.info("Showing Stable diffusion UI")
        
        
    def show_settings(self, prompt="", full_context=""):
        self.prepare()
        webbrowser.open(self.personality_config.sd_address+"/?__theme=dark")        
        self.full("Showing Stable diffusion settings UI")
        
    def show_last_image(self, prompt="", full_context=""):
        self.prepare()
        if len(self.personality.image_files)>0:
            self.full(f"![]({self.personality.image_files})")        
        else:
            self.full("Showing Stable diffusion settings UI")        
        
    def add_file(self, path, client, callback=None):
        self.new_message("")
        pth = str(path).replace("\\","/").split('/')
        idx = pth.index("uploads")
        pth = "/".join(pth[idx:])

        output = f"## Image:\n![]({pth})\n\n"
        self.full(output)
        if callback is None and self.callback is not None:
            callback = self.callback

        self.prepare()
        super().add_file(path, client)
        self.personality.image_files.append(path)
        if self.personality_config.caption_received_files:
            self.new_message("", MSG_TYPE.MSG_TYPE_CHUNK, callback=callback)
            self.step_start("Understanding the image", callback=callback)
            img = Image.open(str(path))
            # Convert the image to RGB mode
            img = img.convert("RGB")
            description = self.personality.model.interrogate_blip([img])[0]
            # description = self.sd.interrogate(str(path)).info
            self.print_prompt("Blip description",description)
            self.step_end("Understanding the image", callback=callback)           
            file_html = self.make_selectable_photo(path.stem,f"/{pth}",{"name":path.stem,"type":"Imported image", "prompt":description})
            output += f"##  Image description :\n{description}\n"
            self.full(output, callback=callback)
            self.ui(self.make_selectable_photos(file_html))
            self.finished_message()
        else:    
            self.full(f"File added successfully\n", callback=callback)
        
    def regenerate(self, prompt="", full_context=""):
        self.prepare()
        if self.previous_sd_positive_prompt:
            self.new_message("Regenerating using the previous prompt",MSG_TYPE.MSG_TYPE_STEP_START)
            output0 = f"### Positive prompt:\n{self.previous_sd_positive_prompt}\n\n### Negative prompt:\n{self.previous_sd_negative_prompt}"
            output = output0
            self.full(output)

            infos, output = self.paint(self.previous_sd_positive_prompt, self.previous_sd_negative_prompt, self.previous_sd_title, output)
         
            self.step_end("Regenerating using the previous prompt")
        else:
            self.full("Please generate an image first then retry")

    

    def get_styles(self, prompt, full_context):
        self.step_start("Selecting style")
        styles=[
            "Oil painting",
            "Octane rendering",
            "Cinematic",
            "Art deco",
            "Enameled",
            "Etching",
            "Arabesque",
            "Cross Hatching",
            "Callegraphy",
            "Vector art",
            "Vexel art",
            "Cartoonish",
            "Cubism",
            "Surrealism",
            "Pop art",
            "Pop surrealism",
            "Roschach Inkblot",
            "Flat icon",
            "Material Design Icon",
            "Skeuomorphic Icon",
            "Glyph Icon",
            "Outline Icon",
            "Gradient Icon",
            "Neumorphic Icon",
            "Vintage Icon",
            "Abstract Icon"

        ]
        stl=", ".join(styles)
        prompt=f"{full_context}\n!@>user:{prompt}\nSelect what style(s) among those is more suitable for this {self.personality_config.production_type.split()[-1]}: {stl}\n!@>assistant:I select"
        stl = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
        self.step_end("Selecting style")

        selected_style = ",".join([s for s in styles if s.lower() in stl])
        return selected_style

    def get_resolution(self, prompt, full_context, default_resolution=[512,512]):

        def extract_resolution(text, default_resolution=[512, 512]):
            # Define a regular expression pattern to match the (w, h) format
            pattern = r'\((\d+),\s*(\d+)\)'
            
            # Search for the pattern in the text
            match = re.search(pattern, text)
            
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
                return width, height
            else:
                return default_resolution
                    
        self.step_start("Choosing resolution")
        prompt=f"{full_context}\n!@>user:{prompt}\nSelect a suitable image size (width, height).\nThe default resolution uis ({default_resolution[0]},{default_resolution[1]})\n!@>selected_image_size:"
        sz = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","").split("\n")[0]

        self.step_end("Choosing resolution")

        return extract_resolution(sz, default_resolution)

    def paint(self, sd_positive_prompt, sd_negative_prompt, sd_title, metadata_infos):
        files = []
        ui=""
        metadata_infos0=metadata_infos
        for img in range(self.personality_config.num_images):
            self.step_start(f"Generating image {img+1}/{self.personality_config.num_images}")
            if self.personality_config.generation_engine=="stable_diffusion":
                file, infos = self.sd.paint(
                                sd_positive_prompt, 
                                sd_negative_prompt,
                                self.personality.image_files,
                                sampler_name = self.personality_config.sampler_name,
                                seed = self.personality_config.seed,
                                scale = self.personality_config.scale,
                                steps = self.personality_config.steps,
                                img2img_denoising_strength = self.personality_config.img2img_denoising_strength,
                                width = self.personality_config.width,
                                height = self.personality_config.height,
                                restore_faces = self.personality_config.restore_faces,
                            )
                infos["title"]=sd_title
                file = str(file)

                escaped_url =  file_path_to_url(file)

                file_html = self.make_selectable_photo(Path(file).stem, escaped_url, infos)
                files.append(escaped_url)
                ui += file_html
                metadata_infos += f'\n![]({escaped_url})'
                self.full(metadata_infos)
                
            elif self.personality_config.generation_engine=="dall-e-2" or  self.personality_config.generation_engine=="dall-e-3":
                if not PackageManager.check_package_installed("openai"):
                    PackageManager.install_package("openai")
                import openai
                openai.api_key = self.personality_config.config["openai_key"]
                if self.personality_config.generation_engine=="dall-e-2":
                    supported_resolutions = [
                        [512, 512],
                        [1024, 1024],
                    ]
                    # Find the closest resolution
                    closest_resolution = min(supported_resolutions, key=lambda res: abs(res[0] - self.personality_config.width) + abs(res[1] - self.personality_config.height))
                    
                else:
                    supported_resolutions = [
                        [1024, 1024],
                        [1024, 1792],
                        [1792, 1024]
                    ]
                    # Find the closest resolution
                    if self.personality_config.width>self.personality_config.height:
                        closest_resolution = [1792, 1024]
                    elif self.personality_config.width<self.personality_config.height: 
                        closest_resolution = [1024, 1792]
                    else:
                        closest_resolution = [1024, 1024]


                # Update the width and height
                self.personality_config.width = closest_resolution[0]
                self.personality_config.height = closest_resolution[1]                    

                if len(self.personality.image_files)>0 and self.personality_config.generation_engine=="dall-e-2":
                    # Read the image file from disk and resize it
                    image = Image.open(self.personality.image_files[0])
                    width, height = self.personality_config.width, self.personality_config.height
                    image = image.resize((width, height))

                    # Convert the image to a BytesIO object
                    byte_stream = BytesIO()
                    image.save(byte_stream, format='PNG')
                    byte_array = byte_stream.getvalue()
                    response = openai.images.create_variation(
                        image=byte_array,
                        n=1,
                        model=self.personality_config.generation_engine, # for now only dalle 2 supports variations
                        size=f"{self.personality_config.width}x{self.personality_config.height}"
                    )
                else:

                    response = openai.images.generate(
                        model=self.personality_config.generation_engine,
                        prompt=sd_positive_prompt.strip(),
                        quality="standard",
                        size=f"{self.personality_config.width}x{self.personality_config.height}",
                        n=1,
                        
                        )
                infos = {
                    "title":sd_title,
                    "prompt":self.previous_sd_positive_prompt,
                    "negative_prompt":""
                }
                # download image to outputs
                output_dir = self.personality.lollms_paths.personal_outputs_path/"dalle"
                output_dir.mkdir(parents=True, exist_ok=True)
                image_url = response.data[0].url

                # Get the image data from the URL
                response = requests.get(image_url)

                if response.status_code == 200:
                    # Generate the full path for the image file
                    file_name = output_dir/find_next_available_filename(output_dir, "img_dalle_")  # You can change the filename if needed

                    # Save the image to the specified folder
                    with open(file_name, "wb") as file:
                        file.write(response.content)
                    ASCIIColors.yellow(f"Image saved to {file_name}")
                else:
                    ASCIIColors.red("Failed to download the image")
                file = str(file_name)

                url = "/"+file[file.index("outputs"):].replace("\\","/")
                file_html = self.make_selectable_photo(Path(file).stem, url, infos)
                files.append("/"+file[file.index("outputs"):].replace("\\","/"))
                ui += file_html
                metadata_infos += f'\n![]({url})'
                self.full(metadata_infos)

            self.step_end(f"Generating image {img+1}/{self.personality_config.num_images}")

        if self.personality_config.continue_from_last_image:
            self.personality.image_files= [file]
        self.full(metadata_infos0)
        self.new_message(self.make_selectable_photos(ui),MSG_TYPE.MSG_TYPE_UI)        
        return infos

    def main_process(self, initial_prompt, full_context,context_details:dict=None):
        sd_title = "unnamed"    
        metadata_infos=""
        try:
            full_context = context_details["discussion_messages"]
        except:
            ASCIIColors.warning("Couldn't extract full context portion")    
        if self.personality_config.imagine:
            if self.personality_config.activate_discussion_mode:

                classification = self.multichoice_question("Classify the user prompt.", 
                                                           [
                                                               "The user is making an affirmation",
                                                               "The user is asking a question",
                                                               "The user is requesting to generate an artwork",
                                                               "The user is requesting to modify the artwork"
                                                            ], "!@>user: "+initial_prompt)

                if classification<=1:
                    prompt = self.build_prompt([
                                    "!@>instructions>Artbot is an art generation AI that discusses with humains about art.", #conditionning
                                    "!@>discussion:",
                                    full_context,
                                    initial_prompt,
                                    context_details["ai_prefix"],
                    ],2)
                    self.print_prompt("Discussion",prompt)

                    response = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
                    self.full(response)
                    return


            if self.personality_config.automatic_resolution_selection:
                res = self.get_resolution(initial_prompt, full_context, [self.personality_config.width,self.personality_config.height])
                self.width=res[0]
                self.height=res[1]
            else:
                self.width=self.personality_config.width
                self.height=self.personality_config.height
            metadata_infos += self.add_collapsible_entry("Chosen resolution",f"{self.width}x{self.height}") 
            self.full(f"{metadata_infos}")     
            # ====================================================================================
            if self.personality_config.add_style:
                styles = self.get_styles(initial_prompt,full_context)
                metadata_infos += self.add_collapsible_entry("Chosen style",f"{styles}") 
                self.full(f"{metadata_infos}")     
            else:
                styles = None
            stl = f"!@>style_choice: {styles}\n" if styles is not None else ""
            self.step_start("Imagining positive prompt")
            # 1 first ask the model to formulate a query
            past = self.remove_image_links(full_context)
            prompt = self.build_prompt([
                            f"@>instructions:Act as artbot, the art prompt generation AI. Use the previous discussion information to come up with an image generation prompt without referring to it. Be precise and describe the style as well as the {self.personality_config.production_type.split()[-1]} description details.", #conditionning
                            "!@>discussion:",
                            past if self.personality_config.continuous_discussion else '',
                            stl,
                            f"!@>art_generation_prompt: Create {self.personality_config.production_type} ",
            ],2)
            


            self.print_prompt("Positive prompt",prompt)

            sd_positive_prompt = f"{self.personality_config.production_type} "+self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
            self.step_end("Imagining positive prompt")
            metadata_infos += self.add_collapsible_entry("Positive prompt",f"{sd_positive_prompt}") 
            self.full(f"{metadata_infos}")     
            # ====================================================================================
            # ====================================================================================
            if not self.personality_config.use_fixed_negative_prompts:
                self.step_start("Imagining negative prompt")
                # 1 first ask the model to formulate a query
                prompt = self.build_prompt([
                                "@>instructions:Act as artbot, the art prompt generation AI. Use the previous discussion information to come up with a negative generation prompt.", #conditionning
                                "The negative prompt is a list of keywords that should not be present in our image.",
                                "Try to force the generator not to generate text or extra fingers or deformed faces.",
                                "Use as many words as you need depending on the context.",
                                "To give more importance to a term put it ibti multiple brackets ().",
                                "!@>discussion:",
                                past if self.personality_config.continuous_discussion else '',
                                stl,
                                f"!@>positive prompt: {sd_positive_prompt}",
                                f"!@>negative prompt: ((morbid)),",
                ],6)

                self.print_prompt("Generate negative prompt", prompt)
                sd_negative_prompt = "((morbid)),"+self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
                self.step_end("Imagining negative prompt")
            else:
                sd_negative_prompt = self.personality_config.fixed_negative_prompts
            metadata_infos += self.add_collapsible_entry("Negative prompt",f"{sd_negative_prompt}") 
            self.full(f"{metadata_infos}")     
            # ====================================================================================            
            if self.personality_config.build_title:
                self.step_start("Making up a title")
                # 1 first ask the model to formulate a query
                pr  = PromptReshaper("""!@>instructions:
Given this image description prompt and negative prompt, make a consize title
!@>positive_prompt:
{{positive_prompt}}
!@>negative_prompt:
{{negative_prompt}}
!@>title:
""")
                prompt = pr.build({
                        "positive_prompt":sd_positive_prompt,
                        "negative_prompt":sd_negative_prompt,
                        }, 
                        self.personality.model.tokenize, 
                        self.personality.model.detokenize, 
                        self.personality.model.config.ctx_size,
                        ["negative_prompt"]
                        )
                self.print_prompt("Make up a title", prompt)
                sd_title = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
                self.step_end("Making up a title")
                metadata_infos += self.add_collapsible_entry(f"{sd_title}","")
                self.full(f"{metadata_infos}")
        else:
            self.width=self.personality_config.width
            self.height=self.personality_config.height
            prompt = initial_prompt.split("\n")
            if len(prompt)>1:
                sd_positive_prompt = prompt[0]
                sd_negative_prompt = prompt[1]
            else:
                sd_positive_prompt = prompt[0]
                sd_negative_prompt = ""
            
        self.previous_sd_positive_prompt = sd_positive_prompt
        self.previous_sd_negative_prompt = sd_negative_prompt
        self.previous_sd_title = sd_title

        output = metadata_infos

        if self.personality_config.paint:
            self.prepare()
            infos = self.paint(sd_positive_prompt, sd_negative_prompt, sd_title, metadata_infos)
            self.full(output.strip())

        else:
            infos = None
        if self.personality_config.show_infos and infos:
            self.json("infos", infos)

    async def handle_request(self, request: Request) -> Dict[str, Any]:
        """
        Handle client requests.

        Args:
            data (dict): A dictionary containing the request data.

        Returns:
            dict: A dictionary containing the response, including at least a "status" key.

        This method should be implemented by a class that inherits from this one.

        Example usage:
        ```
        handler = YourHandlerClass()
        request_data = {"command": "some_command", "parameters": {...}}
        response = await handler.handle_request(request_data)
        ```
        """
        data = (await request.json())

        operation = data.get("name","variate")
        prompt = data.get("prompt","")
        negative_prompt =  data.get("negative_prompt","")
        if operation=="variate":
            imagePath = data.get("imagePath","")
            ASCIIColors.info(f"Regeneration requested for file : {imagePath}")
            self.new_image()
            ASCIIColors.info("Building new image")
            self.personality.image_files.append(self.personality.lollms_paths.personal_outputs_path/"sd"/imagePath.split("/")[-1])
            self.personality.info("Regenerating")
            self.previous_sd_positive_prompt = prompt
            self.previous_sd_negative_prompt = negative_prompt
            self.new_message(f"Generating {self.personality_config.num_images} variations")
            self.prepare()
            self.regenerate()
            
            return {"status":True, "message":"Image is now ready to be used as variation"}
        elif operation=="set_as_current":
            imagePath = data.get("imagePath","")
            ASCIIColors.info(f"Regeneration requested for file : {imagePath}")
            self.new_image()
            ASCIIColors.info("Building new image")
            self.personality.image_files.append(self.personality.lollms_paths.personal_outputs_path/"sd"/imagePath.split("/")[-1])
            ASCIIColors.info("Regenerating")
            return {"status":True, "message":"Image is now set as the current image for image to image operation"}

        return {"status":False, "message":"Unknown operation"}

    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, context_details:dict=None):
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
                - fun_mode (str): The fun mode conditionning text
                - ai_prefix (str): The AI prefix information.
            n_predict (int): The number of predictions to generate.
            client_id: The client ID for code generation.
            callback (function, optional): The callback function for code generation.

        Returns:
            None
        """
        self.callback = callback
        self.main_process(prompt, previous_discussion_text,context_details)

        return ""


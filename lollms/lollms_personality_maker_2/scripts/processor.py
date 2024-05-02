from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import os
import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.services.sd.lollms_sd import LollmsSD
from lollms.types import MSG_TYPE
from lollms.utilities import git_pull
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PromptReshaper, git_pull, file_path_to_url, find_next_available_filename
from safe_store import TextVectorizer, GenericDataLoader, VisualizationMethod, VectorizationMethod
from typing import Dict, Any

import re
import importlib
import requests
from tqdm import tqdm
import shutil
import yaml
import urllib.parse

from typing import Callable

from PIL import Image
from io import BytesIO


class AIBuildingRequestData(BaseModel):
    ai_name: str
    ai_author: str
    ai_version: str
    ai_category: str
    ai_language: str
    ai_description: str
    ai_conditionning: str
    ai_welcome_message: str
    ai_temperature: float
    ai_disclaimer: str

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
        self.word_callback = None
        self.sd = None
        self.sd_models_folder = self.sd_folder/"models"/"Stable-diffusion"
        if self.sd_models_folder.exists():
            self.sd_models = [f.stem for f in self.sd_models_folder.iterdir()]
        else:
            self.sd_models = ["Not installeed"]
        personality_config_template = ConfigTemplate(
            [
                {"name":"generation_engine","type":"str","value":"stable_diffusion", "options":["none (use default icon)","stable_diffusion", "dall-e-2", "dall-e-3"],"help":"Select the engine to be used to generate the images. Notice, dalle2 requires open ai key"},                
                {"name":"openai_key","type":"str","value":"","help":"A valid open AI key to generate images using open ai api (optional)"},
                {"name":"optimize_prompt","type":"bool","value":False, "help":"This is an extra layer to build a more comprehensive conditionning of the AI"},
                {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
                {"name":"data_folder_path","type":"str","value":"", "help":"A path to a folder containing data to feed the AI. Supported file types are: txt,pdf,docx,pptx"},
                {"name":"audio_sample_path","type":"str","value":"", "help":"A path to an audio file containing some voice sample to set as the AI's voice. Supported file types are: wav, mp3"},
                {"name":"default_negative_prompt","type":"str","value":"((((ugly)))), ((((text)))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))", "help":"A negative prompt to be used in icon generation. The worlds list is the lis tof words to avoid having in the image"},

                {"name":"generate_icon","type":"bool","value":True, "help":"generates an icon for the persona. if deactivated, the persona will have the same icon as lollms"},
                {"name":"sd_model_name","type":"str","value":self.sd_models[0], "options":self.sd_models, "help":"Name of the model to be loaded for stable diffusion generation"},
                {"name":"sd_address","type":"str","value":"http://127.0.0.1:7860","help":"The address to stable diffusion service"},
                {"name":"share_sd","type":"bool","value":False,"help":"If true, the created sd server will be shared on yourt network"},
                
                {"name":"sampler_name","type":"str","value":"DPM++ 3M SDE", "options":["Euler a","Euler","LMS","Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM++ 2M SDE", "DPM fast", "DPM adaptive", "DPM Karras", "DPM2 Karras", "DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DPM++ 2M SDE Karras" ,"DDIM", "PLMS", "UniPC", "DPM++ 3M SDE", "DPM++ 3M SDE Karras", "DPM++ 3M SDE Exponential"], "help":"Select the sampler to be used for the diffusion operation. Supported samplers ddim, dpms, plms"},                
                {"name":"ddim_steps","type":"int","value":50, "min":10, "max":1024},
                {"name":"scale","type":"float","value":7.5, "min":0.1, "max":100.0},
                {"name":"steps","type":"int","value":50, "min":10, "max":1024},                
                {"name":"W","type":"int","value":512, "min":10, "max":2048},
                {"name":"H","type":"int","value":512, "min":10, "max":2048},
                {"name":"skip_grid","type":"bool","value":True,"help":"Skip building a grid of generated images"},
                {"name":"img2img_denoising_strength","type":"float","value":7.5, "min":0.01, "max":1.0, "help":"The image to image denoising strength"},
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
                                        "regenerate_icons":self.regenerate_icons,
                                        "manual_building":self.manual_building
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        self.sd = None
        self.assets_path = None

    def install(self):
        super().install()
        
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")

    def help(self, prompt="", full_context=""):
        self.personality.InfoMessage(self.personality.help)

    def regenerate_icons(self, prompt="", full_context=""):
        try:
            index = full_context.index("name:")
            self.build_icon(full_context,full_context[index:].split("\n")[0].strip())
        except:
            self.warning("Couldn't find name")


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
        try:
            data = (await request.json())
            imageSource = data['imageSource']
            assets_path= data['assets_path']

            shutil.copy(self.personality.lollms_paths.personal_outputs_path/"sd"/imageSource.split("/")[-1] , Path(assets_path)/"logo.png")
            ASCIIColors.success("image Selected successfully")
            return {"status":True}
        except Exception as ex:
            trace_exception(ex)
            form_data = await request.form()
            ai_icon: Optional[UploadFile] = None
            if 'ai_icon' in form_data:
                ai_icon = form_data['ai_icon'].file

            # Parse the form data using Pydantic
            request_data = AIBuildingRequestData(**form_data)


            yaml_data="\n".join([
                f"## {request_data.ai_name} Chatbot conditionning file",
                f"## Author: {request_data.ai_author}",
                f"## Version: {request_data.ai_version}",
                f"## Description:",
                f"## {request_data.ai_description}",
                "## talking to.",
                "",
                "# Credits",
                f"author: {request_data.ai_author}",
                f"version: {request_data.ai_version}",
                f"category: {request_data.ai_category}",
                f"language: {request_data.ai_language}",
                f"name: {request_data.ai_name}",
                "personality_description: |",
                f"    {request_data.ai_description}",
                "disclaimer: |",
                f"    {request_data.ai_disclaimer}",
                "",
                "# Actual useful stuff",
                "personality_conditioning: |",
                "    !@>system: ",
                f"    {request_data.ai_conditionning}",
                "user_message_prefix: '!@>user:'",
                f"ai_message_prefix: '!@>{request_data.ai_name.lower().replace(' ','_')}:'",
                "# A text to put between user and chatbot messages",
                "link_text: '\n'",
                "welcome_message: |",
                f"    {request_data.ai_welcome_message}",
                "# Here are default model parameters",
                f"model_temperature: {request_data.ai_temperature} # higher: more creative, lower: more deterministic",
                "model_n_predicts: 8192 # higher: generates more words, lower: generates fewer words",
                "model_top_k: 50",
                "model_top_p: 0.90",
                "model_repeat_penalty: 1.0",
                "model_repeat_last_n: 40",
                "",
                "# Recommendations",
                "recommended_binding: ''",
                "recommended_model: ''",
                "",
                "# Here is the list of extensions this personality requires",
                "dependencies: []",
                "",
                "# A list of texts to be used to detect that the model is hallucinating and stop the generation if any one of these is output by the model",
                "anti_prompts: ['!@>']"
            ])   
            self.personality_path:Path = self.personality.lollms_paths.custom_personalities_path/request_data.ai_name.lower().replace(" ","_").replace("\n","").replace('"','')
            self.assets_path:Path = self.personality_path/"assets"
            self.personality_path.mkdir(parents=True, exist_ok=True)
            self.assets_path.mkdir(parents=True, exist_ok=True)
         
            with open(self.personality_path/"config.yaml","w", encoding="utf8") as f:
                f.write(yaml_data)

            # Process the file data
            if ai_icon:
                # Save the file to disk or process it as needed
                with open(self.assets_path/'logo.png', 'wb') as f:
                    while chunk := ai_icon.read(8192):
                        f.write(chunk)

            # Return a response indicating success or failure
            return {"status": True}        


    def prepare(self):
        if self.personality_config.generation_engine=="stable_diffusion":
            if self.sd is None:
                self.step_start("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service")
                self.sd = self.personality.app.sd if hasattr(self.personality.app, 'sd') else LollmsSD(self.personality.app, "Artbot", max_retries=-1,auto_sd_base_url=self.personality_config.sd_address,share = self.personality_config.share_sd)
                self.step_end("Loading ParisNeo's fork of AUTOMATIC1111's stable diffusion service")
            model = self.sd.util_get_current_model()
            if model!=self.personality_config.sd_model_name:
                self.step_start(f"Changing the model to {self.personality_config.sd_model_name}")
                self.sd.util_set_model(self.personality_config.sd_model_name,True)
                self.step_end(f"Changing the model to {self.personality_config.sd_model_name}")

    def get_sd(self):
        
        sd_script_path = self.sd_folder / "lollms_sd.py"
        git_pull(self.sd_folder)
        
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


    
    def make_selectable_photo(self, image_id, image_source, assets_path=None):
        pth = image_source.split('/')
        idx = pth.index("outputs")
        pth = "/".join(pth[idx:])

        with(open(Path(__file__).parent.parent/"assets/photo.html","r") as f):
            str_data = f.read()
        
        reshaper = PromptReshaper(str_data)
        str_data = reshaper.replace({
            "{image_id}":f"{image_id}",
            "{thumbneil_width}":f"256",
            "{thumbneil_height}":f"256",
            "{image_source}":pth,
            "{assets_path}":str(assets_path).replace("\\","/") if assets_path else str(self.assets_path).replace("\\","/")
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


    def build_icon(self, discussion_messages, name, output_text=""):
        self.prepare()
        # ----------------------------------------------------------------
        
        # Now we generate icon        
        # ----------------------------------------------------------------
        self.step_start("Imagining Icon")
        crafted_prompt = self.build_prompt(
            [

                "!@>system: icon imaginer is a personality icon description AI.",
                "The user describes a personality and the ai should describe a suitable icon for the ai personality",
                "icon imaginer tries to express the personality of by describing a suitable eye catching icon",
                "icon imaginer uses english to describe the icon.",
                "icon imaginer may emphesize some aspects of the icon by putting it inside multiple brackets, like (((beautiful))) or ((detailed)) etc...",
                "the more important the text is, the bigger the number of brackets.",
                "icon imaginer description starts by describing the icon in details, then adds the name of the style or a description of the style for more original vibes then add boosting words, like detailed, beautiful, hires etc...",
                "!@>context:",
                discussion_messages,
                f"!@>name: {name}",
                f"Answer with only the prompt with no extra comments. All the prompt should be written in a single line.",
                f"!@>icon imaginer : An (((icon))) of a "
            ],5
        )
        sd_prompt = "An (((icon))) of a "+self.generate(crafted_prompt,256,0.1,10,0.98, debug=True, callback=self.sink).strip().split("\n")[0]
        self.step_end("Imagining Icon")
        ASCIIColors.yellow(f"sd prompt:{sd_prompt}")
        output_text+=self.build_a_document_block('icon sd_prompt',"",sd_prompt)
        self.full(output_text)
        self.chunk("")
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------

        sd_negative_prompt = self.personality_config.default_negative_prompt
        output_text+= self.build_a_document_block('icon sd_negative_prompt',"",sd_negative_prompt)
        self.full(output_text)
        self.chunk("")

        self.new_message("")
        self.step_start("Painting Icon")
        try:
            files = []
            ui=""
            for img in range(self.personality_config.num_images):
                self.step_start(f"Generating image {img+1}/{self.personality_config.num_images}")
                if self.personality_config.generation_engine=="stable_diffusion":
                    file, infos = self.sd.paint(
                                    sd_prompt, 
                                    sd_negative_prompt,
                                    [],
                                    sampler_name = self.personality_config.sampler_name,
                                    seed = self.personality_config.seed,
                                    scale = self.personality_config.scale,
                                    steps = self.personality_config.steps,
                                    img2img_denoising_strength = self.personality_config.img2img_denoising_strength,
                                    width = 512,
                                    height = 512,
                                    restore_faces = True,
                                )
                    if file is None:
                        self.step_end(f"Generating image {img+1}/{self.personality_config.num_images}", False)
                        continue
                    self.step_end(f"Generating image {img+1}/{self.personality_config.num_images}")
                    file = str(file)

                    files.append(file)
                    escaped_url =  file_path_to_url(file)
                    file_html = self.make_selectable_photo(Path(file).stem, escaped_url, self.assets_path)
                    ui += file_html
                    self.full(f'\n![]({escaped_url})')
                elif self.personality_config.generation_engine=="dall-e-2" or  self.personality_config.generation_engine=="dall-e-3":
                    import openai
                    openai.api_key = self.personality_config.config["openai_key"]
                    if self.personality_config.generation_engine=="dall-e-2":
                        closest_resolution = [512, 512]
                    else:
                        closest_resolution = [1024, 1024]


                    # Update the width and height
                    self.personality_config.width = closest_resolution[0]
                    self.personality_config.height = closest_resolution[1]                    

                    # Read the image file from disk and resize it
                    if len(self.personality.image_files)>0 and self.personality_config.generation_engine=="dall-e-2":
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
                            prompt=sd_prompt.strip(),
                            quality="standard",
                            size=f"{self.personality_config.width}x{self.personality_config.height}",
                            n=1,
                            
                            )
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
                        files.append(file)
                        escaped_url =  file_path_to_url(file)
                        file_html = self.make_selectable_photo(Path(file).stem, escaped_url, self.assets_path)
                        ui += file_html
                        self.full(f'\n![]({escaped_url})')
                        self.chunk("")

        except Exception as ex:
            try:
                self.exception("Couldn't generate the personality icon.\nPlease make sure that the personality is well installed and that you have enough memory to run both the model and stable diffusion")
                ASCIIColors.error("Couldn't generate the personality icon.\nPlease make sure that the personality is well installed and that you have enough memory to run both the model and stable diffusion")
                shutil.copy("assets/logo.png",self.assets_path)
                files.append(self.assets_path/"logo.png")
                trace_exception(ex)
            except Exception as ex:
                trace_exception(ex)

            files=[]
        self.step_end("Painting Icon")

        ui = ""
        for i in range(len(files)):
            files[i] = str(files[i]).replace("\\","/")
            file_id = files[i].split(".")[0].split('_')[1]
            shutil.copy(files[i],str(self.assets_path))
            file_path = self.make_selectable_photo(f"Artbot_{file_id}", files[i])
            ui += str(file_path)
            print(f"Generated file in here : {str(files[i])}")

        if self.personality_config.make_scripted:
            ui += f"""
            <a href="#" onclick="openCodeFolder()"> Click here to open the script folder of the persona</a>
            <a href="#" onclick="openCodeFolderInVsCode()"> Click here to open the script folder of the persona in vscode</a>

            <script>
            function openCodeFolder() {{
                const secretMessage = {{
                '"folder_path": {self.scripts_path}'
                }};

                fetch('/open_folder', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify(secretMessage)
                }})
                .then(() => {{
                console.log("🎉 The secret message has been sent and the magic code folder has been opened! 🎉");
                }})
                .catch((error) => {{
                console.error("😱 Oh no! Something went wrong:", error);
                }});
            }}
            function openCodeFolderInVsCode() {{
                const secretMessage = {{
                '"folder_path": {self.scripts_path}'
                }};

                fetch('/open_discussion_folder_in_vs_code', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify(secretMessage)
                }})
                .then(() => {{
                console.log("🎉 The secret message has been sent and the magic code folder has been opened! 🎉");
                }})
                .catch((error) => {{
                console.error("😱 Oh no! Something went wrong:", error);
                }});
            }}
            </script>
            """

        server_path = "/outputs/"+str(self.personality_path)
        # ----------------------------------------------------------------
        self.step_end("Painting Icon")
        
        output_text+= self.build_a_folder_link(str(self.personality_path).replace("\\","/"),"press this text to access personality path")
        self.full(output_text)
        self.new_message('<h2>Please select a photo to be used as the logo</h2>\n'+self.make_selectable_photos(ui),MSG_TYPE.MSG_TYPE_UI)

        
        self.assets_path.mkdir(parents=True, exist_ok=True)
        if len(files)>0:
            shutil.copy(files[-1], self.assets_path/"logo.png")
        else:
            shutil.copy(Path(__file__).parent.parent/"assets"/"lollms_logo.png", self.assets_path/"logo.png")


    def manual_building(self, prompt="", full_context=""):
        form_path = Path(__file__).parent.parent/"assets"/"edit_persona.html"
        with open(form_path,"r") as f:
            form = f.read()
        self.new_message(form,MSG_TYPE.MSG_TYPE_UI)
        pass

    from lollms.client_session import Client
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
                - force_language (str): The force language information.
                - fun_mode (str): The fun mode conditionning text
                - ai_prefix (str): The AI prefix information.
            n_predict (int): The number of predictions to generate.
            client_id: The client ID for code generation.
            callback (function, optional): The callback function for code generation.

        Returns:
            None
        """


        self.word_callback = callback
        output_text = ""
        self.callback = callback

        # First we create the yaml file
        # ----------------------------------------------------------------
        self.step_start("Coming up with the personality name")
        self.chunk("")
        crafted_prompt = self.build_prompt(
            [

                "!@>system: names maker is a personality name making AI.",
                "The user describes a personality and the ai should give it an apropriate name",
                "If the user explicitely proposed a name, qna responds with that name",
                "qna uses the same language as the one spoken by the user to name the personality.",
                "qna only answers with the personality name without any explanation.",
                "!@>context",
                context_details["discussion_messages"],
                f"!@>qna: The chosen personality name is "
            ],6
        )
        name = self.generate(crafted_prompt,50,0.1,10,0.98, debug=True, callback=self.sink).strip().split("\n")[0]
        self.step_end("Coming up with the personality name")
        name = re.sub(r'[\\/:*?"<>|.]', '', name)
        ASCIIColors.yellow(f"Name:{name}")
        Infos_text = ""
        Infos_text+=f"<b>Name</b>: {name}<br>"
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        try:
            author = "lpm prompted by "+self.personality.config.user_name
        except:
            author = "lpm"
        # ----------------------------------------------------------------
        Infos_text+=f"<b>Author</b>: {author}<br>"
        
        # ----------------------------------------------------------------
        version = "1.0" 
        Infos_text+=f"<b>Version</b>: {version}<br>"
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        self.step_start("Coming up with the category")
        crafted_prompt = self.build_prompt(
            [
                "!@>system: category maker is a personality category guessing AI.",
                "The user describes a personality and the ai should guess what category the AI fits in best",
                "If the user explicitely proposed a category, category maker responds with that category",
                "category maker only answers with the personality category name without any explanation.",
                f"the category should be one of these: {[c.stem for c in self.personality.lollms_paths.personalities_zoo_path.iterdir() if c.is_dir()]}",
                "!@>context",
                context_details["discussion_messages"],
                f"!@>category maker: The chosen personality category is "
            ],6
        )        
        category = self.generate(crafted_prompt,256,0.1,10,0.98, debug=True, callback=self.sink).strip().replace("'","").replace('"','').replace(".","").split("\n")[0]
        self.step_end("Coming up with the category")
        category = re.sub(r'[\\/:*?"<>|.]', '', category)
        ASCIIColors.yellow(f"Category:{category}")
        Infos_text+=f"<b>Category</b>: {category}<br>"
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        self.step_start("Coming up with the language")
        crafted_prompt = self.build_prompt(
            [
                "!@>system: language finder is a personality language guessing AI.",
                "The user describes a personality in a specific language and the ai should guess what language should be used for the personality.",
                "!@>context:",
                context_details["discussion_messages"],
                "!@>instructions to follow:",
                "Default language is english, but if the user is using another language to describe the ai then language finder uses that language."
                "Do not take into  condideration the user name in choosing the language. Just look at his prompt.",
                "If the user explicitely states the language that should be used, language finder uses that language",
                "language finder does not provide the language iso name, just the plain english name of the language such as: french, english, spanish, chinese, arabic etc ...",
                "language finder only answers with the personality language name without any explanation.",
                f"!@>language: "
            ],3
        )
        language = self.generate(crafted_prompt,10,0.1,10,0.98, debug=True, callback=self.sink).strip().replace("'","").replace('"','').replace(".","").split("\n")[0]
        self.step_end("Coming up with the language")
        language = re.sub(r'[\\/:*?"<>|.]', '', language)
        ASCIIColors.yellow(f"Language:{language}")
        Infos_text+=f"<b>Language</b>: {language}<br>"
        # ----------------------------------------------------------------
        
        output_text+=self.build_a_document_block('Infos',"",Infos_text)
        self.full(output_text)
        self.chunk("")
        # ----------------------------------------------------------------
        self.step_start("Coming up with the description")
        crafted_prompt = self.build_prompt(
            [
                "!@>system: description builder is a personality description AI.",
                "The user describes a personality and the ai should build a better description of the AI",
                "description builder pays attention to the user description and infer any more details that need to be added while keeping a relatively short description text."
                "description builder makes sure that no information provided by the user is overseen.",
                "description builder only answers with the personality description without any explanation.",
                "!@>context",
                context_details["discussion_messages"],
                f"!@>personality name:{name}",
                f"!@>description in {language}: "
            ],6
        )
        description = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink).strip().replace("'","").replace('"','').replace(".","").split("\n")[0]
        self.step_end("Coming up with the description")
        ASCIIColors.yellow(f"Description: {description}")
        output_text+= self.build_a_document_block('description',"",description)
        self.full(output_text)
        self.chunk("")
        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        self.step_start("Coming up with the disclaimer")
        crafted_prompt = self.build_prompt(
            [
                "!@>system: disclaimer builder is a personality disclaimer AI.",
                "The user describes a personality and the ai should build a disclaimer message to show the users of the personality.",
                "disclaimer builder pays attention to the user description and infer any more details that need to be in the desclaimer while keeping a relatively short disclaimer text."
                "disclaimer builder makes sure that harms that can be caused by the ai personality is clearely stated.",
                "if the personality is harmless or can't be used to cause harm, then disclaimer builder just builds a soft very short assuring disclaimer.",
                "disclaimer builder only answers with the personality disclaimer without any explanation.",
                "!@>context",
                context_details["discussion_messages"],
                f"!@>personality name:{name}",
                f"!@>disclaimer in {language}: "
            ],7
        )
        disclaimer = self.generate(crafted_prompt,256,0.1,10,0.98, debug=True, callback=self.sink).strip().replace("'","").replace('"','').replace(".","").split("\n")[0]
        self.step_end("Coming up with the disclaimer")
        ASCIIColors.yellow(f"Disclaimer: {disclaimer}")
        output_text+=self.build_a_document_block('disclaimer',"",disclaimer)
        self.full(output_text)
        self.chunk("")
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        self.step_start("Coming up with the conditionning")
        crafted_prompt = self.build_prompt(
            [
                "!@>system: conditionning builder is a personality conditionning AI.",
                "The user describes a personality and the ai should build a consistant AI conditionning system text.",
                "conditionning builder pays attention to the user description and infer any more details that need to be in the conditionning while keeping a relatively short conditionning text."
                "conditionning builder only answers with the personality conditionning without any explanation.",
                "!@>context",
                context_details["discussion_messages"],
                f"!@>personality language:{language}",
                f"!@>conditionning builder: {name} is "
                f"Be concise and try to answer with a single paragraph as much as possible unless you need to provide examples.",
                f"!@>conditionning:",
            ],5
        )
        conditioning = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink).strip().replace("'","").replace('"','').replace(".","")
        conditioning = f"{name} is "+conditioning
        self.step_end("Coming up with the conditionning")
        ASCIIColors.yellow(f"Conditioning: {conditioning}")
        output_text+=self.build_a_document_block('conditioning',"",conditioning)
        self.full(output_text)
        self.chunk("")
        # ----------------------------------------------------------------
        if self.personality_config.optimize_prompt:
            self.step_start("Optimizing the prompt")
            crafted_prompt = self.build_prompt(
                [
                    "!@>system: Optimus Persona is a personality improver AI It is designed to analyze, research, and enhance existing personality prompts The AI begins by thoroughly examining the intended tasks and potential areas for improvement It then explores related but overlooked capabilities that could complement the intended task and enhance the overall functionality of the personality The AI breaks down the personality prompts into their core components, evaluates the compatibility of each proposed improvement, and synthesizes the strongest improvements The AI reviews the enhanced prompts for clarity, coherence, and logical flow, and documents the improvements made to the personality prompts The AI is designed to maintain accuracy in the intended task while adding valuable related capabilities.",
                    f"!@>user: Write a comprehensive personality conditionning text for {name} from this rough idea:",
                    f"{conditioning}",
                    f"Be concise and try to answer with a single paragraph as much as possible unless you need to provide examples.",
                    f"!@>optimus:",
                ]
            )
            conditioning = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink).strip().replace("'","").replace('"','').replace(".","")
            conditioning = f"{name} is "+conditioning
            self.step_end("Coming up with the conditionning")
            ASCIIColors.yellow(f"Conditioning: {conditioning}")
            output_text+=self.build_a_document_block('refined conditioning',"",conditioning)
            self.full(output_text)
            self.chunk("")

                 



        # ----------------------------------------------------------------
        
        # ----------------------------------------------------------------
        self.step_start("Coming up with the welcome message")
        crafted_prompt = self.build_prompt(
            [
                "!@>system: welcome message builder is a personality welcome message building AI.",
                "The user describes a personality and the ai should build a short AI welcome message.",
                "welcome message builder pays attention to the user description and infer any more details that need to be in the conditionning while keeping a relatively short welcome message."
                "welcome message builder only answers with the personality conditionning without any explanation.",
                "!@>context",
                context_details["discussion_messages"],
                f"!@>personality name: {name}",
                f"!@>personality welcome message in {language}: "
            ],5
        )
        welcome_message = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink).strip().replace("'","").replace('"','').replace(".","").split("\n")[0]
        self.step_end("Coming up with the welcome message")
        ASCIIColors.yellow(f"Welcome message: {welcome_message}")
        output_text+=self.build_a_document_block('Welcome message',"",welcome_message)
        self.full(output_text)
        self.chunk("")
        # ----------------------------------------------------------------
                         
        # ----------------------------------------------------------------
        self.step_start("Building the yaml file")
        cmt_desc = "\n## ".join(description.split("\n"))
        desc = "\n    ".join(description.split("\n"))
        disclaimer = "\n    ".join(disclaimer.split("\n"))
        conditioning =  "\n    ".join(conditioning.split("\n"))
        welcome_message =  "\n    ".join(welcome_message.split("\n"))
        yaml_data="\n".join([
            f"## {name} Chatbot conditionning file",
            f"## Author: {author}",
            f"## Version: {version}",
            f"## Description:",
            f"## {cmt_desc}",
            "## talking to.",
            "",
            "# Credits",
            f"author: {author}",
            f"version: {version}",
            f"category: {category}",
            f"language: {language}",
            f"name: {name}",
            "personality_description: |",
            f"    {desc}",
            "disclaimer: |",
            f"    {disclaimer}",
            "",
            "# Actual useful stuff",
            "personality_conditioning: |",
            "    !@>system: ",
            f"    {conditioning}",
            "user_message_prefix: '!@>user:'",
            f"ai_message_prefix: '!@>{name.lower().replace(' ','_')}:'",
            "# A text to put between user and chatbot messages",
            "link_text: '\n'",
            "welcome_message: |",
            f"    {welcome_message}",
            "# Here are default model parameters",
            "model_temperature: 0.6 # higher: more creative, lower: more deterministic",
            "model_n_predicts: 8192 # higher: generates more words, lower: generates fewer words",
            "model_top_k: 50",
            "model_top_p: 0.90",
            "model_repeat_penalty: 1.0",
            "model_repeat_last_n: 40",
            "",
            "# Recommendations",
            "recommended_binding: ''",
            "recommended_model: ''",
            "",
            "# Here is the list of extensions this personality requires",
            "dependencies: []",
            "",
            "# A list of texts to be used to detect that the model is hallucinating and stop the generation if any one of these is output by the model",
            "anti_prompts: ['!@>']"
        ])

        self.step_end("Building the yaml file")
        self.step_start("Preparing paths")
        self.personality_path:Path = self.personality.lollms_paths.custom_personalities_path/name.lower().replace(" ","_").replace("\n","").replace('"','')
        self.personality_path.mkdir(parents=True, exist_ok=True)
        self.assets_path = self.personality_path/"assets"
        self.assets_path.mkdir(parents=True, exist_ok=True)
        self.scripts_path = self.personality_path/"scripts"
        self.audio_path = self.personality_path/"audio"
        self.data_path = self.personality_path/"data"
        self.step_end("Preparing paths")

        self.step_start("Saving configuration file")
        with open(self.personality_path/"config.yaml","w", encoding="utf8") as f:
            f.write(yaml_data)
        self.step_end("Saving configuration file")


        if self.personality_config.generate_icon:
            self.step_start("Building icon")
            try:
                self.build_icon(previous_discussion_text, name, output_text)
            except Exception as ex:
                trace_exception(ex)
                ASCIIColors.red("failed to generate icons.\nUsing default icon")
            self.step_end("Building icon")
        else:
            shutil.copy("assets/logo.png",self.assets_path)
            
        if self.personality_config.make_scripted:
            self.scripts_path.mkdir(parents=True, exist_ok=True)
            self.step_start("Creating default script")
            template_fn = Path(__file__).parent/"script_template.py"
            shutil.copy(template_fn, self.scripts_path/"processor.py")
            self.step_end("Creating default script")

        if self.personality_config.data_folder_path!="":
            self.data_path.mkdir()
            self.step_start("Creating vector database")
            text = []
            data_path = Path(self.personality_config.data_folder_path)
            text_files = []
            extensions = ["*.txt","*.pdf","*.pptx","*.docx","*.md"]
            for extension in extensions:
                text_files += [file if file.exists() else "" for file in data_path.glob(extension)]
            for file in text_files:
                self.step_start(f"Adding file: {file}")
                try:
                    text.append(GenericDataLoader.read_file(file))
                    self.step_end(f"Adding file: {file}")
                except Exception as ex:
                    trace_exception(ex)
                    self.step_end(f"Adding file: {file}",False)

            # Replace 'example_dir' with your desired directory containing .txt files
            self._data = "\n".join(map((lambda x: f"\n{x}"), text))
            self.step_start("Building data ...")
            try:
                self.persona_data_vectorizer = TextVectorizer(
                            self.personality.config.data_vectorization_method, # supported "model_embedding" or "tfidf_vectorizer"
                            model=self.personality.model, #needed in case of using model_embedding
                            save_db=True,
                            database_path=self.data_path/"db.json",
                            data_visualization_method=VisualizationMethod.PCA,
                            database_dict=None)
                self.persona_data_vectorizer.add_document("persona_data", self._data, 512, 0)
                self.persona_data_vectorizer.index()
                self.persona_data_vectorizer.save_to_json()
                self.step_end("Building data ...")
            except Exception as ex:
                trace_exception(ex)
                self.step_end("Building data ...",False)

            self.step_end("Creating vector database")

        if self.personality_config.audio_sample_path!="":
            audio_sample_path=Path(self.personality_config.audio_sample_path)
            self.audio_path.mkdir(exist_ok=True, parents=True)
            self.step_start("Creating a voice for the AI")
            shutil.copy(audio_sample_path, self.audio_path/audio_sample_path.name)
            self.step_end("Creating a voice for the AI")
        

        return output_text



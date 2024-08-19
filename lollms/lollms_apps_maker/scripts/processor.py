"""
Project: LoLLMs
Personality: # Placeholder: Personality name (e.g., "Science Enthusiast")
Author: # Placeholder: Creator name (e.g., "ParisNeo")
Description: # Placeholder: Personality description (e.g., "A personality designed for enthusiasts of science and technology, promoting engaging and informative interactions.")
"""

from lollms.types import MSG_OPERATION_TYPE
from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from lollms.client_session import Client
from lollms.functions.generate_image import build_image_from_simple_prompt
from lollms.functions.select_image_file import select_image_file_function
from lollms.functions.take_a_photo import take_a_photo_function

from lollms.utilities import discussion_path_to_url, app_path_to_url
import subprocess
from typing import Callable, Any
from functools import partial
from ascii_colors import trace_exception
import yaml
from datetime import datetime
from pathlib import Path
import shutil
import os
import shutil
import yaml
import git
import json

class Processor(APScript):
    """
    Defines the behavior of a personality in a programmatic manner, inheriting from APScript.
    
    Attributes:
        callback (Callable): Optional function to call after processing.
    """
    
    def __init__(
                 self, 
                 personality: AIPersonality,
                 callback: Callable = None,
                ) -> None:
        """
        Initializes the Processor class with a personality and an optional callback.

        Parameters:
            personality (AIPersonality): The personality instance.
            callback (Callable, optional): A function to call after processing. Defaults to None.
        """
        
        self.callback = callback
        
        # Configuration entry examples and types description:
        # Supported types: int, float, str, string (same as str, for back compatibility), text (multiline str),
        # btn (button for special actions), bool, list, dict.
        # An 'options' entry can be added for types like string, to provide a dropdown of possible values.
        personality_config_template = ConfigTemplate(
            [
                {"name":"update_mode", "type":"str", "value":"rewrite", "options":["rewrite","edit"], "help":"The update mode specifies if the AI needs to rewrite the whole code which is a good idea if the code is not long or just update parts of the code which is more suitable for long codes."},
                {"name":"create_a_plan", "type":"bool", "value":False, "help":"Create a plan for the app before starting."},
                {"name":"generate_icon", "type":"bool", "value":False, "help":"Generate an icon for the application (requires tti to be active)."},
                {"name":"use_lollms_library", "type":"bool", "value":False, "help":"Activate this if the application requires interaction with lollms."},
                {"name":"use_lollms_tasks_library", "type":"bool", "value":False, "help":"Activate this if the application needs to use text code extraction, text summary, yes no question answering, multi choice question answering etc."},
                {"name":"use_lollms_rag_library", "type":"bool", "value":False, "help":"(not ready yet) Activate this if the application needs to use text code extraction, text summary, yes no question answering, multi choice question answering etc."},
                {"name":"use_lollms_image_gen_library", "type":"bool", "value":False, "help":"(not ready yet) Activate this if the application requires image generation."},
                {"name":"use_lollms_audio_gen_library", "type":"bool", "value":False, "help":"(not ready yet) Activate this if the application requires audio manipulation."},

                {"name":"use_lollms_localization_library", "type":"bool", "value":False, "help":"Activate this library if you want to automatically localize your application into multiple languages."},

                # Boolean configuration for enabling scripted AI
                #{"name":"make_scripted", "type":"bool", "value":False, "help":"Enables a scripted AI that can perform operations using python scripts."},
                
                # String configuration with options
                #{"name":"response_mode", "type":"string", "options":["verbose", "concise"], "value":"concise", "help":"Determines the verbosity of AI responses."},
                
                # Integer configuration example
                #{"name":"max_attempts", "type":"int", "value":3, "help":"Maximum number of attempts for retryable operations."},
                
                # List configuration example
                #{"name":"favorite_topics", "type":"list", "value":["AI", "Robotics", "Space"], "help":"List of favorite topics for personalized responses."}
            ]
        )
        self.application_categories = [
            "Productivity",
            "Games",
            "Communication",
            "Entertainment",
            "Finance",
            "Health & Fitness",
            "Education",
            "Travel & Navigation",
            "Utilities",
            "Creative",
            "E-commerce"
        ]
        personality_config_vals = BaseConfig.from_template(personality_config_template)

        personality_config = TypedConfig(
            personality_config_template,
            personality_config_vals
        )
        
        super().__init__(
                            personality,
                            personality_config,
                            states_list=[
                                {
                                    "name": "idle",
                                    "commands": {
                                        "help": self.help, # Command triggering the help method
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )

    def mounted(self):
        """
        triggered when mounted
        """
        pass


    def selected(self):
        """
        triggered when selected
        """
        pass
        # self.play_mp3(Path(__file__).parent.parent/"assets"/"borg_threat.mp3")


    # Note: Remember to add command implementations and additional states as needed.

    def install(self):
        """
        Install the necessary dependencies for the personality.

        This method is responsible for setting up any dependencies or environment requirements
        that the personality needs to operate correctly. It can involve installing packages from
        a requirements.txt file, setting up virtual environments, or performing initial setup tasks.
        
        The method demonstrates how to print a success message using the ASCIIColors helper class
        upon successful installation of dependencies. This step can be expanded to include error
        handling and logging for more robust installation processes.

        Example Usage:
            processor = Processor(personality)
            processor.install()
        
        Returns:
            None
        """        
        super().install()
        # Example of implementing installation logic. Uncomment and modify as needed.
        # requirements_file = self.personality.personality_package_path / "requirements.txt"
        # subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")

    def help(self, prompt="", full_context=""):
        """
        Displays help information about the personality and its available commands.

        This method provides users with guidance on how to interact with the personality,
        detailing the commands that can be executed and any additional help text associated
        with those commands. It's an essential feature for enhancing user experience and
        ensuring users can effectively utilize the personality's capabilities.

        Args:
            prompt (str, optional): A specific prompt or command for which help is requested.
                                    If empty, general help for the personality is provided.
            full_context (str, optional): Additional context information that might influence
                                          the help response. This can include user preferences,
                                          historical interaction data, or any other relevant context.

        Example Usage:
            processor = Processor(personality)
            processor.help("How do I use the 'add_file' command?")
        
        Returns:
            None
        """
        # Example implementation that simply calls a method on the personality to get help information.
        # This can be expanded to dynamically generate help text based on the current state,
        # available commands, and user context.
        self.set_message_content(self.personality.help)

    def buildPlan(self, context_details, metadata, client:Client):
        self.step_start("Building initial_plan.txt")
        crafted_prompt = self.build_prompt(
            [
                self.system_full_header,
                "you are Lollms Apps Planner. Your objective is to build the initial plan for a specific lollms application.",
                "The user describes a web application and the ai should build the plaintext file and return it inside a plaintext markdown tag",
                "The plan should reformulate and cover all user specifications along with some enhancements from your own experience on such apps",
                "This plan should be consise yet precise."
                "Your sole objective is to build the description.yaml file. Do not ask the user for any extra information and only respond with the yaml content in a yaml markdown tag.",
                self.system_custom_header("context"),
                context_details["discussion_messages"],
                self.system_custom_header("Lollms Apps Planner")                
        ])
        app_plan = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink)
        codes = self.extract_code_blocks(app_plan)
        if len(codes)>0:
            app_plan = codes[0]["content"]
            # Store plan into context
            metadata["plan"]=app_plan
            client.discussion.set_metadata(metadata)
            self.step_end("Building initial_plan.txt")
            return app_plan
        self.step_end("Building initial_plan", False)
        return None

    def buildDescription(self, context_details, metadata, client:Client):
        self.step_start("Building description.yaml")
        crafted_prompt = self.build_prompt(
            [
                self.system_full_header,
                "you are Lollms Apps Maker. Your objective is to build the description.yaml file for a specific lollms application.",
                "The user describes a web application and the ai should build the yaml file and return it inside a yaml markdown tag",
                f"""
```yaml
name: Give a name to the application using the user provided information
description: Here you can make a detailed description of the application. do not use : or lists, just plain text in a single paragraph.
version: 1.0
author: make the user the author
category: give a suitable category name from {self.application_categories}
model: {self.personality.model.model_name}
disclaimer: If needed, write a disclaimer. else null
```
""",
                "If the user explicitely proposed a name, use that name",
                "Your sole objective is to build the description.yaml file. Do not ask the user for any extra information and only respond with the yaml content in a yaml markdown tag.",
                self.system_custom_header("context"),
                context_details["discussion_messages"],
                self.system_custom_header("Lollms Apps Maker")
            ],6
        )
        
        app_description = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink)
        codes = self.extract_code_blocks(app_description)
        if len(codes)>0:
            ASCIIColors.info(codes[0]["content"])
            infos = yaml.safe_load(codes[0]["content"].encode('utf-8').decode('ascii', 'ignore'))
            infos["creation_date"]=datetime.now().isoformat()
            infos["last_update_date"]=datetime.now().isoformat()
            if self.config.debug:
                ASCIIColors.yellow("--- Description file ---")
                ASCIIColors.yellow(infos)
            app_path = self.personality.lollms_paths.apps_zoo_path/infos["name"].replace(" ","_")
            app_path.mkdir(parents=True, exist_ok=True)
            metadata["app_path"]=str(app_path)
            metadata["infos"]=infos
            client.discussion.set_metadata(metadata)
            self.step_end("Building description.yaml")
            return infos
        else:
            self.step_end("Building description.yaml", False)
            return None

    def updateDescription(self, context_details, metadata, old_infos, client:Client):
        self.step_start("Building description.yaml")
        crafted_prompt = self.build_prompt(
            [
                self.system_full_header,
                "you are Lollms Apps Maker. Your objective is to build the description.yaml file for a specific lollms application.",
                "The user is acsking to modify the description file of the web application and the ai should build the yaml file and return it inside a yaml markdown tag",
                f"""
```yaml
name: {old_infos.get("name", "the name of the app")}
description: {old_infos.get("description", "the description of the app")}
version: {old_infos.get("version", "the version of the app (if not specified by the user it should be 1.0)")}
author: {old_infos.get("author", "the author of the app (if not specified by the user it should be the user name)")}
category: {old_infos.get("category", "the category of the app")}
model: {self.personality.model.model_name}
disclaimer: {old_infos.get("disclaimer", "If needed, write a disclaimer. else null")} 
```
""",
                "If the user explicitely proposed a name, use that name",
                "Your sole objective is to build the description.yaml file. Do not ask the user for any extra information and only respond with the yaml content in a yaml markdown tag.",
                self.system_custom_header("context"),
                context_details["discussion_messages"],
                self.system_custom_header("Lollms Apps Maker")
            ],6
        )
        
        app_description = self.generate(crafted_prompt,512,0.1,10,0.98, debug=True, callback=self.sink)
        codes = self.extract_code_blocks(app_description)
        if len(codes)>0:
            infos = yaml.safe_load(codes[0]["content"].encode('utf-8').decode('ascii', 'ignore'))
            infos["creation_date"]=old_infos["creation_date"]
            infos["last_update_date"]=datetime.now().isoformat()
            if self.config.debug:
                ASCIIColors.yellow("--- Description file ---")
                ASCIIColors.yellow(infos)
            app_path = self.personality.lollms_paths.apps_zoo_path/infos["name"].replace(" ","_")
            app_path.mkdir(parents=True, exist_ok=True)
            metadata["app_path"]=str(app_path)
            metadata["infos"]=infos
            client.discussion.set_metadata(metadata)
            self.step_end("Building description.yaml")
            return infos
        else:
            self.step_end("Building description.yaml", False)
            return None

    def buildIndex(self, context_details, plan, infos, metadata, client:Client):
        self.step_start("Building index.html")
        if self.personality_config.use_lollms_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_client_js_info.md","r", errors="ignore") as f:
                lollms_infos = f.read()
        else:
            lollms_infos = ""
        if self.personality_config.use_lollms_rag_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_rag_info.md","r", errors="ignore") as f:
                lollms_infos += f.read()


        if self.personality_config.use_lollms_localization_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_auto_localizer.md","r", errors="ignore") as f:
                lollms_infos += f.read()

        crafted_prompt = self.build_prompt(
            [
                self.system_full_header,
                "you are Lollms Apps Maker. Your objective is to build the index.html file for a specific lollms application.",
                "The user describes a web application and the ai should build a single html code to fullfill the application requirements.",
                "Make sure the application is visually appealing and try to use reactive design with tailwindcss",
                "The output must be in a html markdown code tag",
                "Your sole objective is to build the index.yaml file that does what the user is asking for.",
                "Do not ask the user for any extra information and only respond with the html content in a html markdown tag.",
                self.system_custom_header("context"),
                context_details["discussion_messages"],
                "\n".join([
                "```yaml",
                str(infos),
                "```"
                ]) if plan is None else "Plan:\n"+plan,                        
                lollms_infos,
                self.system_custom_header("Lollms Apps Maker")
            ],6
        )
        code_content = self.generate(crafted_prompt,temperature=0.1, top_k=10, top_p=0.98, debug=True, callback=self.sink)
        if self.config.debug:
            ASCIIColors.yellow("--- Code file ---")
            ASCIIColors.yellow(code_content)
        codes = self.extract_code_blocks(code_content)
        app_path = metadata["app_path"]
        if len(codes)>0:
            code = codes[0]["content"]
            # Backup the existing index.html file
            index_file_path = Path(metadata["app_path"]) / "index.html"
            if index_file_path.exists():
                try:
                    if not (app_path / ".git").exists():
                        repo = git.Repo.init(app_path)
                    else:
                        repo = git.Repo(app_path)

                    # Stage the current version of index.html
                    repo.index.add([os.path.relpath(index_file_path, app_path)])
                    repo.index.commit("Backup before update")
                except Exception as ex:
                    trace_exception(ex)
            self.step_end("Building index.html")
            return code
        else:
            self.step_end("Building index.html", False)
            self.set_message_content("The model you are using failed to build the index.html file. Change the prompt a bit and try again.")
            return None

    def update_index(self, context_details, metadata, out:str):
        if not metadata.get("app_path", None):
            self.set_message_content("""
<div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <h3 style="margin-top: 0;">⚠️ No Application Path Found</h3>
    <p>It appears that no application path is present in this discussion. Before attempting to make updates, you need to create a new project first.</p>
    <p>Please ask about creating a new project, and I'll be happy to guide you through the process.</p>
</div>
            """)
            return

        out = ""
        self.step_start("Updating index.html")
        
        if self.personality_config.use_lollms_library:
            with open(Path(__file__).parent.parent/"assets"/"docs"/"lollms_client_js_info.md","r", errors="ignore") as f:
                lollms_infos = f.read()
        else:
            lollms_infos = ""

        app_path = Path(metadata["app_path"])
        index_file_path = app_path / "index.html"

        # Initialize Git repository if not already initialized
        if not (app_path / ".git").exists():
            repo = git.Repo.init(app_path)
        else:
            repo = git.Repo(app_path)

        with open(index_file_path, "r", encoding="utf8") as f:
            original_content = f.read()


        if self.personality_config.update_mode=="rewrite":
            crafted_prompt = self.build_prompt(
                [
                    self.system_full_header,
                    "You are Lollms Apps Maker. Your objective is to update the HTML, JavaScript, and CSS code for a specific lollms application.",
                    "The user gives the code the AI should rewrite the code while applying the modifications suggested by the user",
                    "Always write the output in a html markdown tag",
                    self.system_custom_header("context"),
                    context_details["discussion_messages"],
                    lollms_infos,
                    self.system_custom_header("Code"),
                    "index.html",
                    "```html",
                    original_content,
                    "```",
                    self.system_custom_header("Lollms Apps Maker")
                ],5
            )

            updated_sections = self.generate(crafted_prompt, temperature=0.1, top_k=10, top_p=0.98, debug=True, callback=self.sink)

            # Extract code blocks
            codes = self.extract_code_blocks(updated_sections)
            if len(codes) > 0:
                content = codes[0]["content"]    
                self.step_start("Backing up previous version")       
                # Stage the current version of index.html
                repo.index.add([os.path.relpath(index_file_path, app_path)])
                repo.index.commit("Backup before update")
                self.step_end("Backing up previous version")
                while not codes[0]["is_complete"]:
                    updated_sections = self.generate(crafted_prompt+updated_sections+"\n"+self.user_full_header("system request")+"Continue coding from last line in a html markdown code tag.\n"+self.ai_full_header(), temperature=0.1, top_k=10, top_p=0.98, debug=True, callback=self.sink)
                    codes = self.extract_code_blocks(updated_sections)
                    if len(codes) > 0:
                        content = "\n".join(content.split("\n")[:-1])+codes[0]["content"]    
                    else:
                        return None
                # Write the updated content back to index.html
                index_file_path.write_text(codes[0]["content"], encoding='utf8')
                
                # Stage and commit the changes
                repo.index.add([os.path.relpath(index_file_path, app_path)])
                repo.index.commit("Update index.html")
                
                out += f"Updated index file:\n```html\n{codes[0]['content']}\n```\n"
            else:
                out += "No sections were updated."

            self.step_end("Updating index.html")

            self.set_message_content_invisible_to_ai(out)            
        else:
            crafted_prompt = self.build_prompt(
                [
                    self.system_full_header,
                    "You are Lollms Apps Maker. Your objective is to update the HTML, JavaScript, and CSS code for a specific lollms application.",
                    "The user gives the code the AI should update the code parts using the following syntax",
                    "To update existing code:",
                    "```python",
                    "# REPLACE",
                    "# ORIGINAL",
                    "<old_code>",
                    "# SET",
                    "<new_code_snippet>",
                    "```",
                    "The ORIGINAL statement (<old_code>) should contain valid code from the orginal code. It should be a full statement and not just a fragment of a statement.",
                    "The SET statement (<new_code_snippet>) is mandatory. You should put the new lines of code just after it.",
                    "Make sure if possible to change full statements or functions. The code to SET must be fully working and without placeholders.",
                    "If there is no ambiguity, just use a single line of code for each (first line to be changed and last line to be changed).",
                    "When providing code changes, make sure to respect the indentation in Python. Only provide the changes, do not repeat unchanged code. Use comments to indicate the type of change.",
                    "If too many changes needs to be done, and you think a full rewrite of the code is much more adequate, use this syntax:",
                    "```python",
                    "# FULL_REWRITE",
                    "<new_full_code>",
                    "```",
                    "Select the best between full rewrite and replace according to the amount of text to update.",
                    "Update the code from the user suggestion",
                    self.system_custom_header("context"),
                    context_details["discussion_messages"],
                    lollms_infos,
                    self.system_custom_header("Code"),
                    "<file_name>index.html</file_name>",
                    "```html",
                    original_content,
                    "```",
                    self.system_custom_header("Lollms Apps Maker")
                ],24
            )

            updated_sections = self.generate(crafted_prompt, temperature=0.1, top_k=10, top_p=0.98, debug=True, callback=self.sink)

            # Extract code blocks
            codes = self.extract_code_blocks(updated_sections)
            if len(codes) > 0:
                # Stage the current version of index.html
                repo.index.add([os.path.relpath(index_file_path, app_path)])
                repo.index.commit("Backup before update")

                for code_block in codes:
                    out_ = self.update_code(original_content, code_block["content"])
                    if out_["hasQuery"]:
                        out += f"Updated index file with new code\n"
                    else:
                        print(f"Warning: The AI did not manage to update the code!")
                
                # Write the updated content back to index.html
                index_file_path.write_text(out_["updatedCode"], encoding='utf8')
                
                # Stage and commit the changes
                repo.index.add([os.path.relpath(index_file_path, app_path)])
                repo.index.commit("Update index.html")
                
                out += f"Updated index file:\n```html\n{out_['updatedCode']}\n```\n"
            else:
                out += "No sections were updated."

            self.step_end("Updating index.html")


    def generate_icon(self, metadata, infos, client):
        app_path = Path(metadata["app_path"])
        if not (app_path / ".git").exists():
            repo = git.Repo.init(app_path)
        else:
            repo = git.Repo(app_path)

        if self.personality_config.generate_icon:
            self.step_start("Generating icon")
            crafted_prompt = self.build_prompt(
                [
                    "Make an icon for this application:"
                    "```yaml",
                    str(infos),
                    "```",
                    "The icon should depict the essence of the application as described in the description."
                ]
            )
            icon_infos = build_image_from_simple_prompt(crafted_prompt, self, client, production_type="icon")
            
            icon_src = str(Path(icon_infos["path"]))
            icon_dst = str(app_path/"icon.png")
            shutil.copy(icon_src, icon_dst)
            self.step_end("Generating icon")
        else:
            self.step_start("Using default icon")
            # Copy icon.png
            icon_src = str(Path(__file__).parent.parent/"assets"/"icon.png")
            icon_dst = str(app_path/"icon.png")
            shutil.copy(icon_src, icon_dst)
            
            # Stage and commit the icon
            repo.index.add([os.path.relpath(icon_dst, app_path)])
            repo.index.commit("Add icon.png")        
            self.step_end("Using default icon")
        return icon_dst

    def create_git_repository(self, infos, metadata):
        self.step_start("Initializing Git repository")
        app_path = metadata["app_path"]
        if not (app_path / ".git").exists():
            repo = git.Repo.init(app_path)
        else:
            repo = git.Repo(app_path)


        # Create .gitignore file
        gitignore_content = """
        # Ignore common Python files
        __pycache__/
        *.py[cod]
        *$py.class

        # Ignore common development files
        .vscode/
        .idea/

        # Ignore OS generated files
        .DS_Store
        Thumbs.db

        # Ignore backup files
        index_v*.html
        """
        with open(Path(metadata["app_path"]) / ".gitignore", "w") as f:
            f.write(gitignore_content)

        # Create README.md file
        readme_content = f"""
        # {infos['name']}

        {infos['description']}

        ## Version
        {infos['version']}

        ## Author
        {infos['author']}

        ## Category
        {infos['category']}

        ## Disclaimer
        {infos.get('disclaimer', 'N/A')}

        ## How to use
        1. Open the `index.html` file in a web browser.
        2. Follow the on-screen instructions to use the application.

        ## Development
        This project was created using the Lollms Apps Maker. To make changes, edit the `index.html` file and commit your changes to the Git repository.
        """
        with open(Path(metadata["app_path"]) / "README.md", "w") as f:
            f.write(readme_content)

        # Add all files to Git
        repo.git.add(all=True)

        # Commit the initial code
        repo.index.commit("Initial commit")

        self.step_end("Initializing Git repository")

    
    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str | list | None, MSG_OPERATION_TYPE, str, AIPersonality| None], bool]=None, context_details:dict=None, client:Client=None):
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
        metadata = client.discussion.get_metadata()

        choices = self.multichoice_question("select the best suited option", [
                "The user is discussing",
                "The user is asking to build the webapp",
                "The user is asking for a modification or reporting a bug in the weapp",
                "The user is asking for a modification of the information (description, author, vertion etc)",
                "The user is asking for a modification of the information (description, author, vertion etc)",
        ], context_details["discussion_messages"])
        if choices ==0:
            extra_infos="""
The Lollms apps maker is a lollms personality built for making lollms specific apps.
Lollms apps are webapps with a possible fastapi backend. These webapps are created in html/css/javascript and can interact with lollms is the option is activated in the settings.
Multiple libraries can be activated in the settings of the personality to instruct the personality to use them.
The lollms libraries suite allows your app to interact with lollms and benefits from its generative capabilities (text to speach, text to image, text to music, text to code etc..)
The code contains description.yaml that describes the application, the author, the creation date and a short description.
"""
            self.answer(context_details+"Extra infos about the process:"+extra_infos)
        elif choices ==1:

            out = ""
            out += "You asked me to build an app. I am building the description file."
            self.set_message_content_invisible_to_ai(out)

            # ----------------------------------------------------------------
            infos = self.buildDescription(context_details, metadata, client)
            if infos is None:
                out += "\n<p style='color:red'>It looks like I failed to build the description.<br>That's the easiest part to do!! If the model wasn't able to do this simple task, I think you better change it, or maybe give it another shot.<br>As you know, I depend highly on the model I'm running on. Please give me a better brain and plug me to a good model.</p>"
                self.set_message_content_invisible_to_ai(out)
                return
            with open(Path(metadata["app_path"])/"description.yaml","w", encoding="utf8") as f:
                yaml.dump(infos, f, encoding="utf8")
            out += "\n".join([
                "\nDescription built successfully !",
                "Here is the metadata built for this app:",
                "```yaml",
                yaml.dump(infos, default_flow_style=False),
                "```"
            ])
            self.set_message_content_invisible_to_ai(out)
            # ----------------------------------------------------------------
            if self.personality_config.create_a_plan:
                out += "\nIn my settings, you have activated planning, so let me build a plan for the application."
                self.set_message_content_invisible_to_ai(out)
                plan = self.buildPlan(context_details, metadata, client)
                if plan:
                    with open(Path(metadata["app_path"])/"plan.md","w", encoding="utf8") as f:
                        f.write(plan)
                    out += f"\nThe plan is ready.\nHere is my plan\n{plan}"
                    self.set_message_content_invisible_to_ai(out)
                else:
                    out += "\n<p style='color:red'>It looks like I failed to build the plan. As you know, I depend highly on the model I'm running on. Please give me a better brain and plug me to a good model.</p>"
                    self.set_message_content_invisible_to_ai(out)
                    return
            else:
                plan = None
            # ----------------------------------------------------------------
            out +="\nOk, now I'm building the real thing. The code for our application. Please wait as this may take a little time."
            self.set_message_content_invisible_to_ai(out)
            code = self.buildIndex(context_details, plan, infos, metadata, client)
            if code:
                with open(Path(metadata["app_path"])/"index.html","w", encoding="utf8") as f:
                    f.write(code)
                out +=f"\n<p style='color:green'>Coding done successfully.</p>"
                self.set_message_content_invisible_to_ai(out)
            else:
                out +=f"\n<p style='color:red'>It looks like I failed to build the code. I think the model you are using is not smart enough to do the task. I remind you that the quality of my output depends highly on the model you are using. Give me a better brain if you want me to do better work.</p>"
                self.set_message_content_invisible_to_ai(out)

            # ----------------------------------------------------------------
            out += "\nBefore we end, let's build an icon. I'll use the default icon if you did not specify build icon in my settings. You can build new icons whenever you cant in the future, just ask me to make a new icon And I'll do (ofcourse, lollms needs to have its TTI active)."
            self.set_message_content_invisible_to_ai(out)
            icon_dst = self.generate_icon(metadata, infos, client)
            icon_url = app_path_to_url(icon_dst)
            out += "\n" + f'\n<img src="{icon_url}" style="width: 200px; height: 200px;">'
            out += f"Refresh the apps zoo and you should find this app at category {infos['category']}"
            self.set_message_content_invisible_to_ai(out)
            # Show the user everything that was created
            out = f"""
<div class="panels-color bg-gray-100 p-6 rounded-lg shadow-md">
    <h2 class="text-2xl font-bold mb-4">Application Created Successfully!</h2>
    <img  src ="{icon_url}" style="width: 200px; height: 200px;">
    <p class="mb-4">Your application <span class="font-semibold">{infos['name']}</span> has been created in the following directory:</p>
    <pre class="panel-color p-2 rounded">{metadata["app_path"]}</pre>
    <h3 class="text-xl font-bold mt-6 mb-2">Files created:</h3>
    <ul class="list-disc list-inside">
        <li>description.yaml</li>
        <li>index.html</li>
        <li>icon.png</li>
        <li>.gitignore</li>
        <li>README.md</li>
    </ul>
    <h3 class="text-xl font-bold mt-6 mb-2">Git Repository:</h3>
    <p>A Git repository has been initialized in the application folder with an initial commit.</p>
    <h3 class="text-xl font-bold mt-6 mb-2">Next Steps:</h3>
    <ol class="list-decimal list-inside">
        <li>Refresh the apps zoo and you should find this app at category {infos['category']}</li>
        <li>Review the created files and make any necessary adjustments.</li>
        <li>Test the application by opening the index.html file in a web browser.</li>
        <li>Continue development by making changes and committing them to the Git repository.</li>
    </ol>
</div>
            """
            self.ui(out)
            client.discussion.set_metadata(metadata)
        elif choices ==2:
            out = ""
            self.update_index(context_details, metadata, out)
        elif choices ==2:
            out = ""
            infos = self.updateDescription(context_details, metadata, client)
            if infos is None:
                out += "\n<p style='color:red'>It looks like I failed to build the description. That's the easiest part to do. As you know, I depend highly on the model I'm running on. Please give me a better brain and plug me to a good model.</p>"
                self.set_message_content_invisible_to_ai(out)
                return
            with open(Path(metadata["app_path"])/"description.yaml","w", encoding="utf8") as f:
                yaml.dump(infos, f, encoding="utf8")
            out += "\n".join([
                "\nDescription built successfully !",
                "Here is the metadata built for this app:",
                "```yaml",
                yaml.dump(infos, default_flow_style=False),
                "```"
            ])
            self.set_message_content_invisible_to_ai(out)

        elif choices ==3:
            out = "I'm generating a new icon based on your request.\n"
            self.set_message_content_invisible_to_ai(out)
            out += self.generate_icon(metadata, infos, client)
            self.set_message_content_invisible_to_ai(out)


            

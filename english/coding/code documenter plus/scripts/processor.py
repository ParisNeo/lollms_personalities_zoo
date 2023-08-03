import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors
from lollms.utilities import PackageManager
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import re
import importlib
import requests
from tqdm import tqdm
import shutil
from lollms.types import GenerationPresets
import json
from functools import partial

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
        self.word_callback = None
        personality_config_template = ConfigTemplate(
            [
                {"name":"project_path","type":"str","value":'', "help":"The path to the project to document"},
                {"name":"layout_max_size","type":"int","value":2048, "min":10, "max":personality.config["ctx_size"]},
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
                                        "start_doc": self.start_documenting
                                    },
                                    "default": self.idle
                                },                               
                            ],
                            callback=callback
                        )
        self.previous_versions = []
        self.code=[]
        
    def install(self):
        super().install()
        # Get the current directory
        root_dir = self.personality.lollms_paths.personal_path
        # We put this in the shared folder in order as this can be used by other personalities.
        shared_folder = root_dir/"shared"

        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Step 2: Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])            
        ASCIIColors.success("Installed successfully")
        
        
    def idle(self, prompt, full_context):
        structure = self.parse_python_code(prompt)
        text=f"""Json structure of the code:
{json.dumps(structure)}
!@>instruction: Document this python code by describing the used libraries, the classes and their use, how to use the methods etc. 
Please use markdown format for your output.
!@>documentation:
# Introduction:"""
        out = "# Introduction:"+ self.generate(text,self.personality_config.layout_max_size)
        self.full(out)
        self.new_message("document structure", MSG_TYPE.MSG_TYPE_JSON_INFOS,structure)
        
    def path_to_json(self, path):
        if not isinstance(path, Path):
            raise ValueError("Input must be a pathlib.Path object.")

        result = {}

        if path.is_file():
            return path.name

        if path.is_dir():
            result[path.name] = {}

            for item in path.iterdir():
                result[path.name][item.name] = self.path_to_json(item)

        return result
    
    def process_python_files(self, path, file_function):
        if not isinstance(path, Path):
            raise ValueError("Input 'path' must be a pathlib.Path object.")
        if not callable(file_function):
            raise ValueError("Input 'file_function' must be a callable function.")

        if path.is_file():
            if path.suffix == '.py':
                file_function(path)
        elif path.is_dir():
            for item in path.iterdir():
                self.process_python_files(item, file_function)
                
                
    def start_documenting(self, prompt, full_context):
        if self.personality_config.project_path=="":
            self.warning("Please select a project path in personality settings first")
            return
        project_path = Path(self.personality_config.project_path)
        if not project_path.exists():
            self.warning("Please select a project path in personality settings first")
        else:
            self.step_start(f"-- Started documentation of {project_path} --")
            docs_dir=project_path/"docs"
            docs_dir.mkdir(parents=True, exist_ok=True)
            structure = self.path_to_json(docs_dir)
            text=f"""Json structure of the project folder:
{json.dumps(structure)}
!@>instruction: Create a description of the project structure.
!@>documentation:
# Project structure:"""
            doc = "# Project structure:"+ self.generate(text,self.personality_config.layout_max_size)
            with open(docs_dir/"project_structure.md","w") as f:
                f.write(doc)
            self.process_python_files(project_path, partial(self.parse_python_file, docs_dir=docs_dir))
            self.step_end(f"-- Started documentation of {project_path} --")
            
            
                    
    def parse_python_code(self, source_code: str):
        try:
            import ast
        except:
            PackageManager.install_package("ast")
            import ast

        tree = ast.parse(source_code)

        imports = []
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    imports.append(import_name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    import_name = node.module + '.' + alias.name
                    imports.append(import_name)
            elif isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_code = ast.unparse(node)
                functions.append({"name": function_name, "content": function_code})
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                methods = []

                for child_node in ast.walk(node):
                    if isinstance(child_node, ast.FunctionDef):
                        method_name = child_node.name
                        method_code = ast.unparse(child_node)
                        methods.append({"name": method_name, "content": method_code})

                classes.append({"name": class_name, "methods": methods})

        # Remove functions that are found within classes
        top_level_functions = [func for func in functions if not any(func['name'] == method['name'] for cls in classes for method in cls['methods'])]

        result = {
            "imports": imports,
            "functions": top_level_functions,
            "classes": classes
        }

        return result

    def find_common_path_parts(self, path1, path2):
        if not isinstance(path1, Path) or not isinstance(path2, Path):
            raise ValueError("Inputs must be pathlib.Path objects.")

        common_parts = path1.resolve().parts[:len(path2.resolve().parts)]
        extra_path1 = path1.resolve().relative_to(*common_parts)
        extra_path2 = path2.resolve().relative_to(*common_parts)

        return Path(*common_parts), extra_path1, extra_path2

       
    def parse_python_file(self, file_path:Path, docs_dir:Path):
        if file_path is not None:
            with open(file_path, 'r') as file:
                source_code = file.read()
            doc =  self.parse_python_code(source_code)
            _,extra_path1,_ = self.find_common_path_parts(file_path, docs_dir)
            output_file_path = docs_dir / extra_path1
            output_file_path.suffix=".md"
            with open(output_file_path,"w") as f:
                f.write(doc)
                


    def convert_string_to_sections(self, string):
        table_of_content = ""
        lines = string.split('\n')  # Split the string into lines
        sections = []
        current_section = None
        for line in lines:
            line = line.strip()
            if line!="":
                table_of_content+=line+"\n"
                if line.startswith('## '):  # Detect section
                    section_title = line.replace('## ', '')
                    current_section = {'title': section_title, 'subsections': [], "content":""}
                    sections.append(current_section)
                elif line.startswith('### '):  # Detect subsection
                    if current_section is not None:
                        subsection_title = line.replace('### ', '')
                        current_section['subsections'].append(subsection_title)
        return sections, table_of_content

    def run_workflow(self, prompt, full_context="", callback=None):
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

        self.process_state(prompt, full_context, callback)
        



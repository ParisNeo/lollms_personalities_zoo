# processor.py

from typing import Callable, Dict, Any
from lollms.personality import APScript, AIPersonality
from lollms.types import MSG_OPERATION_TYPE
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
import json
from pathlib import Path
import subprocess
from ascii_colors import ASCIIColors, trace_exception
import os

class Processor(APScript):
    def __init__(self, personality: AIPersonality, callback: Callable = None) -> None:
        personality_config_template = ConfigTemplate([
            {"name": "work_folder", "type": "str", "value":"work_folder", "help": "The working directory"}
        ])
        personality_config_vals = BaseConfig.from_template(personality_config_template)
        personality_config = TypedConfig(personality_config_template, personality_config_vals)
        
        super().__init__(
            personality,
            personality_config,
            states_list=[
                {
                    "name": "idle",
                    "commands": {
                        "help": self.help,
                    },
                    "default": None
                },
            ],
            callback=callback
        )
        self.workflow_steps = []
        self.project_details = None
        self.project_structure = {}

    def mounted(self):
        self.step_start("Personality mounted successfully.")

    def selected(self):
        self.step_start("Personality selected.")

    def install(self):
        self.step_start("Installing necessary dependencies.")
        self.step_end("Dependencies installed.")

    def help(self, prompt="", full_context=""):
        help_text = (
            "This personality helps you build Python projects step by step. "
            "Provide an instruction to get started."
        )
        self.set_message_content(help_text)

    def run_workflow(self, prompt: str, previous_discussion_text: str = "", callback: Callable = None,
                     context_details: Dict[str, Any] = None):
        self.callback = callback
        self.step_start("Starting project build workflow.")

        # Check if work_folder is set
        work_folder = self.personality.config.work_folder
        if not work_folder or work_folder == "work_folder":
            self.step_start("Work folder not set. Please set the work_folder in the personality configuration.")
            return

        self.work_folder = Path(work_folder)
        self.work_folder.mkdir(parents=True, exist_ok=True)

        # Parse the instruction
        self.project_details = self.parse_instruction(prompt)
        if not self.project_details:
            self.step_start("Failed to parse the instruction. Please try again with a clearer instruction.")
            return

        # Execute the project tasks
        self.execute_project()

    def parse_instruction(self, prompt: str) -> Dict[str, Any]:
        formatted_prompt = (
            "Please provide a JSON representation of the steps to fulfill the following instruction: "
            f"{prompt}. The JSON should include a project title (string), project type (string), and a list of tasks (array), "
            "where each task is an object with the following structure:\n"
            "- task (string): The type of task (e.g., 'execute_command', 'write_code', 'run_application').\n"
            "- parameters (object): An object containing specific parameters for the task type. The parameters should be:\n"
            "  - For 'execute_command': command (string)\n"
            "  - For 'write_code': file_name (string)\n"
            "  - For 'run_application': command (string)\n"
            "Make sure to include the JSON delimiters as follows:\n"
            "```json\n"
            "YOUR_JSON_HERE\n"
            "```\n"
        )
        
        json_str = self.generate_code(formatted_prompt)
        
        if json_str:
            try:
                json_response = json.loads(json_str)
                return json_response
            except Exception as ex:
                trace_exception(ex)
                ASCIIColors.error("Error decoding JSON response or extracting JSON part.")
                return None
        
        return None

    def generate_file_content(self, user_instructions, file_name: str, project_context: Dict[str, Any]) -> str:
        prompt = (
            f"User prompt: {user_instructions}"
            f"Generate the content for the file '{file_name}' in the context of the following project structure:\n"
            f"{json.dumps(project_context, indent=2)}\n\n"
            "Please provide the code for this file, ensuring it's compatible with the existing project structure and functions."
        )
        return self.generate_code(prompt)

    def extract_functions(self, code: str) -> Dict[str, Any]:
        prompt = (
            "Extract the functions, classes, and methods from the following code and return them as a JSON object. "
            "Include the function/class/method names, parameters, and brief descriptions.\n\n"
            f"Code:\n{code}\n\n"
            "Return the JSON in the following format:\n"
            "```json\n"
            "{\n"
            "  \"functions\": [\n"
            "    {\"name\": \"function_name\", \"parameters\": [\"param1\", \"param2\"], \"description\": \"Brief description\"}\n"
            "  ],\n"
            "  \"classes\": [\n"
            "    {\n"
            "      \"name\": \"ClassName\",\n"
            "      \"methods\": [\n"
            "        {\"name\": \"method_name\", \"parameters\": [\"param1\", \"param2\"], \"description\": \"Brief description\"}\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
        )
        json_str = self.generate_code(prompt)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    def execute_task(self, user_instructions, task: Dict[str, Any]) -> bool:
        task_type = task.get("task")
        parameters = task.get("parameters", {})

        if task_type == "execute_command":
            command = parameters.get("command")
            if command:
                self.step_start(f"Executing command: {command}")
                try:
                    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True, cwd=self.work_folder)
                    self.step_start(f"Command output: {result.stdout}")
                    return True
                except subprocess.CalledProcessError as e:
                    self.step_start(f"Error executing command: {e.stderr}")
                    return False

        elif task_type == "write_code":
            file_name = parameters.get("file_name")
            if file_name:
                self.step_start(f"Generating code for {file_name}")
                code = self.generate_file_content(user_instructions, file_name, self.project_structure)
                file_path = self.work_folder / file_name
                with file_path.open('w') as file:
                    file.write(code)
                self.step_start(f"Wrote code to {file_path}")
                
                # Extract functions and update project structure
                functions_info = self.extract_functions(code)
                self.project_structure[file_name] = functions_info
                self.step_start(f"Updated project structure with {file_name}")
                return True
            else:
                self.step_start("Missing file name")
                return False

        elif task_type == "run_application":
            command = parameters.get("command")
            if command:
                self.step_start(f"Running application with command: {command}")
                try:
                    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True, cwd=self.work_folder)
                    self.step_start(f"Application output: {result.stdout}")
                    return True
                except subprocess.CalledProcessError as e:
                    self.step_start(f"Error running application: {e.stderr}")
                    return False

        else:
            self.step_start(f"Unknown task type: {task_type}")
            return False

    def execute_project(self):
        if not self.project_details:
            self.step_start("No project details available. Please parse the instruction first.")
            return

        self.step_start("Executing project tasks...")

        for task in self.project_details.get("tasks", []):
            success = self.execute_task(self.project_details.get("title", ""), task)
            if not success:
                self.step_start(f"Failed to execute task: {task['task']}")
                return

        self.step_start("Project execution completed successfully.")

if __name__ == "__main__":
    # This block is for testing the Processor class independently
    class DummyPersonality:
        def __init__(self):
            self.config = type('Config', (), {'work_folder': 'test_work_folder'})()

    dummy_personality = DummyPersonality()
    processor = Processor(dummy_personality)
    
    # Test run_workflow
    processor.run_workflow("Create a simple Python script that prints 'Hello, World!' and save it as hello.py")

# processor.py

from typing import Callable, Dict, Any
from lollms.personality import APScript, AIPersonality
from lollms.types import MSG_OPERATION_TYPE
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
import json
from pathlib import Path
import subprocess
from ascii_colors import ASCIIColors, trace_exception
import os, sys
from lollms.client_session import Client

class Processor(APScript):
    def __init__(self, personality: AIPersonality, callback: Callable = None) -> None:
        personality_config_template = ConfigTemplate([
            {"name": "work_folder", "type": "str", "value":"", "help": "The working directory"}
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
                     context_details: Dict[str, Any] = None, client:Client = None):
        self.callback = callback
        self.step_start("Starting project build workflow.")

        # Check if work_folder is set
        work_folder = self.personality_config.work_folder
        if not work_folder or work_folder == "":
            self.step_start("Work folder not set. Please set the work_folder in the personality configuration.")
            return

        self.work_folder = Path(work_folder)
        self.work_folder.mkdir(parents=True, exist_ok=True)

        # Parse the instruction
        self.project_details = self.parse_instruction(prompt)
        if not self.project_details:
            self.step_start("Failed to parse the instruction. Please try again with a clearer instruction.")
            return
        self.new_message("")

        # Execute the project tasks
        self.execute_project()

    def parse_instruction(self, prompt: str) -> Dict[str, Any]:
        formatted_prompt = (
            "Please provide a JSON representation of the steps to fulfill the following instruction: "
            f"{prompt}.\n"
            "The JSON should include a project title (string), project type (string), project_author:(string), platform (string), project description (string), and a list of tasks (array)"
            "where each task is an object with the following structure:\n"
            "- task (string): The type of task one of:\n"
            "   'execute_command': executes a command and returns the response.\n"
            "   'create_file': creates a file and triggers the generation of its content. Do not write the content in the generated json\n"
            "   'run_application': executes an application using a command.\n"
            "- parameters (object): An object containing specific parameters for the task type. The parameters should be:\n"
            "  - For 'execute_command': command (string)\n"
            "  - For 'create_file': file_name (string)\n"
            "  - For 'run_application': command (string)\n"
            "Make sure to include the JSON delimiters as follows:\n"
            "```json\n"
            "YOUR_JSON_HERE\n"
            "```\n"
            f"Platform information: {sys.platform}\n"
            f'User name: {self.config.user_name if self.config.user_name and self.config.user_name!="user" else "Lollms Project Builder"}\n'
            f"Make sure you make a rigorous setup and plan documentation."
        )
        
        json_str = self.generate_code(formatted_prompt,callback=self.sink)
        
        if json_str:
            try:
                json_response = json.loads(json_str)
                formatted_info = (
                    f"📋 Project Information:\n"
                    f"{'='*50}\n"
                    f"🏷️ Title: {json_response.get('project_title', 'N/A')}\n"
                    f"📁 Type: {json_response.get('project_type', 'N/A')}\n"
                    f"👤 Author: {json_response.get('project_author', 'N/A')}\n"
                    f"💻 Platform: {json_response.get('platform', 'N/A')}\n"
                    f"\n📝 Description:\n{json_response.get('project_description', 'N/A')}\n"
                    f"\n📋 Tasks:\n{'='*50}\n"
                )

                for idx, task in enumerate(json_response.get('tasks', []), 1):
                    task_type = task.get('task', 'N/A')
                    params = task.get('parameters', {})
                    
                    formatted_info += f"\n🔹 Task {idx}: {task_type}\n"
                    formatted_info += f"  Parameters:\n"
                    for param_key, param_value in params.items():
                        formatted_info += f"    - {param_key}: {param_value}\n"

                self.new_message(formatted_info)
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
        return self.generate_code(prompt, callback=self.sink)

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

    def execute_task(self, context_memory, task: Dict[str, Any]) -> bool:
        task_type = task.get("task")
        parameters = task.get("parameters", {})

        if task_type == "execute_command":
            command = parameters.get("command")
            if command:
                self.step_start(f"Executing command: {command}")
                try:
                    # For Windows
                    if os.name == 'nt':
                        commands = [f'cd /d "{self.work_folder}"', command]
                        full_command = ' && '.join(commands)
                        result = subprocess.run(full_command, shell=True, check=True, text=True, capture_output=True)
                    else:
                        commands = [f'cd "{self.work_folder}"', command]
                        full_command = ' && '.join(commands)
                        result = subprocess.run(full_command, shell=True, check=True, text=True, capture_output=True, executable='/bin/bash')
                    
                    # Show output in markdown format
                    output_md = f"""```
Command: {command}
Output:
{result.stdout if result.stdout and result.stdout!="" else "success"}
```"""
                    self.new_message(output_md)
                    
                    # Ask AI to analyze the output
                    analysis_prompt = f"""
Analyze the following command execution results and determine next steps:
Command: {command}
Output: {result.stdout if result.stdout and result.stdout!="" else "success"}
Error output: {result.stderr if result.stderr and result.stderr!="" else "no errors detected"}

Respond with a JSON containing:
1. status: "success" or "error"
2. message: Description of what happened
Example:
```json
{{
    "status": "success",
    "message": "Command executed successfully",
}}
```
"""
                    self.add_chunk_to_message_content("\n")
                    analysis_response = self.generate_code(analysis_prompt)
                    try:
                        analysis = json.loads(analysis_response)
                        if analysis.get("status") == "success":
                            self.step_start("Task completed successfully: " + analysis.get("message", ""))
                            return True
                        else:
                            self.step_start(f"Task failed: {analysis.get('message', 'Unknown error')}")
                            return False
                    except json.JSONDecodeError:
                        self.step_start("Failed to parse AI analysis response")
                        return False
                    
                except subprocess.CalledProcessError as e:
                    error_md = f"""```
Command: {command}
Error:
{e.stderr}
```"""
                    self.new_message(error_md)
                    return False

        elif task_type == "create_file":
            file_name = parameters.get("file_name")
            if file_name:
                self.step_start(f"Generating code for {file_name}")
                code = self.generate_file_content(context_memory, file_name, self.project_structure)
                file_path = self.work_folder / file_name
                
                try:
                    with file_path.open('w') as file:
                        file.write(code)
                    self.add_chunk_to_message_content(f"Content of file: {file_path} written successfuly\n")
                    return True
                except Exception as e:
                    self.step_start(f"Error writing file: {str(e)}")
                    return False

        elif task_type == "run_application":
            command = parameters.get("command")
            if command:
                self.step_start(f"Running application with command: {command}")
                try:
                    # For Windows
                    if os.name == 'nt':
                        commands = [f'cd /d "{self.work_folder}"', command]
                        full_command = ' && '.join(commands)
                        result = subprocess.run(full_command, shell=True, check=True, text=True, capture_output=True)
                    else:
                        commands = [f'cd "{self.work_folder}"', command]
                        full_command = ' && '.join(commands)
                        result = subprocess.run(full_command, shell=True, check=True, text=True, capture_output=True, executable='/bin/bash')
                    
                    # Show output in markdown format
                    output_md = f"""```
Application: {command}
Output:
{result.stdout}
```"""
                    self.new_message(output_md)
                    
                    # Ask AI to analyze the application output
                    analysis_prompt = f"""
Analyze the following application execution results:
Command: {command}
Output: {result.stdout if result.stdout and result.stdout!="" else "success"}
Error output: {result.stderr if result.stderr and result.stderr!="" else "no errors detected"}

Respond with a JSON containing:
1. status: "success" or "error"
2. message: Description of application behavior
"""
                    self.add_chunk_to_message_content("\n")
                    analysis_response = self.generate_code(analysis_prompt)
                    try:
                        analysis = json.loads(analysis_response)
                        if analysis.get("status") == "success":
                            return True
                        else:
                            self.step_start(f"Application run failed: {analysis.get('message', 'Unknown error')}")
                            return False
                    except json.JSONDecodeError:
                        self.step_start("Failed to parse AI analysis response")
                        return False
                    
                except subprocess.CalledProcessError as e:
                    error_md = f"""```
Application: {command}
Error:
{e.stderr}
```"""
                    self.new_message(error_md)
                    return False

        elif task_type == "finished":
            self.step_start("Workflow completed successfully.")
            return True

        else:
            self.step_start(f"Unknown task type: {task_type}")
            return False


    def execute_project(self):
        if not self.project_details:
            self.step_start("No project details available. Please parse the instruction first.")
            return

        self.step_start("Executing project tasks...")

        for task in self.project_details.get("tasks", []):
            success = self.execute_task(self.project_details, task)
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

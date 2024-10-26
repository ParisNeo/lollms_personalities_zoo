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
import platform
import tempfile
import time
import subprocess
import os
import sys
import time
import threading
import queue

class CommandExecutor:
    def __init__(self, work_dir = None, shell='cmd.exe' if os.name == 'nt' else '/bin/bash'):
        self.work_dir = work_dir
        self.shell = shell
        self.process = subprocess.Popen(
            self.shell,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )

        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()

        # Start threads to read stdout and stderr
        threading.Thread(target=self._enqueue_output, args=(self.process.stdout, self.stdout_queue), daemon=True).start()
        threading.Thread(target=self._enqueue_output, args=(self.process.stderr, self.stderr_queue), daemon=True).start()

        self.execute_command(f"cd {work_dir}")

    def _enqueue_output(self, stream, queue):
        for line in iter(stream.readline, ''):
            queue.put(line)
        stream.close()

    def execute_command(self, command, timeout=5):
        try:
            # Send the command to the shell
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()

            # Collect output and errors
            output = ''
            errors = ''
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Fetch stdout
                while not self.stdout_queue.empty():
                    line = self.stdout_queue.get()
                    output += line

                # Fetch stderr
                while not self.stderr_queue.empty():
                    line = self.stderr_queue.get()
                    errors += line

                # Check if the process has terminated
                if self.process.poll() is not None:
                    break

                time.sleep(0.1)

            return {
                'stdout': output,
                'stderr': errors,
                'returncode': self.process.returncode or 0
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'error': 'An error occurred'
            }

    def get_current_directory(self):
        if os.name == 'nt':
            result = self.execute_command('cd')
        else:
            result = self.execute_command('pwd')
        
        # Extract the actual path from the output
        path = result['stdout'].strip().split('\n')[-1]
        return path

    def create_file(self, filename, content):
        try:
            with open(filename, 'w') as f:
                f.write(content)
            return {'success': True, 'message': f"File {filename} created successfully"}
        except Exception as e:
            return {'success': False, 'message': f"Error creating file {filename}: {str(e)}"}

    def close(self):
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait(timeout=0.2)
class Processor(APScript):
    def __init__(self, personality: AIPersonality, callback: Callable = None) -> None:
        personality_config_template = ConfigTemplate([
            {"name": "work_folder", "type": "str", "value":"", "help": "The working directory"},
            {"name": "save_context_for_recovery", "type": "bool", "value":True, "help": "Saves current context for recovering from previous generation"},
            {"name": "max_retries", "type": "int", "value":1, "help": "When something fails, retry n times before stopping"},
            {"name": "build_image_assets", "type": "bool", "value":False, "help": "Build image assets"},
            {"name": "build_sound_assets", "type": "bool", "value":False, "help": "Build sound assets"},
            
            {"name": "verbose", "type": "bool", "value":False, "help": "If true, you will see all details in the message"}
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
            self.set_message_content("<b>Work folder not set. Please set the work_folder in the personality configuration.</b>")
            return

        self.work_folder = Path(work_folder)
        self.work_folder.mkdir(parents=True, exist_ok=True)

        # Parse the instruction
        self.project_details = self.parse_instruction(prompt)
        self.terminal = CommandExecutor(self.work_folder)
        if not self.project_details:
            self.step_start("Failed to parse the instruction. Please try again with a clearer instruction.")
            return
        self.add_chunk_to_message_content("\n")

        # Execute the project tasks
        self.execute_project()
        self.terminal.close()

    def parse_instruction(self, prompt: str) -> Dict[str, Any]:
        task_types = "execute_command, create_file, run_application"
        if self.personality.app.tti and self.personality_config.build_image_assets:
            image_generator = (
                "   {\n"
                '       "task_type":"generate_image", generates an image asset for the project\n'
                '       "file_name":"relative path to the file to be generated"\n'
                '       "generation_prompt":"A prompt for generating the image"\n'
                "   },\n"
            )
            task_types += ", generate_image"
        else:
            image_generator ="Do not use any image assets in the code"

        if self.personality.app.tts and self.personality_config.build_sound_assets:
            sound_generator = (
                "   {\n"
                '       "task_type":"generate_sound", generates a sound or music asset for the project\n'
                '       "file_name":"relative path to the file to be generated"\n'
                '       "generation_prompt":"A prompt for generating the sound or music"\n'
                "   },\n"
            )
            task_types += ", generate_sound"
        else:
            sound_generator ="Do not use any sound or music assets in the code\n"

        formatted_prompt = (
            "Please provide a JSON representation of the steps to fulfill the following instruction: "
            f"{prompt}.\n"
            "Json structure:"
            "```json\n"
            "{\n"

            '"project_title":"A string representing the project title",\n'
            '"project_type":"A string representing the project type",\n'
            f'"project_author":"{self.config.user_name if self.config.user_name and self.config.user_name!="user" else "Lollms Project Builder"}",\n'
            '"project_description":"A string representing the project type",\n'
            f'"platform":"{sys.platform}",\n'
            '"tasks":"[\n'
            "   {\n"
            '       "task_type":"execute_command", This executes a console command\n'
            '       "command":"command to execute"\n'
            "   },\n"
            "   {\n"
            '       "task_type":"create_file", This creates a file at a specific path\n'
            '       "file_name":"relative path to the file",\n'
            '       "content_description":"If the file is a code file, then write a structure of the file content and detailed funtions/classes/variables so that the file builder builds the full code. If the file is a documentation file, state that it is a documentation file without much detail",\n'
            "   },\n"
            "   {\n"
            '       "task_type":"run_application", This runs an application and returns back the output to check\n'
            '       "command":"the command to run the application"\n'
            "   },\n"+image_generator+sound_generator+""

            "]\n"
            "}\n"
            "```\n"
            "Only the following task types are allowed: "+task_types+".\n"
            "Make sure to include the JSON delimiters and respect the formatting.\n"
            "The tasks are executed in a consistant shell. The folders can be changed using cd.\n"
            "Make sure you make a rigorous setup and plan documentation.\n"
        )
        if self.personality_config.verbose:
            self.print_prompt("Generating project structure", formatted_prompt)

        json_str = self.generate_code(formatted_prompt,callback=self.sink)
        
        if json_str:
            try:
                json_response = json.loads(json_str)
                formatted_info = f'''
<div class="max-w-3xl mx-auto p-6 bg-white rounded-lg shadow-lg">
<div class="mb-6">
<h1 class="text-2xl font-bold text-gray-800 mb-4">📋 Project Information</h1>
<div class="border-b border-gray-300 mb-4"></div>

<div class="grid grid-cols-2 gap-4 mb-4">
    <div class="flex items-center">
        <span class="text-gray-600">🏷️ Title:</span>
        <span class="ml-2 font-medium">{json_response.get('project_title', 'N/A')}</span>
    </div>
    <div class="flex items-center">
        <span class="text-gray-600">📁 Type:</span>
        <span class="ml-2 font-medium">{json_response.get('project_type', 'N/A')}</span>
    </div>
    <div class="flex items-center">
        <span class="text-gray-600">👤 Author:</span>
        <span class="ml-2 font-medium">{json_response.get('project_author', 'N/A')}</span>
    </div>
    <div class="flex items-center">
        <span class="text-gray-600">💻 Platform:</span>
        <span class="ml-2 font-medium">{json_response.get('platform', 'N/A')}</span>
    </div>
</div>

<div class="mb-6">
    <h2 class="text-xl font-semibold text-gray-800 mb-2">📝 Description</h2>
    <p class="text-gray-700">{json_response.get('project_description', 'N/A')}</p>
</div>

<div>
    <h2 class="text-xl font-semibold text-gray-800 mb-4">📋 Tasks</h2>
    <div class="border-b border-gray-300 mb-4"></div>
    <div class="space-y-4">'''

                # Add tasks
                for idx, task in enumerate(json_response.get('tasks', []), 1):
                    formatted_info += f'''
<div class="bg-gray-50 p-4 rounded-lg">
    <h3 class="text-lg font-medium text-gray-800 mb-2">🔹 Task {idx}: {task.get("task_type","N/A")}</h3>
    <div class="pl-4">
        <h4 class="text-gray-700 font-medium mb-2">Parameters:</h4>
        <ul class="list-disc pl-6">'''
                    
                    for param_key, param_value in task.items():
                        if param_key != "task_type":
                            formatted_info += f'''
<li class="text-gray-600"><span class="font-medium">{param_key}:</span> {param_value}</li>'''
                    
                    formatted_info += '''
        </ul>
    </div>
</div>'''

                # Close all divs
                formatted_info += '''
            </div>
        </div>
    </div>
</div>
'''


                self.add_chunk_to_message_content(formatted_info)
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
        task_type = task.get("task_type")
        if task_type == "execute_command":
            command = task.get("command")
            if command:
                self.step_start(f"Executing command: {command}")
                try:
                    result = self.terminal.execute_command(command)
                    
                    # Show output in markdown format
                    output_md = f"""```
Command: {command}
Output:
{result["stdout"] if result and result["stdout"]!="" else "success"}
```"""
                    self.add_chunk_to_message_content(output_md)
                    
                    # Ask AI to analyze the output
                    analysis_prompt = f"""
Analyze the following command execution results and determine next steps:
Command: {command}
Output: {result["stdout"] if result and result["stdout"]!="" else "success"}
Error output: {result["stderr"] if result["stderr"] and result["stderr"]!="" else "no errors detected"}

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
                    analysis_response = self.generate_code(analysis_prompt, callback=self.sink)
                    try:
                        analysis = json.loads(analysis_response)
                        if analysis.get("status") == "success":
                            self.add_chunk_to_message_content("Task completed successfully: " + analysis.get("message", "")+"\n")
                            return True
                        else:
                            self.step(f"Task failed: {analysis.get('message', 'Unknown error')}")
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
                    self.add_chunk_to_message_content(error_md)
                    return False

        elif task_type == "create_file":
            file_name = task.get("file_name")
            if file_name:
                self.step_start(f"Generating content for {file_name}")
                if "files_information" not in  context_memory:
                    context_memory["files_information"]={}
                code = self.generate_file_content(context_memory, file_name, self.project_structure)
                current_dir = self.terminal.get_current_directory()
                file_path:Path = Path(current_dir) / file_name
                file_path.parent.mkdir(parents=True, exist_ok=True)

                # context_memory["files_information"][file_name]= self.generate(f"list all elements of this file in a textual simplified manner:\n{code}", callback=self.sink)
                
                try:
                    self.terminal.create_file(file_path, code)
                    self.add_chunk_to_message_content(f"Content of file: {file_path} written successfuly\n")
                    return True
                except Exception as e:
                    self.step_start(f"Error writing file: {str(e)}")
                    return False

        elif task_type == "run_application":
            command = task.get("command")
            if command:
                self.step_start(f"Running application with command: {command}")
                try:
                    # For Windows
                    result = self.terminal.execute_command(command)
                    # Show output in markdown format
                    output_md = f"""```
Application: {command}
Output:
{result}
```"""
                    self.add_chunk_to_message_content(output_md)
                    
                    # Ask AI to analyze the application output
                    analysis_prompt = f"""
Analyze the following application execution results:
Command: {command}
Output: {result["stdout"] if result["stdout"] and result["stdout"]!="" else "success"}
Error output: {result["stderr"] if result["stderr"] and result["stderr"]!="" else "no errors detected"}

Respond with a JSON containing:
1. status: "success" or "error"
2. message: Description of application behavior
"""
                    self.add_chunk_to_message_content("\n")
                    analysis_response = self.generate_code(analysis_prompt, callback=self.sink)
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
                    self.add_chunk_to_message_content(error_md)
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
            n=0
            while n<self.personality_config.max_retries:
                success = self.execute_task(self.project_details, task)
                if not success:
                    self.add_chunk_to_message_content(f"Failed to execute task: {task['task_type']}")
                    self.step(f"Failed to execute task: {task['task_type']}")
                    self.step(f"Thinking...")
                    n += 1
                else:
                    n=6

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

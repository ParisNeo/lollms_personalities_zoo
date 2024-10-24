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
import uuid

class InteractiveTerminal:
    def __init__(self, working_directory=None):
        """Initialize the interactive terminal handler"""
        self.working_directory= working_directory
        self.system = platform.system().lower()
        self.process = None
        self.terminal_id = str(uuid.uuid4())[:8]  # Unique identifier for this terminal session
        self.temp_dir = Path(tempfile.gettempdir()) / f"terminal_{self.terminal_id}"
        self.temp_dir.mkdir(exist_ok=True)
        
    def open_terminal(self, working_directory=None):
        """
        Opens a new interactive terminal window in the specified working directory
        
        Args:
            working_directory (str or Path, optional): The directory to start in
        """
        if working_directory:
            working_directory = Path(working_directory)
            if not working_directory.exists():
                raise FileNotFoundError(f"Directory {working_directory} does not exist")
        elif self.working_directory:
            working_directory = self.working_directory
        else:    
            working_directory = Path.cwd()

        try:
            if self.system == 'windows':
                # Create a starter batch file that will keep the terminal open
                starter_script = self.temp_dir / "terminal_starter.bat"
                with open(starter_script, 'w') as f:
                    f.write('@echo off\n')
                    f.write(f'cd /d "{working_directory}"\n')
                    f.write('echo Terminal ready!\n')
                    f.write('echo Alls commands will be executed here. Please wait\n')
                    f.write('cd\n')
                    f.write(':loop\n')
                    # Check for command file
                    f.write(f'if exist "{self.temp_dir}\\command.bat" (\n')
                    f.write(f'  call "{self.temp_dir}\\command.bat"\n')
                    f.write('ping -n 2 127.0.0.1 > nul\n')
                    f.write(f'  del "{self.temp_dir}\\command.bat"\n')
                    f.write(')\n')
                    f.write('ping -n 2 127.0.0.1 > nul\n')
                    f.write('goto loop\n')

                # Start the terminal with our starter script
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
                self.process = subprocess.Popen(
                    ['start', 'cmd.exe', '/k', str(starter_script)],
                    cwd=str(working_directory),
                    shell=True,
                    startupinfo=startupinfo
                )

            else:
                # Create a starter shell script
                starter_script = self.temp_dir / "terminal_starter.sh"
                with open(starter_script, 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write(f'cd "{working_directory}"\n')
                    f.write('echo "Terminal ready!"\n')
                    f.write('while true; do\n')
                    f.write(f'  if [ -f "{self.temp_dir}/command.sh" ]; then\n')
                    f.write(f'    source "{self.temp_dir}/command.sh"\n')
                    f.write(f'    rm "{self.temp_dir}/command.sh"\n')
                    f.write('  fi\n')
                    f.write('  sleep 1\n')
                    f.write('done\n')

                # Make the script executable
                os.chmod(starter_script, 0o755)

                # Find available terminal emulator
                terminal_cmd = None
                if os.system('which gnome-terminal') == 0:
                    terminal_cmd = ['gnome-terminal', '--']
                elif os.system('which xterm') == 0:
                    terminal_cmd = ['xterm', '-e']
                elif os.system('which konsole') == 0:
                    terminal_cmd = ['konsole', '-e']
                
                if terminal_cmd is None:
                    raise RuntimeError("No suitable terminal emulator found")

                self.process = subprocess.Popen(
                    terminal_cmd + [str(starter_script)],
                    cwd=str(working_directory)
                )

            # Wait a bit for terminal to initialize
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"Error opening terminal: {e}")
            return False

    def execute_command(self, command, wait_for_output=True, timeout=30):
        """
        Executes a command in the terminal and optionally waits for output
        
        Args:
            command (str): The command to execute
            wait_for_output (bool): Whether to wait for and return command output
            timeout (int): Maximum time to wait for output in seconds
            
        Returns:
            dict: Contains 'stdout', 'stderr', and 'success' if wait_for_output is True
        """
        if not self.process:
            raise RuntimeError("Terminal is not opened. Call open_terminal() first.")

        try:
            # Create unique output files for this command
            stdout_file = self.temp_dir / f"stdout_{time.time()}.txt"
            stderr_file = self.temp_dir / f"stderr_{time.time()}.txt"

            if self.system == 'windows':
                command_file = self.temp_dir / "command.bat"
                with open(command_file, 'w') as f:
                    f.write('@echo off\n')
                    f.write(f'{command} > "{stdout_file}" 2> "{stderr_file}"\n')
            else:
                command_file = self.temp_dir / "command.sh"
                with open(command_file, 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write(f'{command} > "{stdout_file}" 2> "{stderr_file}"\n')

            if not wait_for_output:
                return {'success': True}

            # Wait for command to complete and output files to be created
            start_time = time.time()
            while not stdout_file.exists() or not stderr_file.exists():
                if time.time() - start_time > timeout:
                    return {
                        'stdout': '',
                        'stderr': 'Command execution timed out',
                        'success': False
                    }
                time.sleep(0.1)

            # Read the output
            time.sleep(0.1)  # Give a small delay to ensure writing is complete
            try:
                with open(stdout_file, 'r') as f:
                    stdout = f.read()
                with open(stderr_file, 'r') as f:
                    stderr = f.read()
            except Exception as e:
                return {
                    'stdout': '',
                    'stderr': f'Error reading output: {str(e)}',
                    'success': False
                }

            # Cleanup
            try:
                stdout_file.unlink()
                stderr_file.unlink()
            except:
                pass

            return {
                'stdout': stdout,
                'stderr': stderr,
                'success': True
            }

        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    def close(self):
        """Closes the terminal session and cleans up temporary files"""
        if self.process:
            try:
                if self.system == 'windows':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)], 
                                 stderr=subprocess.DEVNULL)
                else:
                    self.process.terminate()
            except:
                pass
            finally:
                self.process = None

        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def __enter__(self):
        self.open_terminal()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
class Processor(APScript):
    def __init__(self, personality: AIPersonality, callback: Callable = None) -> None:
        personality_config_template = ConfigTemplate([
            {"name": "work_folder", "type": "str", "value":"", "help": "The working directory"},
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
        self.terminal = InteractiveTerminal(self.work_folder)
        self.terminal.open_terminal()
        if not self.project_details:
            self.step_start("Failed to parse the instruction. Please try again with a clearer instruction.")
            return
        self.add_chunk_to_message_content("\n")

        # Execute the project tasks
        self.execute_project()
        self.terminal.close()

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
        task_type = task.get("task")
        parameters = task.get("parameters", {})

        if task_type == "execute_command":
            command = parameters.get("command")
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
            file_name = parameters.get("file_name")
            if file_name:
                self.step_start(f"Generating code for {file_name}")
                if "files_information" not in  context_memory:
                    context_memory["files_information"]={}
                code = self.generate_file_content(context_memory, file_name, self.project_structure)
                file_path = self.work_folder / file_name
                
                context_memory["files_information"][file_name]= self.generate(f"list all elements of this file in a textual simplified manner:\n{code}", callback=self.sink)
                
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
            success = self.execute_task(self.project_details, task)
            if not success:
                self.step(f"Failed to execute task: {task['task']}")
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

from ascii_colors import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, LoLLMsAction, LoLLMsActionParameters
from lollms.utilities import PackageManager
from lollms.types import MSG_OPERATION_TYPE
from typing import Callable, Any

from functools import partial
if PackageManager.check_package_installed("pyautogui"):
    import pyautogui
else:
    PackageManager.install_package("pyautogui")
    import pyautogui
from PIL import Image
import webbrowser
import subprocess
import time

# Helper functions
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
        
        self.callback = None
        # Example entry
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        # Supported types:
        # str, int, float, bool, list
        # options can be added using : "options":["option1","option2"...]        
        personality_config_template = ConfigTemplate(
            [
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
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        
    def install(self):
        super().install()
        
        # requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        # subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")        

    def help(self, prompt="", full_context=""):
        self.set_message_content(self.personality.help)
    
    def add_file(self, path, client, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, client, callback)

    def open_new_tab(self, url="https://www.example.com"):
        """
        Opens a new tab in the currently open web browser.
        
        :param url: The URL to open in the new tab. Defaults to "https://www.example.com" if not provided.
        """
        webbrowser.open_new_tab(url)

    # Move the mouse
    def move_mouse(self, x, y):
        screen_width, screen_height = pyautogui.size()
        try:
            target_x = int(x * screen_width)/100
            target_y = int(y * screen_height)/100
            pyautogui.moveTo(target_x, target_y)
        except:
            ASCIIColors.error(f"Couldn't locate mouse: x:{x},y:{y}")


    # type text
    def type_text(self, text, add_return=False):
        pyautogui.typewrite(text + "\n" if add_return else "")

    def mouse_click(self, button):
        if button == 'left':
            pyautogui.click()
        elif button == 'right':
            pyautogui.click(button='right')

    def move_mouse_and_click(self, x, y):
        screen_width, screen_height = pyautogui.size()
        try:
            target_x = int(x * screen_width)/100
            target_y = int(y * screen_height)/100
            pyautogui.moveTo(target_x, target_y)
            pyautogui.click()
        except:
            ASCIIColors.error(f"Couldn't locate mouse: x:{x},y:{y}")        

    def make_screenshot(self, file_name):
        self.save_screenshot(self.take_screenshot(), file_name)

    def take_screenshot(self):
        return pyautogui.screenshot()
    
    def save_screenshot(self, image, filename):
        image.save(filename)

    def run_app(name: str) -> None:
        try:
            subprocess.run(name)
        except FileNotFoundError:
            print(f"Application '{name}' not found.")


    def done(self):
        print("Congrats!")
        self.personality.success("Done")


    def analyze_screenshot_and_replan(self, prompt, previous_discussion_text,sc_path):
        self.step_start("Observing")
        img = self.take_screenshot()
        self.save_screenshot(img, sc_path)
        self.step_end("Observing")
        self.step_start("Planning operation")
        try:
            def replan(objective, context):
                prompt = self.system_custom_header("objective")+objective+self.separator_template+self.system_custom_header("context")+context+self.separator_template+self.system_custom_header("Instruction")+"In your plan, only issue actions starting from the current state and do not issue actions that are not needed given the current state. if this is not the last required operation, or issue a done action."
                self.analyze_screenshot_and_replan(prompt=prompt,previous_discussion_text=previous_discussion_text, sc_path=sc_path)
            self.add_chunk_to_message_content("\n")
            plan = self.plan(prompt, [sc_path],
                [
                LoLLMsAction(
                            "take_screenshot_and_plan_next",[LoLLMsActionParameters("objective", str, ""), LoLLMsActionParameters("context", str, "")],
                            replan,
                            "Takes a screen shot then replans the next operations. add this action to all plans except if the objective is reached."
                            ),
                LoLLMsAction(
                    "open_new_tab",
                    [
                        LoLLMsActionParameters("url", str, "https://www.example.com")
                    ],
                    self.open_new_tab,
                    "Opens a new tab in the currently open web browser with the specified URL. If no URL is provided, it opens a blank new tab."
                ),                      
                LoLLMsAction(
                            "move_mouse",[
                                            LoLLMsActionParameters("x", int, ""), 
                                            LoLLMsActionParameters("y", int, value="")
                                        ],
                            self.move_mouse,
                            "Move the mouse to the position x,y of the screen. x and y are values between 0 and 100"
                            ),
                LoLLMsAction(
                            "mouse_click",[
                                            LoLLMsActionParameters("button", str, options=["left","right"], value="left")
                                        ],
                            self.mouse_click,
                            "Click on the screen using left or right button of the mouse"),
                LoLLMsAction(
                            "click_at",[
                                            LoLLMsActionParameters("x", int, ""), 
                                            LoLLMsActionParameters("y", int, value="")
                                        ],
                            self.move_mouse_and_click,
                            "Move the mouse to the position x,y of the screen then left click. x and y are values between 0 and 100"
                            ),
                LoLLMsAction(
                            "type_text",[
                                            LoLLMsActionParameters("text", str, value=""),
                                            LoLLMsActionParameters("add_return", bool, value="True")
                                        ],
                            self.type_text,
                            "Typing text"),
                LoLLMsAction(
                            "done",[],
                            self.done,
                            "This triggers the end of the operation. It should be called when the objective is reached."
                            ),
                ],
                previous_discussion_text+self.separator_template,max_answer_length=1024)
            self.step_end("Planning operation")
            for action in plan:
                if action.name!="done":
                    action.run()
                else:
                    break
        except Exception as ex:
            trace_exception(ex)
            self.step_end("Planning operation", False)        

    from lollms.client_session import Client
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

        ASCIIColors.info("Generating")
        self.callback = callback
        output_path = self.personality.lollms_paths.personal_outputs_path/self.personality.name
        output_path.mkdir(exist_ok=True, parents=True)
        sc_path = output_path/"sc.png"
        self.analyze_screenshot_and_replan(prompt, previous_discussion_text, sc_path)

        # out = self.fast_gen_with_images(previous_discussion_text, [sc_path], show_progress=True)
        # self.set_message_content(out)
        return ""


from ascii_colors import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, LoLLMsAction, LoLLMsActionParameters
from lollms.utilities import PackageManager
from lollms.types import MSG_TYPE
from typing import Callable

from functools import partial
if PackageManager.check_package_installed("pyautogui"):
    import pyautogui
else:
    PackageManager.install_package("pyautogui")
    import pyautogui
from PIL import Image
import subprocess

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
        self.full(self.personality.help)
    
    def add_file(self, path, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, callback)

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
    def type_text(self, text):
        pyautogui.typewrite(text)

    def mouse_click(self, button):
        if button == 'left':
            pyautogui.click()
        elif button == 'right':
            pyautogui.click(button='right')

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
            plan = self.plan_with_images(prompt, [sc_path],
                [
                LoLLMsAction(
                            "take_screenshot_and_plan_next",[],
                            partial(self.analyze_screenshot_and_replan,prompt=prompt,previous_discussion_text=previous_discussion_text, sc_path=sc_path),
                            "Takes a screen shot then replans the next operations. Make sure to call this step if you have unsufficient information about the situation."
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
                            "type_text",[
                                            LoLLMsActionParameters("text", str, value="")
                                        ],
                            self.type_text,
                            "Typing text"),
                LoLLMsAction(
                            "done",[],
                            self.done,
                            "This triggers the end of the operation. It should be called when the objective is reached."
                            ),
                ],
                previous_discussion_text+"\n!@>obligation:Do not close the lollms tabin the browser.\n",max_answer_length=512)
            self.full("\n".join([p.description for p in plan]))
            self.step_end("Planning operation")
            for action in plan:
                if action.name!="done":
                    action.run()
                else:
                    break
        except Exception as ex:
            trace_exception(ex)
            self.step_end("Planning operation", False)        

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

        ASCIIColors.info("Generating")
        self.callback = callback
        output_path = self.personality.lollms_paths.personal_outputs_path/self.personality.name
        output_path.mkdir(exist_ok=True, parents=True)
        sc_path = output_path/"sc.png"
        self.analyze_screenshot_and_replan(prompt, previous_discussion_text, sc_path)

        # out = self.fast_gen_with_images(previous_discussion_text, [sc_path], show_progress=True)
        # self.full(out)
        return ""


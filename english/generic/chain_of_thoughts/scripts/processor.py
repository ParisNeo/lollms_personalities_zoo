from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import subprocess
from pathlib import Path
import os
import sys
sd_folder = Path(__file__).resolve().parent.parent / "sd"
sys.path.append(str(sd_folder))
import sys
import yaml
import random
import re

def find_matching_number(numbers, text):
    for index, number in enumerate(numbers):
        number_str = str(number)
        pattern = r"\b" + number_str + r"\b"  # Match the whole word
        match = re.search(pattern, text)
        if match:
            return number, index
    return None, None  # No matching number found

class Processor(APScript):
    """
    A class that processes model inputs and outputs.

    Inherits from APScript.
    """

    def __init__(
                 self, 
                 personality: AIPersonality
                ) -> None:
        

        personality_config_template = ConfigTemplate([
                {"name":"max_thought_size","type":"int","value":512, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_judgement_size","type":"int","value":512, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"nb_ideas","type":"int","value":3, "min":2, "max":100},
                
            ])
        personality_config = BaseConfig.from_template(personality_config_template)
        personality_config = TypedConfig(
            personality_config_template,
            personality_config
        )
        super().__init__(
                            personality,
                            personality_config
                        )
        
    def install(self):
        super().install()
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])        
        ASCIIColors.success("Installed successfully")


    def remove_text_from_string(self, string, text_to_find):
        """
        Removes everything from the first occurrence of the specified text in the string (case-insensitive).

        Parameters:
        string (str): The original string.
        text_to_find (str): The text to find in the string.

        Returns:
        str: The updated string.
        """
        index = string.lower().find(text_to_find.lower())

        if index != -1:
            string = string[:index]

        return string
    
    def process(self, text, message_type:MSG_TYPE):
        bot_says = self.bot_says + text
        ASCIIColors.success(f"generated:{len(bot_says)} words", end='\r')
        antiprompt = self.personality.detect_antiprompt(bot_says)
        if antiprompt:
            self.bot_says = self.remove_text_from_string(bot_says,antiprompt)
            print("Detected hallucination")
            return False
        else:
            self.bot_says = bot_says
            if self.callback is not None:
                self.callback(text,MSG_TYPE.MSG_TYPE_CHUNK)            
            return True

    def generate(self, prompt, max_size, temperature = None, top_k = None, top_p=None, repeat_penalty=None ):
        self.bot_says = ""
        ASCIIColors.info("Text generation started: Warming up")
        return self.personality.model.generate(
                                prompt, 
                                max_size, 
                                self.process,
                                temperature=self.personality.model_temperature if temperature is None else temperature,
                                top_k=self.personality.model_top_k if top_k is None else top_k,
                                top_p=self.personality.model_top_p if top_p is None else top_p,
                                repeat_penalty=self.personality.model_repeat_penalty if repeat_penalty is None else repeat_penalty,
                                ).strip()    
        

    def run_workflow(self, prompt, previous_discussion_text="", callback=None):
        """
        Runs the workflow for processing the model input and output.

        This method should be called to execute the processing workflow.

        Args:
            generate_fn (function): A function that generates model output based on the input prompt.
                The function should take a single argument (prompt) and return the generated text.
            prompt (str): The input prompt for the model.
            previous_discussion_text (str, optional): The text of the previous discussion. Default is an empty string.
            callback a callback function that gets called each time a new token is received
        Returns:
            None
        """
        bot_says = ""
        self.callback = callback
        # 1 first ask the model to formulate a query
        final_ideas = []
        summary_prompt = ""
        for j in range(self.personality_config.nb_ideas):
            if callback:
                callback("Building idea {j+1}", MSG_TYPE.MSG_TYPE_STEP_START)
            
            ASCIIColors.info(f"============= Starting level {j} of the chain =====================")
            ideas=[]
            print(f"\nIdea {j+1}")
            if len(final_ideas)>0:
                final_ideas_text = "\n".join([f'Idea {n}: {i}' for n,i in enumerate(final_ideas)])
                idea_prompt = f""">instruction: Write the next idea. Please give a single idea. 
>prompt: {prompt}
>previous ideas: {final_ideas_text}
>idea:"""
            else:
                idea_prompt = f""">instruction:Write one idea. Do not give multiple ideas. 
>prompt: {prompt}
>idea:"""
            print(idea_prompt,end="",flush=True)
            idea = self.generate(idea_prompt, self.personality_config["max_thought_size"]).strip()
            ideas.append(idea)

            if callback:
                callback("Building idea {j+1}", MSG_TYPE.MSG_TYPE_STEP_END)

        summary_prompt += f""">Instructions:
Combine these ideas in a comprihensive and detailed essai that explains how to answer the user's question: {prompt}
"""
        for idea in ideas:
            summary_prompt += f">idea: {idea}\n"
        summary_prompt += ">essai:"
        print(summary_prompt)
        answer = self.generate(summary_prompt, self.personality_config["max_summary_size"])
        if callback:
            callback(answer, MSG_TYPE.MSG_TYPE_FULL)
        return answer



from lollms.config import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import subprocess
from pathlib import Path
import os
import sys
sd_folder = Path(__file__).resolve().parent.parent / "sd"
sys.path.append(str(sd_folder))
import urllib.parse
import urllib.request
import json
import time

from functools import partial
import sys
import yaml
import re
import random

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
                {"name":"max_thought_size","type":"int","value":50, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_judgement_size","type":"int","value":50, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_summary_size","type":"int","value":50, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"nb_samples_per_idea","type":"int","value":3, "min":2, "max":100},
                {"name":"nb_ideas","type":"int","value":3, "min":2, "max":100}
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


    def process(self, text, message_type:MSG_TYPE):
        bot_says = self.bot_says + text
        ASCIIColors.success(f"generated:{len(bot_says)} words", end='\r')
        if self.personality.detect_antiprompt(bot_says):
            print("Detected hallucination")
            return False
        else:
            self.bot_says = bot_says
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
        self.word_callback = callback
        self.bot_says = ""

        # 1 first ask the model to formulate a query
        final_ideas = []
        summary_prompt = ""
        for j in range(self.personality_config.nb_ideas):
            print(f"============= Starting level {j} of the tree =====================")
            if callback:
                callback(f"Starting level {j} of the tree", MSG_TYPE.MSG_TYPE_STEP_START)
            local_ideas=[]
            judgement_prompt = f">prompt: {prompt}\n"
            for i in range(self.personality_config.nb_samples_per_idea):
                print(f"\nIdea {i+1}")
                if len(final_ideas)>0:
                    final_ideas_text = "\n".join([f'Idea {n}:{i}' for n,i in enumerate(final_ideas)])
                    idea_prompt = f""">Instructions: Write the next idea. Please give a single idea. 
>prompt: {prompt}
>previous ideas: {final_ideas_text}
>idea:"""
                else:
                    idea_prompt = f""">Instruction: 
Write the next idea. Please give a single idea. 
>prompt:{prompt}
>idea:"""
                idea = self.generate(idea_prompt,self.personality_config.max_thought_size)
                if callback is not None:
                    callback(f"Idea: {idea}", MSG_TYPE.MSG_TYPE_STEP)

                local_ideas.append(idea.strip())
                judgement_prompt += f"\n>Idea {i}:{idea}\n"
            prompt_ids = ",".join([str(i) for i in range(self.personality_config["nb_samples_per_idea"])])
            judgement_prompt += f"""
>Instructions: Which idea seems the most approcpriate. Answer the question by giving the best idea number without explanations. What is the best idea number {prompt_ids}?
>judgement: The best idea is idea number"""
            print(judgement_prompt)
            self.bot_says = ""
            best_local_idea = self.generate(judgement_prompt,self.personality_config.max_judgement_size).strip()
            number, index = find_matching_number([i for i in range(self.personality_config["nb_samples_per_idea"])], best_local_idea)
            if index is not None:
                print(f"Chosen thoght n:{number}")
                final_ideas.append(local_ideas[number]) 
                if callback is not None:
                    callback(f"Best local idea:\n{best_local_idea}", MSG_TYPE.MSG_TYPE_STEP)
            else:
                print("Warning, the model made a wrong answer, taking random idea as the best")
                number = random.randint(0,self.personality_config["nb_samples_per_idea"])-1
                print(f"Chosen thoght n:{number}")
                final_ideas.append(local_ideas[number]) 
                if callback is not None:
                    callback(f"### Best local idea:\n{best_local_idea}", MSG_TYPE.MSG_TYPE_STEP)
            if callback:
                callback(f"Starting level {j} of the tree", MSG_TYPE.MSG_TYPE_STEP_END)

        if callback:
            callback(f"Starting final summary", MSG_TYPE.MSG_TYPE_STEP_START)
        summary_prompt += ">Instructions: Combine these ideas in a comprihensive essai. Give a detailed explanation.\n"
        for idea in final_ideas:
            summary_prompt += f">Idea: {idea}\n"
        summary_prompt += ">Ideas summary:"
        print(summary_prompt)
        best_local_idea = self.generate(summary_prompt, self.personality_config.max_summary_size)
        if callback:
            callback(f"Starting final summary", MSG_TYPE.MSG_TYPE_STEP_END)
        return best_local_idea



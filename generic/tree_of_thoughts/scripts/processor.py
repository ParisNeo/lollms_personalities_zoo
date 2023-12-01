from lollms.config import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import subprocess
from pathlib import Path
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
                 personality: AIPersonality,
                 callback = None,
                ) -> None:
        personality_config_template = ConfigTemplate([
                {"name":"max_thought_size","type":"int","value":512, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_judgement_size","type":"int","value":512, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_summary_size","type":"int","value":512, "min":10, "max":personality.model.config["ctx_size"]},
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
                            personality_config,
                            callback=callback
                        )
        
    def install(self):
        super().install()
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])        
        ASCIIColors.success("Installed successfully")


    def process(self, text, message_type:MSG_TYPE):
        if text is None:
            return
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
        self.callback = callback
        self.bot_says = ""
        # 1 first ask the model to formulate a query
        final_ideas = []
        summary_prompt = ""
        prev_number = -1
        layers = []
        selections = []
        output = ""
        for j in range(self.personality_config.nb_ideas):
            print(f"============= Starting level {j+1} of the tree =====================")
            output += f"\n-- level {j+1} ---\n"
            self.full(output)
            self.step_start(f"Processing Level {j+1} of the tree")
            local_ideas=[]
            judgement_prompt = f"!@>prompt: {prompt}\n"
            for i in range(self.personality_config.nb_samples_per_idea):
                self.step_start(f"Generating idea number {j+1}:{i+1}/{self.personality_config.nb_samples_per_idea}")
                print(f"\nIdea {i+1}")
                if len(final_ideas)>0:
                    final_ideas_text = "\n".join([f'Idea {n}:{i}' for n,i in enumerate(final_ideas)])
                    idea_prompt = f"""!@>instructions: given the following discussion and previous ideas, try to give another idea to solve the proposed problem or to enritch the discussion. 
!@>discussion:
{previous_discussion_text}
!@>previous ideas: {final_ideas_text}
!@>idea:"""
                else:
                    idea_prompt = f"""!@>instructions: given the following discussion, try to give an original idea to solve the proposed problem or to enritch the discussion. 
!@>discussion:
{previous_discussion_text}
!@>idea:"""
                idea = self.generate(idea_prompt,self.personality_config.max_thought_size)
                output += f"\n## Idea {i+1}:\n {idea}\n"
                self.full(output)
                local_ideas.append(idea.strip())
                judgement_prompt += f"\n!@>Idea {i}:{idea}\n"
                self.step_end(f"Generating idea number {j+1}:{i+1}/{self.personality_config.nb_samples_per_idea}")

            idea_id = self.multichoice_question(f"What is the most adequate idea to the context?\n",[f"{i} - {local_idea}" for local_idea in local_ideas],previous_discussion_text)
            if idea_id>=0 and idea_id<len(local_ideas):
                print(f"Chosen thought n:{idea_id}")
                final_ideas.append(local_ideas[idea_id])
                self.step(f"Best local idea:\n{local_ideas[idea_id]}")
            else:
                print("Warning, the model made a wrong answer, taking random idea as the best")
                idea_id = random.randint(0,self.personality_config["nb_samples_per_idea"])
                print(f"Chosen thought n:{idea_id+1}")
                if idea_id>=0 and idea_id<len(local_ideas):
                    final_ideas.append(local_ideas[idea_id]) 
                else:
                    final_ideas.append(local_ideas[0]) 
            output += f"Best idea : {idea_id}"
            self.full(output)
            layers.append(local_ideas)
            selections.append(idea_id)
            
            self.step_end(f"Processing Level {j+1} of the tree")

        self.step_start(f"Building final summary")
        summary_prompt += "!@>Instructions: Combine these ideas in a comprihensive essai. Give a detailed explanation.\n"
        for idea in final_ideas:
            summary_prompt += f">Idea: {idea}\n"
        summary_prompt += "!@>Ideas summary:"

        final_summary = self.generate(summary_prompt, self.personality_config.max_summary_size)

        ASCIIColors.success("Summary built successfully")
        self.step_end(f"Building final summary")
        output += f"## Final summary:\n{summary_prompt}"
        self.full(output)
        
        
        tree_full_output = {
            "tree_layers": layers,
            "selections":selections,
            "summary":final_summary
        }
        
        self.json("infos", tree_full_output)

        return final_summary



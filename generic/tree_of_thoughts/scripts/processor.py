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

import networkx as nx
import matplotlib.pyplot as plt


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
        
    def add_graph_level(self, G, ideas, prev_selected_id):
        # Get the nodes at the current level
        level_nodes = list(G.nodes)[-len(ideas):]

        for i, idea in enumerate(ideas):
            # Generate a new idea
            new_idea = f'Idea_{len(G.nodes) + 1}'

            # Add the new idea to the graph
            G.add_node(new_idea)

            if prev_selected_id>=0:
                # Connect the selected node at the previous level to the new idea
                G.add_edge(level_nodes[prev_selected_id], new_idea)

    def visualize_thought_graph(self, G):
        # Set the positions of the graph nodes
        pos = nx.spring_layout(G)

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue')

        # Highlight the selected node
        nx.draw_networkx_nodes(G, pos, node_color='red')

        # Draw edges with different styles for selected and unselected nodes
        edge_labels = {}
        for edge in G.edges:
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='red', arrowstyle='->', width=2)
            edge_labels[edge] = edge[1]

        # Add edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Display the graph
        plt.axis('off')
        plt.show()

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
        # Create an empty graph
        thought_graph = nx.DiGraph()

        # 1 first ask the model to formulate a query
        final_ideas = []
        summary_prompt = ""
        prev_number = -1
        layers = []
        selections = []
        for j in range(self.personality_config.nb_ideas):
            print(f"============= Starting level {j} of the tree =====================")
            self.step_start(f"Starting level {j} of the tree", callback)
            local_ideas=[]
            judgement_prompt = f"!@>prompt: {prompt}\n"
            for i in range(self.personality_config.nb_samples_per_idea):
                print(f"\nIdea {i+1}")
                if len(final_ideas)>0:
                    final_ideas_text = "\n".join([f'Idea {n}:{i}' for n,i in enumerate(final_ideas)])
                    idea_prompt = f""">Instructions: Write the next idea. Please give a single idea. 
>prompt: {prompt}
>previous ideas: {final_ideas_text}
>idea:"""
                else:
                    idea_prompt = f"""!@>Instruction: 
Write the next idea. Please give a single idea. 
!@>prompt:{prompt}
!@>idea:"""
                idea = self.generate(idea_prompt,self.personality_config.max_thought_size)
                self.full(f"Idea: {idea}: {idea}", callback)

                local_ideas.append(idea.strip())
                judgement_prompt += f"\n!@>Idea {i}:{idea}\n"
            prompt_ids = ",".join([str(i) for i in range(self.personality_config["nb_samples_per_idea"])])
            judgement_prompt += f"""
!@>Instructions: Which idea seems the most approcpriate. Answer the question by giving the best idea number without explanations. What is the best idea number {prompt_ids}?
!@>judgement: The best idea is idea number"""
            # print(judgement_prompt)
            self.bot_says = ""
            best_local_idea = self.generate(judgement_prompt,self.personality_config.max_judgement_size, temperature = 0.1, top_k=1).strip()
            number, index = find_matching_number([i for i in range(self.personality_config["nb_samples_per_idea"])], best_local_idea)
            if index is not None and number>=0 and number<len(local_ideas):
                print(f"Chosen thought n:{number}")
                final_ideas.append(local_ideas[number]) 
                
                if callback is not None:
                    callback(f"Best local idea:\n{best_local_idea}", MSG_TYPE.MSG_TYPE_STEP)
            else:
                print("Warning, the model made a wrong answer, taking random idea as the best")
                number = random.randint(0,self.personality_config["nb_samples_per_idea"])
                print(f"Chosen thought n:{number}")
                if number>=0 and number<len(local_ideas):
                    final_ideas.append(local_ideas[number]) 
                else:
                    final_ideas.append(local_ideas[0]) 
                if callback is not None:
                    callback(f"!@> Best local idea:\n{best_local_idea}", MSG_TYPE.MSG_TYPE_STEP)

            layers.append(local_ideas)
            selections.append(number)
            self.add_graph_level(thought_graph, local_ideas, prev_number)
            prev_number = number
            
            self.step_end(f"Starting level {j} of the tree", callback)

        self.visualize_thought_graph(thought_graph)
        self.step_start(f"Starting final summary", callback)
        summary_prompt += "!@>Instructions: Combine these ideas in a comprihensive essai. Give a detailed explanation.\n"
        for idea in final_ideas:
            summary_prompt += f">Idea: {idea}\n"
        summary_prompt += "!@>Ideas summary:"

        final_summary = self.generate(summary_prompt, self.personality_config.max_summary_size)

        ASCIIColors.success("Summary built successfully")
        self.step_end(f"Starting final summary", callback)
        
        self.full(final_summary, callback)
        
        
        tree_full_output = {
            "tree_layers": layers,
            "selections":selections,
            "summary":final_summary
        }
        
        self.json(tree_full_output, callback)

        return final_summary



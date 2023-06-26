from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality


from pathlib import Path
import subprocess
import re

def format_url_parameter(value:str):
    encoded_value = value.strip().replace("\"","")
    return encoded_value


class Processor(APScript):
    """
    A class that processes model inputs and outputs.

    Inherits from APScript.
    """

    def __init__(
                 self, 
                 personality: AIPersonality
                ) -> None:
        self.queries=[]
        self.formulations=[]
        self.summaries=[]
        self.word_callback = None
        self.generate_fn = None
        template = ConfigTemplate([
                {"name":"num_results","type":"int","value":5, "min":2, "max":100},
                {"name":"max_query_size","type":"int","value":50, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_summery_size","type":"int","value":256, "min":10, "max":personality.model.config["ctx_size"]},
            ])
        config = BaseConfig.from_template(template)
        personality_config = TypedConfig(
            template,
            config
        )
        super().__init__(
                            personality,
                            personality_config
                        )
        
        #Now try to import stuff to verify that installation succeeded
        import wikipedia
        
    def install(self):
        super().install()
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])        
        ASCIIColors.success("Installed successfully")

    def uninstall(self):
        super().uninstall()

     
    def wiki_search(self, query):
        """
        Perform an internet search using the provided query.

        Args:
            query (str): The search query.

        Returns:
            dict: The search result as a dictionary.
        """

        import wikipedia
        try:
            summary = wikipedia.summary(query)
            is_ambiguous = False
        except DisambiguationError as ex:
            summary = str(ex)
            is_ambiguous = True
            
        return summary, is_ambiguous

    def run_workflow(self, prompt, previous_discussion_text="", callback=None):
        """
        Runs the workflow for processing the model input and output.

        This method should be called to execute the processing workflow.

        Args:
            generate_fn (function): A function that generates model output based on the input prompt.
                The function should take a single argument (prompt) and return the generated text.
            prompt (str): The input prompt for the model.
            previous_discussion_text (str, optional): The text of the previous discussion. Default is an empty string.

        Returns:
            None
        """
        import wikipedia
        self.word_callback = callback
        if self.personality_config.craft_search_query:
            # 1 first ask the model to formulate a query
            search_formulation_prompt = f"""### Instructions:
Formulate a search query text out of the user prompt.
Keep all important information in the query and do not add unnecessary text.
Write a short query.
Do not explain the query.
## question:
{prompt}
### search query:
    """
            if callback is not None:
                callback("Crafting search query", MSG_TYPE.MSG_TYPE_STEP_START)
            search_query = format_url_parameter(self.generate(search_formulation_prompt, self.personality_config.max_query_size)).strip()
            if search_query=="":
                search_query=prompt
            if callback is not None:
                callback("Crafting search query", MSG_TYPE.MSG_TYPE_STEP_END)
        else:
            search_query = prompt
        search_result, is_ambiguous = self.wiki_search(search_query)
        if is_ambiguous:
            if callback:
                callback(search_result, MSG_TYPE.MSG_TYPE_FULL)
                return search_result
        else:
            prompt = f"""### Instructions:
Use Search engine results to answer user question by summerizing the results in a single coherant paragraph in form of a markdown text with sources citation links in the format [index](source).
Place the citation links in front of each relevant information.
Citation is mandatory.
### search results:
{search_result}
### question:
{prompt}
## answer:
"""
            print(prompt)
            output = self.generate(prompt, self.personality_config.max_summery_size)
            sources_text = "\n# Source :\n"
            page = wikipedia.page(search_query)
            sources_text += f"[source : {page.title}]({page.url})\n\n"

            output = output+sources_text
            if callback is not None:
                callback(output, MSG_TYPE.MSG_TYPE_FULL)

            return output




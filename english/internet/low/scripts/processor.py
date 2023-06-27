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
                {"name":"craft_search_query","type":"bool","value":False,"help":"By default, your question is directly sent to wikipedia search engine. If you activate this, LOW will craft a more optimized version of your question and use that instead."},
                {"name":"synthesize","type":"bool","value":True,"help":"By default, LOW will preprocess the outputs before answering you. If you deactivate this, you will simply get the wikiopedia output."},
                {"name":"num_sentences","type":"int","value":10, "min":2, "max":100,"help":"Number of sentences to recover from wikipedia to be used by LOW to answer you."},
                {"name":"max_nb_images","type":"int","value":10, "min":1, "max":100,"help":"Sometimes, LOW can show you images extracted from wikipedia."},
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

    def data_driven_qa(self, 
                            data, 
                            question, 
                            answer_motivational_text="",
                            max_size=128,
                            instruction = None, 
                            temperature = None, 
                            top_k = None, 
                            top_p=None, 
                            repeat_penalty=None 
                        ):
        if instruction is not None:
            instruction = '### Instructions:\n'+instruction
            search_formulation_prompt = f"""{instruction}
> data:
{data}
> question:
{question}
> answer:
{answer_motivational_text}"""
        else:
            search_formulation_prompt = f"""> data:
{data}
> question:
{question}
> answer:
{answer_motivational_text}"""
            
        if self.word_callback is not None:
            self.word_callback(f"Asking AI: "+question, MSG_TYPE.MSG_TYPE_STEP_START)
        answer = format_url_parameter(
            self.generate(
                        search_formulation_prompt,
                        max_size,
                        temperature = temperature, top_k = top_k, top_p=top_p, repeat_penalty=repeat_penalty 
                        )
            ).strip()
        if self.word_callback is not None:
            self.word_callback(f"Asking AI: "+question, MSG_TYPE.MSG_TYPE_STEP_END)
        return answer
    def wiki_search(self, query, nb_sentences=3):
        """
        Perform an internet search using the provided query.

        Args:
            query (str): The search query.

        Returns:
            dict: The search result as a dictionary.
        """

        import wikipedia
        try:
            summary = wikipedia.summary(query, sentences=nb_sentences)
            is_ambiguous = False
        except wikipedia.DisambiguationError as ex:
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
        try:
            import wikipedia
            self.word_callback = callback
            if self.personality_config.craft_search_query:
                # 1 first ask the model to formulate a query
                search_formulation_prompt = f"""### Instructions>
Formulate a wikipedia search query text out of the user prompt.
Keep all important information in the query and do not add unnecessary text.
The query is in the form of keywords.
Do not explain the query.
## question>
{prompt}
### query>
keywords query:"""
                if callback is not None:
                    callback("Crafting search query", MSG_TYPE.MSG_TYPE_STEP_START)
                search_query = self.generate(search_formulation_prompt, self.personality_config.max_query_size).strip()
                if search_query=="":
                    search_query=prompt
                if callback is not None:
                    callback("Crafting search query", MSG_TYPE.MSG_TYPE_STEP_END)
            else:
                search_query = prompt
            results = wikipedia.search(search_query)
            entry = self.data_driven_qa(
                "\n".join([f"{i}: {res}" for i, res in enumerate(results)]),
                f"{prompt}",
                "The most relevant entry to aswer the question among the proposed ones is entry number:",
                max_size = 4,
                repeat_penalty=0.5)
            try:
                entry=results[int(entry.strip())]
            except:
                try:
                    entry = entry.strip().split("\n")[0]
                    entry=results[int(entry)]
                except:
                    try:
                        entry = entry.strip().split(":")[0]
                        entry=results[int(entry)]
                    except:
                        ASCIIColors.warning(f"couldn't figure out which entry is best. Defaulting to first one")
                        entry=results[0]

            if callback is not None:
                callback(f"Entry: {entry}", MSG_TYPE.MSG_TYPE_STEP_START)
                callback(f"Entry: {entry}", MSG_TYPE.MSG_TYPE_STEP_END)
            page = wikipedia.page(entry)
            search_result = page.summary
            images = [img for img in page.images if img.split('.')[-1].lower() in ["gif","png","jpg","webp","svg"]]
            # cap images
            images = images[:self.personality_config.max_nb_images]
            images = '\n'.join([f"![image {i}]({im})" for i,im in enumerate(images)])
            prompt = f"""{previous_discussion_text}
Use this data and images to answer the user
> wikipedia:
{search_result}
> images:
{images}
> answer:"""
            if callback is not None:
                callback("Generating response", MSG_TYPE.MSG_TYPE_STEP_START)
            output = self.generate(prompt, self.personality_config.max_summery_size)
            sources_text = "\n-----------\n"
            sources_text += "# Source :\n"
            sources_text += f"[{page.title}]({page.url})\n\n"
            if callback is not None:
                callback("Generating response", MSG_TYPE.MSG_TYPE_STEP_END)

            output = output+sources_text
            if callback is not None:
                callback(output, MSG_TYPE.MSG_TYPE_FULL)

            return output
        except Exception as ex:
            output = f"Exception occured while running workflow: {ex}"
            if callback is not None:
                callback(output, MSG_TYPE.MSG_TYPE_EXCEPTION)
            return output   




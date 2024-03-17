from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from lollms.types import MSG_TYPE
from lollms.internet import internet_search
from typing import Callable

from safe_store.generic_data_loader import GenericDataLoader
from safe_store.document_decomposer import DocumentDecomposer
import subprocess
from pathlib import Path

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
                {"name":"search_query","type":"text","value":"", "help":"Here you can put custom search query to be used."},
                {"name":"zip_mode","type":"str","value":"hierarchical","options":["hierarchical","one_shot"], "help":"algorithm"},
                {"name":"zip_size","type":"int","value":1024, "help":"the maximum size of the summary in tokens"},
                {"name":"buttons_to_press","type":"str","value":"'accept'", "help":"Buttons to be pressed in the pages you want to load."},
                {"name":"output_path","type":"str","value":"", "help":"The path to a folder where to put the summary file."},
                {"name":"contextual_zipping_text","type":"text","value":"", "help":"Here you can specify elements of the document that you want the AI to keep or to search for. This garantees that if found, those elements will not be filtered out which results in a more intelligent contextual based summary."},
                {"name":"keep_same_language","type":"bool","value":True, "help":"Force the algorithm to keep the same language and not translate the document to english"},
                {"name":"translate_to","type":"str","value":"", "help":"Force the algorithm to summarize the document in a specific language. If none is provided then it won't do any translation"},
                {"name":"preserve_document_title","type":"bool","value":False, "help":"Force the algorithm to preserve the document title as an important information"},
                {"name":"preserve_authors_name","type":"bool","value":False, "help":"Force the algorithm to preserve the authors names as an important information"},
                {"name":"preserve_results","type":"bool","value":True, "help":"Force the algorithm to preserve the document results the authors names as an important information"},
                {"name":"maximum_compression","type":"bool","value":False, "help":"Force the algorithm to compress the document as much as possible. Useful for what is this document talking about kind of summary"},
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
                                        "start_scraping":self.start_scraping
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        self.cv = None
        self.position = None

    def install(self):
        super().install()
        
        # requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        # subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")        

    def help(self, prompt="", full_context=""):
        self.personality.InfoMessage(self.personality.help)
    
    def add_file(self, path, client, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, client, callback)

    def save_text(self, text, path:Path):
        with open(path,"w", encoding="utf8") as f:
            f.write(text)
            
    def search_and_zip(self, query,  output =""):
        pages = internet_search(query, self.personality.config, buttons_to_press=self.personality_config.buttons_to_press)
        processed_pages = ""
        for page in pages:
            tk = self.personality.model.tokenize(document_text)
            self.step_start(f"summerizing {page['title']}")
            if len(tk)<int(self.personality_config.zip_size):
                    document_text = self.summerize([document_text],"Summerize this document chunk and do not add any comments after the summary.\nOnly extract the information from the provided chunk.\nDo not invent anything outside the provided text.","document chunk")
            else:
                depth=0
                while len(tk)>int(self.personality_config.zip_size):
                    self.step_start(f"Comprerssing.. [depth {depth}]")
                    chunk_size = int(self.personality.config.ctx_size*0.6)
                    document_chunks = DocumentDecomposer.decompose_document(document_text, chunk_size, 0, self.personality.model.tokenize, self.personality.model.detokenize, True)
                    document_text = self.summerize(document_chunks,"\n".join([
                            f"Summerize the document chunk and do not add any comments after the summary.",
                            "The summary should contain exclusively information from the document chunk.",
                            "Do not provide opinions nor extra information that is not in the document chunk",
                            f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                            f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                            f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                            f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                            f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                            f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                            f"{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
                        ]),
                        "Document chunk"
                        )
                    tk = self.personality.model.tokenize(document_text)
                    self.step_end(f"Comprerssing.. [depth {depth}]")
                    self.full(output+f"\n\n## Summerized chunk text:\n{document_text}")
                    depth += 1
            self.step_start(f"Last composition")
            document_text = self.summerize(document_chunks,"\n".join([
                    f"Rewrite this document in a better way while respecting the following guidelines:",
                    f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                    f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                    f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                    f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                    f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                    f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                    f"{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
                ]),
                "Document chunk"
                )

            self.step_end(f"Last composition")
            self.step_end(f"summerizing {page['title']}")
            processed_pages += f"{page['title']}\n{document_text}"

        document_text = self.summerize(processed_pages,"\n".join([
                f"Summerize the document chunk and do not add any comments after the summary.",
                "The summary should contain exclusively information from the document chunk.",
                "Do not provide opinions nor extra information that is not in the document chunk",
                f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                f"{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
            ]),
            "Document chunk"
            )

        self.step_start(f"Last composition")
        document_text = self.summerize(document_chunks,"\n".join([
                f"Rewrite this document in a better way while respecting the following guidelines:",
                f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                f"{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
            ]),
            "Document chunk"
            )

        self.step_end(f"Last composition")

        if self.personality_config.output_path:
            self.save_text(document_text, Path(self.personality_config.output_path)/(page['title']+"_summary.txt"))
        return document_text, output
                    
        

    def start_scraping(self, prompt="", full_context=""):
        self.new_message("")
        if self.personality_config.search_query!="":
            self.search_and_zip(self.personality_config.search_query)
        else:
            self.info("Please put a search query in the search query setting of this personality.")

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

        self.callback = callback
        if len(self.personality.text_files)>0:
            self.step_start("Understanding request")
            if self.yes_no("Is the user asking for doing internet search about a topic?", previous_discussion_text):
                self.step_end("Understanding request")
                self.personality.step_start("Crafting internet search query")
                query = self.personality.fast_gen(f"!@>discussion:\n{previous_discussion_text}\n!@>system: Read the discussion and craft a web search query suited to recover needed information to reply to last {self.config.user_name} message.\nDo not answer the prompt. Do not add explanations.\n!@>current date: {datetime.now()}!@>websearch query: ", max_generation_size=256, show_progress=True, callback=self.personality.sink)
                self.personality.step_end("Crafting internet search query")
                self.search_and_zip(query)
            else:
                self.step_end("Understanding request")
                self.fast_gen(previous_discussion_text, callback=self.callback)
        else:
            self.fast_gen(previous_discussion_text, callback=self.callback)
        return ""



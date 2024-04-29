from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
from lollms.types import MSG_TYPE
from typing import Callable

from safe_store.generic_data_loader import GenericDataLoader
from safe_store.document_decomposer import DocumentDecomposer
import subprocess
from pathlib import Path
from lollms.client_session import Client
from lollms.types import SUMMARY_MODE
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
                {"name":"zip_mode","type":"str","value":"sequencial","options":["sequencial", "hierarchical"], "help":"algorithm"},
                {"name":"zip_size","type":"int","value":512, "help":"the maximum size of the summary in tokens"},
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
                                        "start_zipping":self.start_zipping
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
            
    def zip_document(self, document_path:Path,  output =""):
        document_text = GenericDataLoader.read_file(document_path)
        tk = self.personality.model.tokenize(document_text)
        self.step_start(f"summerizing {document_path.stem}")
        if len(tk)<int(self.personality_config.zip_size):
                document_text = self.summerize_text(document_text,"Summerize this document chunk and do not add any comments after the summary.\nOnly extract the information from the provided chunk.\nDo not invent anything outside the provided text.","document chunk")
        else:
            depth=0
            while len(tk)>int(self.personality_config.zip_size):
                self.step_start(f"Comprerssing.. [depth {depth}]")
                chunk_size = int(self.personality.config.ctx_size*0.6)
                document_chunks = DocumentDecomposer.decompose_document(document_text, chunk_size, 0, self.personality.model.tokenize, self.personality.model.detokenize, True)
                document_text = self.summerize_chunks(document_chunks,"\n".join([
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
                    "Document chunk",
                    summary_mode=SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL if self.personality_config.zip_mode=="sequencial" else SUMMARY_MODE.SUMMARY_MODE_HIERARCHICAL,
                    callback=self.sink
                    )
                tk = self.personality.model.tokenize(document_text)
                self.step_end(f"Comprerssing.. [depth {depth}]")
                self.full(output+f"\n\n## Summerized chunk text:\n{document_text}")
                depth += 1
        self.step_start(f"Last composition")
        document_text = self.summerize_chunks([document_text],"\n".join([
                f"Rewrite this document in a better way while respecting the following guidelines:",
                f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                f"{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
            ]),
            "Document chunk",
            callback=self.sink
            )

        self.step_end(f"Last composition")
        self.step_end(f"summerizing {document_path.stem}")
        if self.personality_config.output_path:
            self.save_text(document_text, Path(self.personality_config.output_path)/(document_path.stem+"_summary.txt"))
        return document_text, output
                    
        

    def start_zipping(self, prompt="", full_context="", client:Client=None):
        self.new_message("")
        if len(self.personality.text_files)==0:

            self.full("\n".join([
                "Hey there! 🌟 It looks like you're itching for a bit of that magic summary action.",
                "📜✨ Well, I'm all revved up and ready to dive into the world of summarization, but here's the kicker—I can't exactly pull off my magic tricks without a hat... or in this case, without any documents. 🎩🚫 So, how about we make this a team effort? 🤝 Go ahead and press that shiny send documents button 📤, pick out some documents for me to sink my teeth into 📄🔍, and then let's reconvene. Summon me back into the arena 📣, and I promise, I'll zip through those documents faster than you can say \"LoLLMs, Lord of Large Language Multimodal Systems,\" extracting the juicy bits and serving you exactly what you need. 🚀 Looking forward to our next encounter.",
                "See ya! 👋"  
            ])
            )
            return
        for file in self.personality.text_files:
            output=""
            file = Path(file)
            summary, output = self.zip_document(file, output)
            output +=f"\n## Summary of {file.stem}\n{summary}"
            self.full(output)


    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, context_details:dict=None, client:Client=None):
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
            if self.yes_no("Is the user asking for summarizing the document?", previous_discussion_text):
                self.step_end("Understanding request")
                self.start_zipping()
            else:
                self.step_end("Understanding request")
                self.fast_gen(previous_discussion_text, callback=self.callback)
        else:
            self.fast_gen(previous_discussion_text, callback=self.callback)
        return ""



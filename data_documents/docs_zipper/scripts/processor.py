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
                {"name":"zip_size","type":"int","value":1024, "help":"the maximum size of the summary in tokens"},
                {"name":"data_folder","type":"str","value":"", "help":"The path to a folder where to get the input files."},
                {"name":"output_path","type":"str","value":"", "help":"The path to a folder where to put the summary file."},
                {"name":"contextual_zipping_text","type":"text","value":"", "help":"Here you can specify elements of the document that you want the AI to keep or to search for for the summary of each document. This garantees that if found, those elements will not be filtered out which results in a more intelligent contextual based summary."},
                {"name":"add_summary_formatting","type":"bool","value":True, "help":"After summarizing a document, you can reformat the final output in a certain way. For example, you can ask the AI to put the output in a certain templete, or to build a latex code etc."},
                {"name":"summary_formatting_text","type":"text","value":"", "help":"Here you can specify elements of the document that you want the AI to keep or to search for for the summary of each document. This garantees that if found, those elements will not be filtered out which results in a more intelligent contextual based summary."},
                {"name":"global_contextual_zipping_text","type":"text","value":"", "help":"Here you can specify elements of the document that you want the AI to keep or to search for the final fusion report. This garantees that if found, those elements will not be filtered out which results in a more intelligent contextual based summary."},
                {"name":"add_global_summary_formatting","type":"bool","value":True, "help":"After summarizing a document, you can reformat the final output in a certain way. For example, you can ask the AI to put the output in a certain templete, or to build a latex code etc."},
                {"name":"global_summary_formatting_text","type":"text","value":"", "help":"Here you can specify elements of the document that you want the AI to keep or to search for for the summary of each document. This garantees that if found, those elements will not be filtered out which results in a more intelligent contextual based summary."},
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

    def zip_text(
                    self, 
                    document_text:str, 
                    instruction=f"Summerize the document chunk in a detailed comprehensive manner.", 
                    add_summary_formatting=True,
                    contextual_zipping_text="",
                    summary_formatting_text="",
                    translate_to=""
                    ):
        start_header_id_template    = self.config.start_header_id_template
        end_header_id_template      = self.config.end_header_id_template
        system_message_template     = self.config.system_message_template

        start_ai_header_id_template     = self.config.start_ai_header_id_template
        end_ai_header_id_template       = self.config.end_ai_header_id_template

        zip_prompt = f"{start_header_id_template}{system_message_template}{end_header_id_template}\n"
        zip_prompt+= instruction + "\n"
        zip_prompt+="Do not provide opinions nor extra information that is not in the document chunk\n"
        zip_prompt+="Keep the same language.\n" if self.personality_config.keep_same_language else ''
        zip_prompt+="Preserve the title of this document if provided.\n" if self.personality_config.preserve_document_title else ''
        zip_prompt+="Preserve author names of this document if provided.\n" if self.personality_config.preserve_authors_name else ''
        zip_prompt+="Preserve results if presented in the chunk and provide the numerical values if present.\n" if self.personality_config.preserve_results else ''
        zip_prompt+="Eliminate any useless information and make the summary as short as possible.\n" if self.personality_config.maximum_compression else ''
        zip_prompt+="Eliminate any useless information and make the summary as short as possible.\n" if self.personality_config.maximum_compression else ''
        zip_prompt+=f"Important information:{contextual_zipping_text}."+"\n" if contextual_zipping_text!='' else ''
        zip_prompt+=f"The summary should be written in "+ translate_to +"\n" if translate_to!='' else ''
        
        tk = self.personality.model.tokenize(document_text)
        if len(tk)>int(self.personality_config.zip_size):
            depth=0
            while len(tk)>int(self.personality_config.zip_size):
                if self.personality_config.zip_mode!="sequencial":
                    self.step_start(f"Comprerssing.. [depth {depth}]")
                chunk_size = int(self.personality.config.ctx_size*0.6)
                document_chunks = DocumentDecomposer.decompose_document(document_text, chunk_size, 0, self.personality.model.tokenize, self.personality.model.detokenize, True)
                document_text = self.summerize_chunks(document_chunks, 
                    zip_prompt,
                    "Document chunk",
                    summary_mode=SUMMARY_MODE.SUMMARY_MODE_SEQUENCIAL if self.personality_config.zip_mode=="sequencial" else SUMMARY_MODE.SUMMARY_MODE_HIERARCHICAL,
                    callback=self.sink
                    )
                tk = self.personality.model.tokenize(document_text)
                depth += 1
                if self.personality_config.zip_mode!="sequencial":
                    self.step_end(f"Comprerssing.. [depth {depth}]")
                else:
                    break
        if add_summary_formatting:
            last_composition_prompt  = f"{start_header_id_template}Summerized document text{end_header_id_template}\n"
            last_composition_prompt += f"{document_text}\n"
            last_composition_prompt += f"{start_header_id_template}{system_message_template}{end_header_id_template}\n"
            
            last_composition_prompt += "Do not provide opinions nor extra information that is not in the document chunk\n"
            last_composition_prompt += "Keep the same language.'}\n" if self.personality_config.keep_same_language else ''
            last_composition_prompt += "Preserve the title of this document if provided.\n'}" if self.personality_config.preserve_document_title else ''
            last_composition_prompt += "Preserve author names of this document if provided.\n'}" if self.personality_config.preserve_authors_name else ''
            last_composition_prompt += "Preserve results if presented in the chunk and provide the numerical values if present.\n'}" if self.personality_config.preserve_results else ''
            last_composition_prompt += "Eliminate any useless information and make the summary as short as possible.\n'}" if self.personality_config.maximum_compression else ''
            last_composition_prompt += "Eliminate any useless information and make the summary as short as possible.\n'}" if self.personality_config.maximum_compression else ''
            last_composition_prompt += f"Important information:{summary_formatting_text}."+"\n" if summary_formatting_text!='' else ''
            last_composition_prompt += "The summary should be written in "+translate_to +"\n" if translate_to!='' else ''
            last_composition_prompt += "Answer directly with the new enhanced document text with no extra comments.\n",
            last_composition_prompt += f"{start_ai_header_id_template}assistant{end_ai_header_id_template}"
            
            self.step_start(f"Last composition")
            document_text = self.fast_gen(last_composition_prompt, self.personality_config.zip_size,
                callback=self.sink
            )

            self.step_end(f"Last composition")

        return document_text
                    
        

    def start_zipping(self, prompt="", full_context="", client:Client=None):
        self.new_message("")
        if self.personality_config.data_folder!="":
            files = [f for f in Path(self.personality_config.data_folder).iterdir()]
        else:
            files = self.personality.text_files
        
        if len(files)==0:

            self.full("\n".join([
                "Hey there! 🌟 It looks like you're itching for a bit of that magic summary action.",
                "📜✨ Well, I'm all revved up and ready to dive into the world of summarization, but here's the kicker—I can't exactly pull off my magic tricks without a hat... or in this case, without any documents. 🎩🚫 So, how about we make this a team effort? 🤝 Go ahead and press that shiny send documents button 📤, pick out some documents for me to sink my teeth into 📄🔍, and then let's reconvene. Summon me back into the arena 📣, and I promise, I'll zip through those documents faster than you can say \"LoLLMs, Lord of Large Language Multimodal Systems,\" extracting the juicy bits and serving you exactly what you need. 🚀 Looking forward to our next encounter.",
                "See ya! 👋"  
            ])
            )
            return
            
        all_summaries=""
        self.step(f"summary mode : {self.personality_config.zip_mode}")
        for file in files:
            if file.suffix.lower() in [".pdf", ".docx", ".pptx"]:
                document_path = Path(file)
                self.step_start(f"summerizing {document_path.stem}")
                document_text = GenericDataLoader.read_file(document_path)
                summary = self.zip_text(
                                            document_text,
                                            add_summary_formatting=self.personality_config.add_summary_formatting,
                                            contextual_zipping_text=self.personality_config.contextual_zipping_text,
                                            summary_formatting_text=self.personality_config.summary_formatting_text,
                                        )
                self.step_end(f"summerizing {document_path.stem}")
                if self.personality_config.output_path:
                    self.save_text(summary, Path(self.personality_config.output_path)/(document_path.stem+"_summary.txt"))
                all_summaries +=f"\n## Summary of {document_path.stem}\n{summary}"
                self.full(all_summaries)
        self.new_message("")        
        ASCIIColors.yellow(all_summaries)
        summary = self.zip_text(
                                    all_summaries, 
                                    f"Fuse the fllowing summaries into a single comprehensive document where you extract relevant information and stick to the context.",
                                    add_summary_formatting=self.personality_config.add_global_summary_formatting,
                                    contextual_zipping_text=self.personality_config.global_contextual_zipping_text,
                                    summary_formatting_text=self.personality_config.global_summary_formatting_text,
                                    
                                )
        output =f"\n## Global summary\n{summary}"
        self.full(output)
        if self.personality_config.output_path:
            self.save_text(summary, Path(self.personality_config.output_path)/("global_summary.txt"))
            


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
                - current_language (str): The force language information.
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



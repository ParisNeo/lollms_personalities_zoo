from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality
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
                {"name":"zip_mode","type":"str","value":"hierarchical","options":["hierarchical","one_shot"], "help":"algorithm"},
                {"name":"zip_size","type":"int","value":512, "help":"the maximum size of the summary in tokens"},
                {"name":"contextual_zipping_text","type":"str","value":"", "help":"Here you can specify elements of the document that you want the AI to keep or to search for. This garantees that if found, those elements will not be filtered out which results in a more intelligent contextual based summary."},
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
        self.full(self.personality.help)
    
    def add_file(self, path, callback=None):
        """
        Here we implement the file reception handling
        """
        self.text_files.append(path)

    def save_text(self, text, path:Path):
        with open(path,"w", encoding="utf8") as f:
            f.write(text)
            
    def zip_document(self, document_path:Path,  output_path:Path=None, output =""):
        document_text = GenericDataLoader.read_file(document_path)
        tk = self.personality.model.tokenize(document_text)
        self.step_start(f"summerizing {document_path.stem}")
        if len(tk)<int(self.personality_config.zip_size):
                document_text = self.summerize([document_text],"Summerize this document chunk and do not add any comments after the summary.\nOnly extract the information from the provided chunk.\nDo not invent anything outside the provided text.","document chunk")
        else:
            depth=0
            while len(tk)>int(self.personality_config.zip_size):
                self.step_start(f"Comprerssing.. [depth {depth}]")
                chunk_size = int(self.personality.config.ctx_size*0.6)
                document_chunks = DocumentDecomposer.decompose_document(document_text, chunk_size, 0, self.personality.model.tokenize, self.personality.model.detokenize, True)
                document_text = self.summerize(document_chunks,f"""
Summerize this document chunk and do not add any comments after the summary.
Only extract the information from the provided chunk.
Do not invent anything outside the provided text.
Reduce the length of the text.
{'Keep the same language.' if self.personality_config.keep_same_language else ''}
{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}
{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}
{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}
{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}
{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}
{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}
""","document chunk")
                tk = self.personality.model.tokenize(document_text)
                self.step_end(f"Comprerssing.. [depth {depth}]")
                self.full(output+f"\n\n## Summerized chunk text:\n{document_text}")
                depth += 1
        self.step_start(f"Last composition")
        document_text = self.summerize(document_chunks,f"""
Summerize this document chunk and do not add any comments after the summary.
Only extract the information from the provided chunk.
Do not invent anything outside the provided text.
Reduce the length of the text.
{'Keep the same language.' if self.personality_config.keep_same_language else ''}
{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}
{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}
{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}
{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}
{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}
{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}
""","document chunk")
        self.step_end(f"Last composition")
        self.step_end(f"summerizing {document_path.stem}")
        if output_path:
            self.save_text(document_text, output_path/(document_path.stem+"_summary.txt"))
        return document_text, output
                    
        

    def start_zipping(self, prompt="", full_context=""):
        self.new_message("")
        for file in self.text_files:
            output=""
            file = Path(file)
            summary, output = self.zip_document(file, file.parent, output)
            output +=f"\n## Summary of {file.stem}\n{summary}"
            self.full(output)


    def run_workflow(self, prompt, previous_discussion_text="", callback=None):
        """
        Runs the workflow for processing the model input and output.

        This method should be called to execute the processing workflow.

        Args:
            prompt (str): The input prompt for the model.
            previous_discussion_text (str, optional): The text of the previous discussion. Default is an empty string.
            callback a callback function that gets called each time a new token is received
        Returns:
            None
        """
        self.callback = callback
        if len(self.text_files)>0:
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



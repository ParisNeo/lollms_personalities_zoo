from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
from lollms.paths import LollmsPaths
from lollms.helpers import ASCIIColors, trace_exception
from lollms.utilities import GenericDataLoader

import numpy as np
import json
from pathlib import Path
import numpy as np
import json
import subprocess
from urllib.parse import quote

class Processor(APScript):
    """
    A class that processes model inputs and outputs.

    Inherits from APScript.
    """

    def __init__(
                 self, 
                 personality: AIPersonality
                ) -> None:
        
        self.word_callback = None    

        personality_config_template = ConfigTemplate(
            [
                {"name":"build_keywords","type":"bool","value":True, "help":"If true, the model will first generate keywords before searching"},
                {"name":"save_db","type":"bool","value":False, "help":"If true, the vectorized database will be saved for future use"},
                {"name":"vectorization_method","type":"str","value":f"model_embedding", "options":["model_embedding", "ftidf_vectorizer"], "help":"Vectoriazation method to be used (changing this should reset database)"},
                
                {"name":"nb_chunks","type":"int","value":2, "min":1, "max":50,"help":"Number of data chunks to use for its vector (at most nb_chunks*max_chunk_size must not exeed two thirds the context size)"},
                {"name":"database_path","type":"str","value":f"{personality.name}_db.json", "help":"Path to the database"},
                {"name":"max_chunk_size","type":"int","value":512, "min":10, "max":personality.config["ctx_size"],"help":"Maximum size of text chunks to vectorize"},
                {"name":"chunk_overlap_sentences","type":"int","value":1, "min":0, "max":personality.config["ctx_size"],"help":"Overlap between chunks"},
                
                {"name":"max_answer_size","type":"int","value":512, "min":10, "max":personality.config["ctx_size"],"help":"Maximum number of tokens to allow the generator to generate as an answer to your question"},
                
                {"name":"data_visualization_method","type":"str","value":f"PCA", "options":["PCA", "TSNE"], "help":"The method to be used to show data"},
                {"name":"interactive_mode_visualization","type":"bool","value":False, "help":"If true, you can get an interactive visualization where you can point on data to get the text"},
                {"name":"visualize_data_at_startup","type":"bool","value":False, "help":"If true, the database will be visualized at startup"},
                {"name":"visualize_data_at_add_file","type":"bool","value":False, "help":"If true, the database will be visualized when a new file is added"},
                {"name":"visualize_data_at_generate","type":"bool","value":False, "help":"If true, the database will be visualized at generation time"},
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
                                        "zip":self.zip_doc,
                                        "clear_database": self.clear_database
                                    },
                                    "default": self.zip_text
                                },                           
                            ]
                        )
        self.state = 0
        self.ready = False
        self.personality = personality
        self.callback = None
        self.text = ""
        self.paragraphs = None


    def install(self):
        super().install()

        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Step 2: Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])            

        ASCIIColors.success("Installed successfully")

    def clear_database(self, prompt, full_context):
        self.text=""

    def help(self, prompt, full_context):
        self.full(self.personality.help, self.callback)

    def build_db(self):
        ASCIIColors.info("-> Loading text data"+ASCIIColors.color_orange)
        for file in self.files:
            try:
                if Path(file).suffix==".pdf":
                    text =  GenericDataLoader.read_pdf_file(file)
                elif Path(file).suffix==".docx":
                    text =  GenericDataLoader.read_docx_file(file)
                elif Path(file).suffix==".docx":
                    text =  GenericDataLoader.read_pptx_file(file)
                elif Path(file).suffix==".json":
                    text =  GenericDataLoader.read_json_file(file)
                elif Path(file).suffix==".csv":
                    text =  GenericDataLoader.read_csv_file(file)
                elif Path(file).suffix==".html":
                    text =  GenericDataLoader.read_html_file(file)
                elif Path(file).suffix in [".txt", ".md"]:
                    text =  GenericDataLoader.read_text_file(file)
                else:
                    ASCIIColors.error("File type not supported")
                    return False


                self.text += text + "\n"
                self.paragraphs = self.chunk_text_into_paragraphs(self.text, self.personality_config.max_chunk_size)
                              
                print(ASCIIColors.color_reset)
                ASCIIColors.success(f"File {file} vectorized successfully")
                self.ready = True
                return True
            except Exception as ex:
                ASCIIColors.error(f"Couldn't vectorize {file}: The vectorizer threw this exception:{ex}")
                trace_exception(ex)
                return False

    def add_file(self, path):
        super().add_file(path)
        self.prepare()
        self.step_start("Reading document",self.callback)
        self.build_db()
        self.step_end("Reading document",self.callback)

    @staticmethod
    def is_end_of_sentence(char):
        # A simple heuristic to detect the end of a sentence.
        # We assume a sentence ends with a period, exclamation mark, or question mark.
        return char in {'.', '!', '?'}

    @staticmethod
    def chunk_text_into_paragraphs(text, max_chunk_size):
        sentences = []  # List to store individual sentences
        current_sentence = ""
        paragraphs = []
        current_paragraph = ""
        current_size = 0

        # Tokenize the text into individual sentences
        for char in text:
            current_sentence += char
            current_size += 1

            if Processor.is_end_of_sentence(char):
                sentences.append(current_sentence.strip())
                current_sentence = ""

        # Create paragraphs from the sentences while respecting the max_chunk_size
        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size <= max_chunk_size:
                current_paragraph += sentence + " "
                current_size += sentence_size + 1  # +1 accounts for the space after the period
            else:
                # Add the current paragraph to the list of paragraphs
                paragraphs.append(current_paragraph.strip())

                # Reset variables for the next paragraph
                current_paragraph = sentence + " "
                current_size = sentence_size + 1

        # Add the last paragraph to the list of paragraphs
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())

        return paragraphs
    
    def zip_text(self, paragraph, full_text):
            full_text = f"""Summerize the following paragraph:
{paragraph}
summary:"""
            summary = self.generate(full_text, self.personality_config["max_answer_size"]).strip()            
            self.full(summary, self.callback)

    def zip_doc(self, prompt, full_context):
        if not self.paragraphs:
            self.full("Please upload a file to be zipped first.")
            return
        full_summery = ""
        for i,paragraph in enumerate(self.paragraphs):
            ASCIIColors.info(f"Processing paragraph {i+1}/{len(self.paragraphs)}")
            self.step_start(f"Processing paragraph {i+1}/{len(self.paragraphs)}", self.callback)
            self.full(paragraph)
            full_text = f"""Summerize the following paragraph:
{paragraph}
summary:"""
            summary = self.generate(full_text, self.personality_config["max_answer_size"]).strip()            
            self.full(f"# Original\n{paragraph}\nSummary:\n{summary}", self.callback)
            full_summery+=summary+"\n"
            self.step_end(f"Processing paragraph {i+1}/{len(self.paragraphs)}", self.callback)
        self.full(full_summery, self.callback)


    def prepare(self):
        if self.text=="":
            self.build_db()
            

    def run_workflow(self, prompt, full_context="", callback=None):
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
        # State machine
        self.callback = callback
        self.prepare()

        self.process_state(prompt, full_context, callback)

        return ""




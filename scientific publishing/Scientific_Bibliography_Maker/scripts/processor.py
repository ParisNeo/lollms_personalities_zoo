import subprocess
from pathlib import Path
from lollms.helpers import ASCIIColors
from lollms.utilities import PackageManager
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality
import re
import importlib
import requests
from tqdm import tqdm
import shutil
from lollms.types import GenerationPresets
import json
from functools import partial
import os
from lollms.utilities import PromptReshaper
from safe_store import TextVectorizer, VectorizationMethod, VisualizationMethod

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
        self.word_callback = None
        self.arxiv = None
        personality_config_template = ConfigTemplate(
            [
                {"name":"Formulate_key_words","type":"bool","value":True, "help":"Before doing the search the AI creates a keywords list that gets sent to the arxiv search engine."},
                {"name":"max_generation_prompt_size","type":"int","value":2048, "min":10, "max":personality.config["ctx_size"]},
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
                            [],
                            callback=callback
                        )
        self.previous_versions = []
        self.code=[]
        self.abstract_vectorizer = TextVectorizer(
                        vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,#=VectorizationMethod.BM25_VECTORIZER,
                        data_visualization_method=VisualizationMethod.PCA,#VisualizationMethod.PCA,
                        save_db=False
                    )
        self.full_documents_vectorizer = TextVectorizer(
                        vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,#=VectorizationMethod.BM25_VECTORIZER,
                        data_visualization_method=VisualizationMethod.PCA,#VisualizationMethod.PCA,
                        save_db=False
                    )

        
    def install(self):
        super().install()
        # Get the current directory
        root_dir = self.personality.lollms_paths.personal_path
        # We put this in the shared folder in order as this can be used by other personalities.
        shared_folder = root_dir/"shared"

        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Step 2: Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])            
        ASCIIColors.success("Installed successfully")
        
    def print_prompt(self, title, prompt):
        ASCIIColors.red("*-*-*-*-*-*-*-* ", end="")
        ASCIIColors.red(title, end="")
        ASCIIColors.red(" *-*-*-*-*-*-*-*")
        ASCIIColors.yellow(prompt)
        ASCIIColors.red(" *-*-*-*-*-*-*-*")        
    
    def prepare(self):
        if self.arxiv is None:
            import arxiv
            self.arxiv = arxiv

    def run_workflow(self, prompt, full_context="", callback=None):
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
        self.prepare()
        conditionning = self.personality.conditionning_commands
        
        # Define your search query
        query = prompt

        # Specify the number of results you want
        num_results = 5

        # Specify the folder where you want to save the articles
        download_folder = self.personality.lollms_paths.personal_outputs_path/"arxiv_articles"

        # Create the download folder if it doesn't exist
        download_folder.mkdir(parents=True, exist_ok=True)
        if self.personality_config.Formulate_key_words:
            self.step_start("Building Keywords...")
            pr  = PromptReshaper("""!@>Instructions:
Act as keywords extractor. Your job is to extract a coma separated list of keywords from the user prompt
!@>User prompt:                               
{{initial_prompt}}
!@>keywords: """)
            prompt = pr.build({
                    "previous_discussion":full_context,
                    "initial_prompt":query
                    }, 
                    self.personality.model.tokenize, 
                    self.personality.model.detokenize, 
                    self.personality.model.config.ctx_size,
                    ["previous_discussion"]
                    )
            self.print_prompt("Ask to build keywords",prompt)
            keywords = self.generate(prompt, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
            self.step_end("Building Keywords...")
            if keywords=="":
                ASCIIColors.error("The AI failed to build a keywords list. Using the prompt as keywords")
                keywords=query
        else:
            keywords=query
        self.step_start(f"Searching for :\n{keywords}")

        # Search for articles
        search_results = self.arxiv.Search(query=query, max_results=num_results).results()

        # Download and save articles
        for result in search_results:
            pdf_url = result.pdf_url
            if pdf_url:
                # Get the PDF content
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    # Create the filename for the downloaded article
                    filename = download_folder/f"{result.entry_id.split('/')[-1]}.pdf"
                    self.abstract_vectorizer.add_document(result.entry_id.split('/')[-1], result.summary, chunk_size=self.personality.config.data_vectorization_chunk_size, overlap_size=self.personality.config.data_vectorization_overlap_size, force_vectorize=False, add_as_a_bloc=False)
                    # Save the PDF to the specified folder
                    with open(filename, "wb") as file:
                        file.write(response.content)
                    print(f"Downloaded {result.title} to {filename}")
                    
                else:
                    print(f"Failed to download {result.title}")
        self.step_end(f"Searching for :\n{keywords}")
        self.step_start(f"Indexing database")
        self.abstract_vectorizer.index()
        self.step_end(f"Indexing database")
        

        self.step_start(f"Building answer")
        str_docs=""
        docs, sorted_similarities = self.abstract_vectorizer.recover_text(keywords, top_k=self.personality.config.data_vectorization_nb_chunks)
        for doc, infos in zip(docs, sorted_similarities):
            str_docs+=f"document chunk:\nchunk path: {infos[0]}\nchunk content:{doc}"



        if str_docs!="":                
            pr = PromptReshaper("{{conditionning}}\n!@>document chunks:\n{{doc}}\n{{content}}")
            discussion_messages = pr.build({
                                    "doc":str_docs,
                                    "conditionning":conditionning,
                                    "content":full_context
                                    }, self.personality.model.tokenize, self.personality.model.detokenize, self.personality.config.ctx_size, place_holders_to_sacrifice=["content"])
        else:
            pr = PromptReshaper("{{conditionning}}\n{{content}}")
            discussion_messages = pr.build({
                                    "conditionning":conditionning,
                                    "content":full_context
                                    }, self.personality.model.tokenize, self.personality.model.detokenize, self.personality.config.ctx_size, place_holders_to_sacrifice=["content"])
        
        self.print_prompt("Ask to build keywords",prompt)
        output = self.generate(discussion_messages, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
        self.step_end(f"Building answer")
        self.full(output)

        



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
from urllib.parse import quote
import requests 

def query_server(base_url, query_params):
    url = base_url + "?" + "&".join([f"{key}={value}" for key, value in query_params.items()])
    response = requests.get(url)
    json_data = response.json()
    return json_data

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
                {"name":"nb_arxiv_results","type":"int","value":10, "min":1, "help":"number of results to recover for ARXIV"},                
                {"name":"nb_hal_results","type":"int","value":10, "min":1, "help":"number of results to recover for HAL"},                
                {"name":"data_vectorization_nb_chunks","type":"int","value":5, "min":1, "help":"number of results to use for final text"},
                {"name":"Formulate_key_words","type":"bool","value":True, "help":"Before doing the search the AI creates a keywords list that gets sent to the arxiv search engine."},
                {"name":"read_abstracts","type":"bool","value":True, "help":"With this, the AI reads each abstract and judges if it is related to the work or not and filter out unrelated papers"},
                {"name":"relevance_check_severiry","type":"int","value":5, "min":1, "max":10, "help":"A value that reflects how severe is selection criterion"},
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
        conditionning = self.personality.personality_conditioning

        #Prepare full report
        report = []

        # Define your search query
        query = prompt

        # output
        articles_checking_text = ""


        # Specify the folder where you want to save the articles
        download_folder = self.personality.lollms_paths.personal_outputs_path/"arxiv_articles"

        # Create the download folder if it doesn't exist
        download_folder.mkdir(parents=True, exist_ok=True)
        if self.personality_config.Formulate_key_words:
            self.step_start("Building Keywords...")
            self.full("")
            keywords = self.fast_gen("""!@>Instructions:
Act as keywords extractor. Your job is to extract a coma separated list of keywords from the user prompt to be used by the search engine. The keywords should be tightly related to the subject.
!@>user prompt: {{initial_prompt}}
!@>keywords: """, self.personality_config.max_generation_prompt_size, {
                    "previous_discussion":full_context,
                    "initial_prompt":query
                    }, ["previous_discussion"], self.personality.config.debug)
            self.step_end("Building Keywords...")
            self.full(keywords)
            if keywords=="":
                ASCIIColors.error("The AI failed to build a keywords list. Using the prompt as keywords")
                keywords=query
        else:
            keywords=query
        articles_checking_text+=f"Keywords :\n{keywords}"
        self.full(articles_checking_text)
        
        self.step_end(f"Searching and processing {self.personality_config.nb_arxiv_results+self.personality_config.nb_hal_results} documents")
        
        # Search for articles
        relevant_file_paths = []
        
        
        
        
        
        
        
        # ----------------------------------- ARXIV ----------------------------------
        if self.personality_config.nb_arxiv_results>0:
            self.step_start(f"Searching articles on arxiv")
            search_results = self.arxiv.Search(query=query, max_results=self.personality_config.nb_arxiv_results).results()
            self.step_end(f"Searching articles on arxiv")

            # Download and save articles
            for i, result in enumerate(search_results):
                pdf_url = result.pdf_url
                if pdf_url:
                    relevance_explanation = ""
                    document_file_name = result.entry_id.split('/')[-1]
                    self.step_start(f"Processing document {i+1}/{self.personality_config.nb_arxiv_results}: {document_file_name}")
                    # Get the PDF content
                    response = requests.get(pdf_url)
                    if response.status_code == 200:
                        # Create the filename for the downloaded article
                        filename = download_folder/f"{document_file_name}.pdf"
                        # Save the PDF to the specified folder
                        with open(filename, "wb") as file:
                            file.write(response.content)
                        ASCIIColors.yellow(f"{i+1}/{self.personality_config.nb_arxiv_results} - Downloaded {result.title}\n    to {filename}")
                        authors = ",".join([str(a.name) for a in result.authors])
                        if self.personality_config.read_abstracts:
                            is_relevant = self.fast_gen("""!@>Instructions:
Act as document relevance and answer the following question with Yes or No.
Use the required relevance level to judge the relevance
required relevance:{{relevance_check_severiry}}/10
!@>document:
title: {{title}}
content: {{content}}
!@>subject:{{initial_prompt}}
!@>relevance of the document to the subject: """, self.personality_config.max_generation_prompt_size, {
                                    "title":result.title,
                                    "authors":authors,
                                    "content":result.summary,
                                    "initial_prompt":query,
                                    "relevance_check_severiry":str(self.personality_config.relevance_check_severiry)
                                    }, debug=self.personality.config.debug)
                            if "yes" in is_relevant.lower():
                                self.abstract_vectorizer.add_document(result.entry_id.split('/')[-1], f"title:{result.title}\nauthors:{authors}\nabstract:{result.summary}", chunk_size=self.personality.config.data_vectorization_chunk_size, overlap_size=self.personality.config.data_vectorization_overlap_size, force_vectorize=False, add_as_a_bloc=False)
                                relevance = "relevant"
                                relevance_explanation = self.fast_gen("""!@>Instructions: Explain why you think this document is relevant to the subject.
!@>document:
title: {{title}}
authors: {{authors}}
content: {{content}}
!@>subject: {{initial_prompt}}
!@>Relevance explanation: """, self.personality_config.max_generation_prompt_size, {
                                    "title":result.title,
                                    "content":result.summary,
                                    "initial_prompt":query,
                                    "relevance_check_severiry":str(self.personality_config.relevance_check_severiry)
                                    }, debug=self.personality.config.debug)
                                relevant_file_paths.append(filename)
                            else:
                                relevance = "irrelevant"
                        else:
                            self.abstract_vectorizer.add_document(result.entry_id.split('/')[-1], f"title:{result.title}\nabstract:{result.summary}", chunk_size=self.personality.config.data_vectorization_chunk_size, overlap_size=self.personality.config.data_vectorization_overlap_size, force_vectorize=False, add_as_a_bloc=False)
                            relevance = "unchecked"

                        report_entry={
                            "title":result.title,
                            "authors":authors,
                            "abstract":result.summary,
                            "relevance":relevance,
                            "explanation":relevance_explanation,
                            "url":pdf_url,
                            "file_path":str(filename)
                        }
                        
                        relevance = f'<p style="color: red;">{relevance}</p>' if relevance=="irrelevant" else f'<p style="color: green;">{relevance}</p>\n<b>Explanation</b>\n{relevance_explanation}'  if relevance=="relevant" else f'<p style="color: gray;">{relevance}</p>' 
                        fn = str(filename).replace('\\','/')
                        articles_checking_text+=f"\n\n---\n\n<b>Title</b>: {result.title}\n\n<b>Authors</b>: {authors}\n{relevance}\n\n<b>File</b>: <a href='/open_file?path={fn}'>{document_file_name}</a>"
                        self.full(articles_checking_text)
                        report.append(report_entry)
                        self.step_end(f"Processing document {i+1}/{self.personality_config.nb_arxiv_results}: {document_file_name}")

                    else:
                        ASCIIColors.red(f"Failed to download {result.title}")
                        self.step_start(f"{i}/{self.personality_config.nb_arxiv_results} {result.title}")
                        self.step_end(f"Processing document {i+1}/{self.personality_config.nb_arxiv_results}: {document_file_name}", False)
            






        # ----------------------------------- HAL ----------------------------------            
        if self.personality_config.nb_hal_results>0:
            self.step_start(f"Searching articles on hal")
            base_url = "http://api.archives-ouvertes.fr/search/"
            query_params = {
                "q": f"title_t:{query}",
                "fl": "label_s,en_title_s,uri_s,abstract_s",
                "rows": self.personality_config.nb_hal_results,
                "sort": "submittedDate_tdate desc"
            }
            search_results = query_server(base_url, query_params)            
            self.step_end(f"Searching articles on hal")
            # Download and save articles
            for i, result in enumerate(search_results["response"]["docs"]):
                pdf_url = result["uri_s"]
                if pdf_url:
                    relevance_explanation = ""
                    document_file_name = result["uri_s"].split('/')[-1]
                    self.step_start(f"Processing document {i+1}/{self.personality_config.nb_hal_results}: {document_file_name}")
                    # Get the PDF content
                    response = requests.get(pdf_url)
                    if response.status_code == 200:
                        # Create the filename for the downloaded article
                        filename = download_folder/f"{document_file_name}.pdf"
                        # Save the PDF to the specified folder
                        with open(filename, "wb") as file:
                            file.write(response.content)
                        ASCIIColors.yellow(f"{i+1}/{self.personality_config.nb_arxiv_results} - Downloaded {result['en_title_s'][0]}\n    to {filename}")
                        if self.personality_config.read_abstracts:
                            is_relevant = self.fast_gen("""!@>Instructions:
Act as document relevance and answer the following question with Yes or No.
Use the required relevance level to judge the relevance
required relevance:{{relevance_check_severiry}}/10
!@>document:
title: {{title}}
content: {{content}}
!@>subject:{{initial_prompt}}
!@>relevance of the document to the subject: """, self.personality_config.max_generation_prompt_size, {
                                    "title":result["en_title_s"][0],
                                    "content":result['abstract_s'][0],
                                    "initial_prompt":query,
                                    "relevance_check_severiry":str(self.personality_config.relevance_check_severiry)
                                    }, debug=self.personality.config.debug)
                            if "yes" in is_relevant.lower():
                                self.abstract_vectorizer.add_document(result["uri_s"].split('/')[-1], f"title:{result['en_title_s'][0]}\nauthors:{result['label_s']}\nabstract:{result['abstract_s'][0]}", chunk_size=self.personality.config.data_vectorization_chunk_size, overlap_size=self.personality.config.data_vectorization_overlap_size, force_vectorize=False, add_as_a_bloc=False)
                                relevance = "relevant"
                                relevance_explanation = self.fast_gen("""!@>Instructions: Explain why you think this document is relevant to the subject.
!@>document:
title: {{title}}
authors: {{authors}}
content: {{content}}
!@>subject: {{initial_prompt}}
!@>Relevance explanation: """, self.personality_config.max_generation_prompt_size, {
                                    "title":result["en_title_s"][0],
                                    "content":result['abstract_s'][0],
                                    "initial_prompt":query,
                                    "relevance_check_severiry":str(self.personality_config.relevance_check_severiry)
                                    }, debug=self.personality.config.debug)
                                relevant_file_paths.append(filename)
                            else:
                                relevance = "irrelevant"
                        else:
                            self.abstract_vectorizer.add_document(result["uri_s"].split('/')[-1], f"title:{result['en_title_s'][0]}\nabstract:{result['abstract_s'][0]}", chunk_size=self.personality.config.data_vectorization_chunk_size, overlap_size=self.personality.config.data_vectorization_overlap_size, force_vectorize=False, add_as_a_bloc=False)
                            relevance = "unchecked"

                        report_entry={
                            "title":result['en_title_s'][0],
                            "authors":result['label_s'],
                            "abstract":result['abstract_s'][0],
                            "relevance":relevance,
                            "explanation":relevance_explanation,
                            "url":pdf_url,
                            "file_path":str(filename)
                        }
                        
                        relevance = f'<p style="color: red;">{relevance}</p>' if relevance=="irrelevant" else f'<p style="color: green;">{relevance}</p>\n<b>Explanation</b>\n{relevance_explanation}'  if relevance=="relevant" else f'<p style="color: gray;">{relevance}</p>' 
                        fn = str(filename).replace('\\','/')
                        articles_checking_text+=f"\n\n---\n\n<b>Title</b>: {result['en_title_s']}\n\n<b>Authors</b>: {result['label_s']}\n{relevance}\n\n<b>File</b>: <a href='/open_file?path={fn}'>{document_file_name}</a>"
                        self.full(articles_checking_text)
                        report.append(report_entry)
                        self.step_end(f"Processing document {i+1}/{self.personality_config.nb_hal_results}: {document_file_name}")

                    else:
                        ASCIIColors.red(f"Failed to download {result['en_title_s']}")
                        self.step_start(f"{i}/{self.personality_config.nb_hal_results} {result['en_title_s']}")
                        self.step_end(f"Processing document {i+1}/{self.personality_config.nb_hal_results}: {document_file_name}", False)
                    

        self.json("Report",report)
        self.step_end(f"Searching and processing {self.personality_config.nb_arxiv_results+self.personality_config.nb_hal_results} documents")
        self.step_start(f"Indexing database")
        self.abstract_vectorizer.index()
        self.step_end(f"Indexing database")
        

        self.step_start(f"Building answer")
        str_docs=""
        self.abstract_vectorizer.recover_chunk_by_index(0)
        docs, sorted_similarities = self.abstract_vectorizer.recover_text(keywords, top_k=self.personality_config.data_vectorization_nb_chunks)
        for doc, infos in zip(docs, sorted_similarities):
            str_docs+=f"->Document<-:\ndocument path: {infos[0]}\nsummary:{doc}"

        if str_docs!="":
            pr = PromptReshaper("Documentation section:\n{{doc}}\nDiscussion:\n{{content}}")
            discussion_messages = pr.build({
                                    "doc":str_docs,
                                    "content":full_context+" Ok, after reading the provided document chunks, here is my answer to your request:"
                                    }, self.personality.model.tokenize, self.personality.model.detokenize, self.personality.config.ctx_size, place_holders_to_sacrifice=["content"])
        else:
            pr = PromptReshaper("{{conditionning}}\n{{content}}")
            discussion_messages = pr.build({
                                    "conditionning":conditionning,
                                    "content":full_context
                                    }, self.personality.model.tokenize, self.personality.model.detokenize, self.personality.config.ctx_size, place_holders_to_sacrifice=["content"])
        self.print_prompt("Ask to build keywords",discussion_messages)
        output = self.generate(discussion_messages, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
        self.step_end(f"Building answer")
        
        ASCIIColors.yellow(output)
        self.new_message("")
        summary_text=f"\n<b>Summary</b>:\n{output}"
        self.full(summary_text)

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
from typing import Callable

def query_server(base_url, query_params):
    url = base_url + "?" + "&".join([f"{key}={value}" for key, value in query_params.items()])
    response = requests.get(url)
    json_data = response.json()
    return json_data

def classify_reports(reports):
    sorted_reports = sorted(reports, key=lambda x: x['relevance_score'], reverse=True)
    return sorted_reports

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
        # C:\Program Files\MiKTeX\miktex\bin\x64
        # https://developer.ieee.org/Python3_Software_Development_Kit#fullTextExample
        personality_config_template = ConfigTemplate(
            [
                {"name":"ieee_explore_key","type":"str","value":"", "min":1, "help":"The key to use for accessing ieeexplore api"},                
                {"name":"research_output_path","type":"str","value":"", "min":1, "help":"path to a folder where to put the downloaded bibliography files as well as the summary and analysis results"},                
                {"name":"pdf_latex_path","type":"str","value":"", "min":1, "help":"path to the pdflatex file pdf compiler used to compile pdf files"},
                {"name":"make_survey","type":"bool","value":True, "min":1, "help":"build a survey report"},
                {"name":"output_file_name","type":"str","value":"summary_latex", "min":1, "help":"Name of the pdf file to generate"},
                {"name":"nb_arxiv_results","type":"int","value":10, "min":1, "help":"number of results to recover for ARXIV"},                
                {"name":"nb_hal_results","type":"int","value":10, "min":1, "help":"number of results to recover for HAL"},                
                {"name":"data_vectorization_nb_chunks","type":"int","value":5, "min":1, "help":"number of results to use for final text"},
                {"name":"Formulate_search_query","type":"bool","value":True, "help":"Before doing the search the AI creates a keywords list that gets sent to the arxiv search engine."},
                {"name":"read_abstracts","type":"bool","value":True, "help":"With this, the AI reads each abstract and judges if it is related to the work or not and filter out unrelated papers"},
                {"name":"read_content","type":"bool","value":True, "help":"With this, the AI reads the whole document and judges if it is related to the work or not and filter out unrelated papers"},
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


    def analyze_pdf(
                        self,
                        initial_prompt, 
                        pdf_url, 
                        title, 
                        authors, 
                        abstract, 
                        file_name, 
                        document_file_name, 
                        articles_checking_text, 
                        report):
        
        if self.personality_config.read_abstracts:
            relevance_score = self.fast_gen("\n".join([
                "!@>system:",
                "Act as document relevance analyzer. Respond with the document relevance level out of 10.",
                "A document with a relevance of 10 is a document that adresses exactly the subject proposed by the user.",
                "Make sure the document is not talking about something else that is not really linked with the subject but may use some terms that can be found in the subject.",
                "!@>document:",
                "title: {{title}}",
                "content: {{content}}",
                "!@>subject:{{initial_prompt}}",
                "!@>relevance value out of 10: "]), 10, {
                    "title":title,
                    "content":abstract,
                    "initial_prompt":initial_prompt,
                    "relevance_check_severiry":str(self.personality_config.relevance_check_severiry)
                    }, debug=self.personality.config.debug, callback=self.sink)
            relevance_score = relevance_score
            relevance_score = self.find_numeric_value(relevance_score)
            if relevance_score is None:
                articles_checking_text.append(f"\n\n---\n\n<b>Title</b>: {title}\n\n<b>Authors</b>: {authors}\n<p style='color: red;'>Couldn't determine relevance</p>\n\n<b>File</b>: <a href='file:///{fn}'>{document_file_name}</a>")
                report_entry={
                    "title":title,
                    "authors":authors,
                    "abstract":abstract,
                    "relevance":None,
                    "relevance_score":0,
                    "explanation":relevance_explanation,
                    "url":pdf_url,
                    "file_path":str(file_name)
                }
                report.append(report_entry)
                self.full("\n".join(articles_checking_text))
                self.warning("The AI agent didn't respond to the relevance question correctly")
                return False
            if relevance_score>=float(self.personality_config.relevance_check_severiry):
                self.abstract_vectorizer.add_document(pdf_url.split('/')[-1], f"title:{title}\nauthors:{authors}\nabstract:{abstract}", chunk_size=self.personality.config.data_vectorization_chunk_size, overlap_size=self.personality.config.data_vectorization_overlap_size, force_vectorize=False, add_as_a_bloc=False)
                relevance = f"relevance score {relevance_score}/10"
                relevance_explanation = self.fast_gen("\n".join([
                        "!@>system:",
                        "Explain why you think this document is relevant to the subject by summerizing the abstract and hilighting interesting information that can serve the subject."
                        "!@>document:",
                        "title: {{title}}",
                        "authors: {{authors}}",
                        "content: {{content}}",
                        "!@>subject: {{initial_prompt}}",
                        "!@>Explanation: "                    
                        ]), self.personality_config.max_generation_prompt_size, {
                        "title":title,
                        "abstract":abstract,
                        "initial_prompt":initial_prompt,
                        "relevance_check_severiry":str(self.personality_config.relevance_check_severiry)
                    },
                    debug=self.personality.config.debug,
                    callback=self.sink)
            else:
                relevance = "irrelevant"
                relevance_explanation = ""
        else:
            self.abstract_vectorizer.add_document(pdf_url.split('/')[-1], f"title:{title}\nabstract:{abstract}", chunk_size=self.personality.config.data_vectorization_chunk_size, overlap_size=self.personality.config.data_vectorization_overlap_size, force_vectorize=False, add_as_a_bloc=False)
            relevance = "unchecked"
            relevance_explanation = ""

        report_entry={
            "title":title,
            "authors":authors,
            "abstract":abstract,
            "relevance":relevance,
            "relevance_score":relevance_score,
            "explanation":relevance_explanation,
            "url":pdf_url,
            "file_path":str(file_name)
        }        
        relevance = f'<p style="color: red;">{relevance}</p>' if relevance=="irrelevant" else f'<p style="color: green;">{relevance}</p>\n<b>Explanation</b>\n{relevance_explanation}'  if relevance_score>float(self.personality_config.relevance_check_severiry) else f'<p style="color: gray;">{relevance}</p>' 
        fn = str(file_name).replace('\\','/')
        articles_checking_text.append(f"\n\n---\n\n<b>Title</b>: {title}\n\n<b>Authors</b>: {authors}\n<b>File</b>:{self.build_a_file_link(fn,document_file_name)}\n\nRelevance:\n{relevance}\n\n")
        self.full("\n".join(articles_checking_text))
        report.append(report_entry)
        return True
        
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
        self.prepare()
        conditionning = self.personality.personality_conditioning

        #Prepare full report
        report = []

        # Define your search query
        query = prompt

        # output
        articles_checking_text = []


        # Specify the folder where you want to save the articles
        if self.personality_config.research_output_path!="":
            download_folder = Path(self.personality_config.research_output_path)
        else:
            download_folder = self.personality.lollms_paths.personal_outputs_path/"research_articles"
            self.warning(f"You did not specify an analysis folder path to put the output into, please do that in the personality settings page.\nUsing default path {download_folder}")

        # Create the download folder if it doesn't exist
        download_folder.mkdir(parents=True, exist_ok=True)
        if self.personality_config.Formulate_search_query:
            self.step_start("Building query...")
            self.full("")
            keywords = self.fast_gen("\n".join([
                "!@>system:",
                "Act as arxiv search specialist. Your job is to reformulate the user requestio into a search query.",
                "!@>user prompt: {{initial_prompt}}",
                "!@>query: "
            ]), self.personality_config.max_generation_prompt_size, {
                    "previous_discussion":previous_discussion_text,
                    "initial_prompt":query
                    }, ["previous_discussion"], self.personality.config.debug)
            self.step_end("Building Keywords...")
            self.full(keywords)
            if keywords=="":
                ASCIIColors.error("The AI failed to build a keywords list. Using the prompt as keywords")
                keywords=query
        else:
            keywords=query
        articles_checking_text.append(f"### Keywords :\n{keywords}\n")
        self.full("\n".join(articles_checking_text))
        
        self.step_end(f"Searching and processing {self.personality_config.nb_arxiv_results+self.personality_config.nb_hal_results} documents")
        
        # Search for articles
       
        # ----------------------------------- ARXIV ----------------------------------
        if self.personality_config.nb_arxiv_results>0:
            self.step_start(f"Searching articles on arxiv")
            search_results_ = self.arxiv.Search(query=query, max_results=self.personality_config.nb_arxiv_results).results()
            self.step_end(f"Searching articles on arxiv")
            search_results =[]
            for i, result in enumerate(search_results_):
                search_results.append(result)
            articles_checking_text.append(f"### Searching on arxiv {self.personality_config.nb_arxiv_results} articles.\n### Found : {len(search_results)} articles on the subject\n")
            self.full("\n".join(articles_checking_text))
            # Download and save articles
            for i, result in enumerate(search_results):
                pdf_url = result.pdf_url
                if pdf_url:
                    document_file_name = result.entry_id.split('/')[-1]
                    self.step_start(f"Processing document {i+1}/{self.personality_config.nb_arxiv_results}: {document_file_name}")
                    # Get the PDF content
                    response = requests.get(pdf_url)
                    if response.status_code == 200:
                        # Create the filename for the downloaded article
                        file_name = download_folder/f"{document_file_name}.pdf"
                        # Save the PDF to the specified folder
                        with open(file_name, "wb") as file:
                            file.write(response.content)
                        ASCIIColors.yellow(f"{i+1}/{self.personality_config.nb_arxiv_results} - Downloaded {result.title}\n    to {file_name}")
                        authors = ",".join([str(a.name) for a in result.authors])
                        
                        if self.analyze_pdf(query, pdf_url, result.title, authors, result.summary, file_name, document_file_name, articles_checking_text, report):
                            self.step_end(f"Processing document {i+1}/{self.personality_config.nb_arxiv_results}: {document_file_name}")
                        else:
                            self.step_end(f"Processing document {i+1}/{self.personality_config.nb_arxiv_results}: {document_file_name}", False)
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
            articles_checking_text.append(f"### Searching on hal {self.personality_config.nb_hal_results} articles.\n### Found : {len(search_results['response']['docs'])} articles on the subject\n")
            self.full("\n".join(articles_checking_text))
            self.step_end(f"Searching articles on hal")

            # Download and save articles
            for i, result in enumerate(search_results["response"]["docs"]):
                if "en_title_s" not in result: result["en_title_s"]=["undefined"]
                pdf_url = result["uri_s"]
                if pdf_url:
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
                        self.analyze_pdf(title=result["en_title_s"][0],
                                authors=result['label_s'],
                                abstract=result['abstract_s'][0],
                                initial_prompt = query,
                                document_file_name = document_file_name)

                        self.step_end(f"Processing document {i+1}/{self.personality_config.nb_hal_results}: {document_file_name}")

                    else:
                        ASCIIColors.red(f"Failed to download {result['en_title_s']}")
                        self.step_start(f"{i}/{self.personality_config.nb_hal_results} {result['en_title_s']}")
                        self.step_end(f"Processing document {i+1}/{self.personality_config.nb_hal_results}: {document_file_name}", False)
                    

        self.json("Report",report)
        self.step_end(f"Searching and processing {self.personality_config.nb_arxiv_results+self.personality_config.nb_hal_results} documents")
        if len(report)>0:
            self.step_start(f"Indexing database")
            self.abstract_vectorizer.index()
            self.step_end(f"Indexing database")
            

            self.step_start(f"Building answer")
            str_docs=""
            #data = self.abstract_vectorizer.recover_chunk_by_index(0)
            docs, sorted_similarities, document_ids = self.abstract_vectorizer.recover_text(keywords, top_k=self.personality_config.data_vectorization_nb_chunks)
            for doc, infos in zip(docs, sorted_similarities):
                str_docs+=f"->Document<-:\ndocument path: {infos[0]}\nsummary:{doc}"

            if str_docs!="":
                pr = PromptReshaper("Documentation section:\n{{doc}}\nDiscussion:\n{{content}}")
                discussion_messages = pr.build({
                                        "doc":str_docs,
                                        "content":previous_discussion_text+" Ok, after reading the provided document chunks, here is my answer to your request:"
                                        }, self.personality.model.tokenize, self.personality.model.detokenize, self.personality.config.ctx_size, place_holders_to_sacrifice=["content"])
            else:
                pr = PromptReshaper("{{conditionning}}\n{{content}}")
                discussion_messages = pr.build({
                                        "conditionning":conditionning,
                                        "content":previous_discussion_text
                                        }, self.personality.model.tokenize, self.personality.model.detokenize, self.personality.config.ctx_size, place_holders_to_sacrifice=["content"])
            self.print_prompt("Ask to build keywords",discussion_messages)
            output = self.generate(discussion_messages, self.personality_config.max_generation_prompt_size).strip().replace("</s>","").replace("<s>","")
            self.step_end(f"Building answer")
            
            report = classify_reports(report)
            with open(download_folder/"report.json","w") as f:
                json.dump(report, f)

            summary_text=f"\n<b>Summary</b>:\n{output}"
            with open(download_folder/"summary.md","w",encoding="utf-8") as f:
                f.write(summary_text)
                        
            ASCIIColors.yellow(output)
            self.new_message("")
            self.full(summary_text)
            if self.personality_config.make_survey:
                summary = self.summerize_text(str(report),f"create a summary where you keep the author names, the title of the article and any thing important related to the prompt:\nprompt:{query}","summary")
                summary_latex = self.fast_gen(f"!@>instruction: Write a survey article out of the information provided below about the subject. Use academic style and cite the contributions.\nInput:\n{summary}\nOutput format : a complete latex document that should compile without errors and should contain inline bibliography\n!@>Output:\n```latex\n")
                with open(download_folder/f"{self.personality_config.output_file_name}.tex","w",encoding="utf-8") as f:
                    f.write(summary_latex)
                if self.personality_config.pdf_latex_path!="":
                    self.compile_latex(download_folder/f"{self.personality_config.output_file_name}.tex",self.personality_config.pdf_latex_path)
        else:
            self.personality.error("No article found about this subject!")

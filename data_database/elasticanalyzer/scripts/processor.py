"""
project: lollms
personality: # Place holder: Personality name 
Author: # Place holder: creator name 
description: # Place holder: personality description
"""
from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, MSG_TYPE
from lollms.databases.discussions_database import Discussion
from pathlib import Path
import subprocess
from typing import Callable
from datetime import datetime
import pandas as pd
import json
import io

import re

def parse_query(query):
    # Match the pattern @@function|param1|param2@@
    lq = len(query)
    parts = query.split("@@")
    if len(parts)>1:
        query_ = parts[1].split("@@")
        query_=query_[0]
        parts = query_.split("|")
        fn = parts[0]
        if len(parts)>1:
            params = parts[1:]
        else:
            params=[]
        try:
            end_pos = query.index("@@")
        except:
            end_pos = lq
        return fn, params, end_pos

    else:
        return None, None, 0

from elasticsearch import Elasticsearch

class ElasticSearchConnector:
    def __init__(self, host='http://localhost:9200', username=None, password=None):
        self.host = host
        self.username = username
        self.password = password
        self.es = self.connect()

    def connect(self):
        if self.username and self.password:
            es = Elasticsearch([self.host], http_auth=(self.username, self.password),verify_certs=False)
        else:
            es = Elasticsearch([self.host])
        return es

    def list_indexes(self):
        return self.es.cat.indices(h='index')

    def view_mapping(self, index):
        return self.es.indices.get_mapping(index=index)

    def query(self, index, body):
        return self.es.search(index=index, body=body)

    def add_entry(self, index, body):
        return self.es.index(index=index, body=body)

    def index_data(self, file_path, index_name):
        try:
            self.create_index(index_name)
            with open(file_path, 'r') as file:
                for line in file:
                    doc = json.loads(line)
                    if 'index' in doc:
                        self.es.index(index=index_name, body=doc['_source'], id=doc['_id'])
                    else:
                        self.es.index(index=index_name, body=doc)
            return True
        except Exception as e:
            print(f"Error indexing data: {e}")
            return False
        

    def ping(self):
        # Ping the Elasticsearch server
        response = self.es.ping()

        # Check if the server is reachable
        if response:
            print("Elasticsearch server is reachable")
            return True
        else:
            print("Elasticsearch server is not reachable")
            return False

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
        # Example entries
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        #       {"name":"make_scripted","type":"bool","value":False, "help":"Makes a scriptred AI that can perform operations using python script"},
        # Supported types:
        # str, int, float, bool, list
        # options can be added using : "options":["option1","option2"...]        
        personality_config_template = ConfigTemplate(
            [
                {"name":"server","type":"str","value":"https://localhost:9200", "help":"List of addresses of the server in form of ip or host name: port"},
                {"name":"index_name","type":"str","value":"", "help":"The index to be used for querying"},
                {"name":"mapping","type":"text","value":"", "help":"Mapping of the elastic search index"},
                {"name":"user","type":"str","value":"", "help":"The user name to connect to the database"},
                {"name":"password","type":"str","value":"", "help":"The password to connect to the elastic search database"},
                {"name":"max_execution_depth","type":"int","value":10, "help":"The maximum execution depth"},
                {"name":"output_folder_path","type":"str","value":"", "help":"Folder to put the output"},
                {"name":"output_format","type":"str","value":"markdown", "options":["markdown","html","latex"], "help":"Output format"},
                {"name":"max_nb_failures","type":"int","value":3, "help":"Maximum number of failures"},
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
                                    "commands": { # list of commands (don't forget to add these to your config.yaml file)
                                        "help":self.help,
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
    def install(self):
        super().install()
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")        

    def help(self, prompt="", full_context=""):
        self.full(self.personality.help)
    
    def add_file(self, path, client, callback=None):
        """
        Here we implement the file reception handling
        """
        super().add_file(path, client, callback)

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
        header_text = f"!@>Extra infos:\n"
        header_text += f"server:{self.personality_config.server}\n"
        if self.personality_config.index_name!="":
            header_text += f"index_name:{self.personality_config.index_name}\n"
        if self.personality_config.mapping!="":
            header_text += f"mapping:\n{self.personality_config.mapping}\n"
        if self.personality_config.user!="":
            header_text += f"user:\n{self.personality_config.user}\n"
        if self.personality_config.password!="":
            header_text += f"password:\n{self.personality_config.user}\n"

        full_prompt = header_text
        full_prompt += context_details["conditionning"]
        full_prompt += context_details["documentation"]
        full_prompt += context_details["knowledge"]
        full_prompt += context_details["user_description"]
        full_prompt += context_details["discussion_messages"]
        full_prompt += context_details["positive_boost"]
        full_prompt += context_details["negative_boost"]
        full_prompt += context_details["force_language"]
        full_prompt += context_details["fun_mode"]
        full_prompt += "If you need to issue a code to es, please do not add any extra text or explanations."
        full_prompt += context_details["ai_prefix"]
        self.personality.info("Generating")
        self.callback = callback

        max_nb_tokens_in_file = 3*self.personality.config.ctx_size/4

        failed=True
        nb_failures = 0
        while failed  and nb_failures<self.personality_config.max_nb_failures:
            nb_failures += 1
            output = self.fast_gen(full_prompt).replace("\\_","_")
            fn, params, next = parse_query(output)
            failed=False
            if fn:
                self.new_message("## Executing ...", MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI)
                es = ElasticSearchConnector(self.personality_config.server, self.personality_config.user, self.personality_config.password)
                if fn=="ping":
                    self.step("The LLM issued a ping command")
                    try:
                        status = es.ping()
                        self.full(self.build_a_document_block(f"Execution result:",None,f"{status}"), msg_type=MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI)
                        output = self.fast_gen(full_prompt+output+f"!@>es: ping response: {'Connection succeeded' if status else 'connection failed'}\n"+context_details["ai_prefix"], callback=self.sink).replace("\\_","_")
                    except Exception as ex:
                        self.full(f"## Execution result:\n{ex}")
                        output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"], callback=self.sink).replace("\\_","_")

                if fn=="list_indexes":
                    self.step("The LLM issued a list_indexes command")
                    try:
                        indexes = es.list_indexes()
                        self.full(self.build_a_document_block(f"Execution result:",None,f"{indexes}"), msg_type=MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI)
                        output = self.fast_gen(full_prompt+output+f"!@>es: indexes {indexes}\n"+context_details["ai_prefix"], callback=self.sink).replace("\\_","_")
                    except Exception as ex:
                        self.full(f"## Execution result:\n{ex}")
                        output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"], callback=self.sink).replace("\\_","_")
                
                if fn=="view_mapping":
                    self.step("The LLM issued a view mapping command")
                    if len(params)==1:
                        try:
                            mappings = es.view_mapping(params[0])
                            self.full(self.build_a_document_block(f"Execution result:",None,f"{mappings}"), msg_type=MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI)
                            output = self.fast_gen(full_prompt+output+f"!@>es: mapping\n{mappings}\n"+context_details["ai_prefix"], callback=self.sink).replace("\\_","_")
                        except Exception as ex:
                            self.full(f"## Execution result:\n{ex}")
                            output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"]).replace("\\_","_")
                    else:
                        ASCIIColors.warning("The AI issued the wrong number of parameters.\nTrying again")
                        self.full("The AI issued the wrong number of parameters.\nTrying again")
                        failed=True

                if fn=="query":
                    self.step("The LLM issued a query command")
                    if len(params)==2:
                        try:
                            qoutput = es.query(params[0], params[1])
                            self.full(self.build_a_document_block(f"Execution result:",None,f"{qoutput}"), msg_type=MSG_TYPE.MSG_TYPE_FULL_INVISIBLE_TO_AI)
                            if "hits" in qoutput.body and len(qoutput.body["hits"])>0:
                                self.step("Found hits")
                                output = ""
                                for hit in qoutput.body["hits"]:
                                    print(hit)
                                    prompt = full_prompt+output+f"!@>query entry:\n{hit}\n"+context_details["ai_prefix"]+"Here is a title followed by a summary of this entries in markdown format:\n"
                                    output += self.fast_gen(prompt, callback=self.sink).replace("\\_","_")
                                if self.personality_config.output_folder_path!="":
                                    # Get the current date
                                    current_date = datetime.date.today()

                                    # Format the date as 'year_month_day'
                                    formatted_date = current_date.strftime('%Y_%m_%d_%H_%M_%S')
                                    with open(Path(self.personality_config.output_folder_path)/f"result_{formatted_date}.md","w") as f:
                                        f.write(output)

                            else:
                                self.step("No Hits found")
                                prompt = full_prompt+output+f"!@>es: query output:\n{qoutput}\n"+context_details["ai_prefix"]
                                output = self.fast_gen(prompt, callback=self.sink).replace("\\_","_")

                            #df = pd.DataFrame(qoutput)

                            # pv = self.personality.model.tokenize(full_prompt+output+f"!@>es: query output:\n{qoutput}\n")
                            # cr = self.personality.model.tokenize(qoutput)
                            # ln = len(pv)
                            #output=""
                            #if ln
                            #while ln+len(cr)>max_nb_tokens_in_file:
                            #    tk = pv+("!@>es: query output:\n" if o=="" else "!@>es: query output:\n...\n")+cr[:max_nb_tokens_in_file-ln] 
                            #    output += self.fast_gen(self.personality.model.detokenize(tk)+context_details["ai_prefix"])
                        except Exception as ex:
                            output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"], callback=self.sink).replace("\\_","_")
                    else:
                        ASCIIColors.warning("The AI issued the wrong number of parameters.\nTrying again")
                        self.full("The AI issued the wrong number of parameters.\nTrying again")
                        failed=True
            self.new_message("")
            self.full(output)

        return output


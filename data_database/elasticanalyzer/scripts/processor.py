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
import subprocess
from typing import Callable
import sys
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
        return None, None

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
        full_prompt += context_details["ai_prefix"]
        self.personality.info("Generating")
        self.callback = callback
        
        output = self.fast_gen(full_prompt)
        fn, params, next = parse_query(output)
        if fn:
            es = ElasticSearchConnector(self.personality_config.server, self.personality_config.user, self.personality_config.password)
            if fn=="ping":
                try:
                    status = es.ping()
                    output = self.fast_gen(full_prompt+output+f"!@>es: ping response: {'Connection succeeded' if status else 'connection failed'}\n"+context_details["ai_prefix"])
                except Exception as ex:
                    output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"])

            if fn=="list_indexes":
                if len(params)==1:
                    try:
                        indexes = es.list_indexes()
                        output = self.fast_gen(full_prompt+output+f"!@>es: indexes {indexes}\n"+context_details["ai_prefix"])
                    except Exception as ex:
                        output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"])
            if fn=="view_mapping":
                if len(params)==1:
                    try:
                        indexes = es.view_mapping(params[0])
                        output = self.fast_gen(full_prompt+output+f"!@>es: mapping\n{indexes}\n"+context_details["ai_prefix"])
                    except Exception as ex:
                        output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"])

            if fn=="query":
                if len(params)==1:
                    try:
                        qoutput = es.query(params[0])
                        output = self.fast_gen(full_prompt+output+f"!@>es: query output:\n{qoutput}\n"+context_details["ai_prefix"])
                    except Exception as ex:
                        output = self.fast_gen(full_prompt+output+f"!@>es: error {ex}\n"+context_details["ai_prefix"])


        self.full(output)

        return output


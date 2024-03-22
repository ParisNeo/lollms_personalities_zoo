from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.utilities import PackageManager
from lollms.personality import APScript, AIPersonality
from lollms.types import MSG_TYPE
from lollms.internet import internet_search
from typing import Callable

from safe_store.generic_data_loader import GenericDataLoader
from safe_store.document_decomposer import DocumentDecomposer
import subprocess
from pathlib import Path
from datetime import datetime
import json

if not PackageManager.check_package_installed("feedparser"):
    PackageManager.install_package("feedparser")

import feedparser
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
                {"name":"output_folder","type":"str","value":"", "help":"The folder where all the files will be stored"},
                 
                {"name":"search_query","type":"text","value":"", "help":"Here you can put custom search query to be used. This automatically deactivates the rss, if you want the rss to work, then please empty this"},
                {"name":"rss_urls","type":"text","value":"https://feeds.bbci.co.uk/news/rss.xml, http://rss.cnn.com/rss/cnn_topstories.rss, https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml, https://www.theguardian.com/world/rss, https://www.reuters.com/rssfeed/topNews, http://feeds.foxnews.com/foxnews/latest, https://www.aljazeera.com/xml/rss/all.xml, https://www.bloomberg.com/politics/feeds/site.xml", "help":"Here you can put rss feed address to recover data."},
                {"name":"categories","type":"text","value":"World News,Entertainment,Sport,Technology,Education,Medicine,Space,R&D,Politics,Music,Business", "help":"The list of categories to help the AI organize the news."},
                {"name":"nb_rss_feed_pages","type":"int","value":5, "help":"the maximum number of rss feed pages to search"},
                {"name":"rss_scraping_type","type":"str","value":"quick","options":["quick","deep"], "help":"quick uses only the breafs to build the summary and the deep will scrape data from the website"},
                {"name":"nb_search_pages","type":"int","value":5, "help":"the maximum number of pages to search"},
                {"name":"quick_search","type":"bool","value":False, "help":"Quick search returns only a brief summary of the webpage"},
                {"name":"zip_mode","type":"str","value":"hierarchical","options":["hierarchical","one_shot"], "help":"algorithm"},
                {"name":"zip_size","type":"int","value":1024, "help":"the maximum size of the summary in tokens"},
                {"name":"buttons_to_press","type":"str","value":"I agree,accept", "help":"Buttons to be pressed in the pages you want to load. A comma separated text that can be seen on the button to press. The buttons will be pressed sequencially"},
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
                                        "start_scraping":self.start_scraping,
                                        "scrape_news":self.scrape_news
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
        
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # Install dependencies using pip from requirements.txt
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
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
        self.step_start("Performing internet search")
        pages = internet_search(query, self.personality_config.nb_search_pages, buttons_to_press=self.personality_config.buttons_to_press, quick_search=self.personality_config.quick_search)
        pages = internet_search("Latest news" if self.personality_config.search_query=="" else self.personality_config.search_query, self.personality_config.nb_search_pages, buttons_to_press=self.personality_config.buttons_to_press, quick_search=True)
        processed_pages = ""
        for page in pages:
            if self.personality_config.quick_search:
                page_text = f"page_title: {page['title']}\npage_brief:{page['brief']}"
            else:
                page_text = f"page_title: {page['title']}\npage_content:{page['content']}"
            tk = self.personality.model.tokenize(page_text)
            self.step_start(f"summerizing {page['title']}")
            if len(tk)<int(self.personality_config.zip_size):
                    page_text = self.summerize(document_chunks,"\n".join([
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
                    self.full(page_text)
            else:
                depth=0
                while len(tk)>int(self.personality_config.zip_size):
                    self.step_start(f"Comprerssing.. [depth {depth}]")
                    chunk_size = int(self.personality.config.ctx_size*0.6)
                    document_chunks = DocumentDecomposer.decompose_document(page_text, chunk_size, 0, self.personality.model.tokenize, self.personality.model.detokenize, True)
                    page_text = self.summerize(document_chunks,"\n".join([
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
                    self.full(page_text)
                    tk = self.personality.model.tokenize(page_text)
                    self.step_end(f"Comprerssing.. [depth {depth}]")
                    self.full(output+f"\n\n## Summerized chunk text:\n{page_text}")
                    depth += 1
            self.step_start(f"Last composition")
            page_text = self.summerize(document_chunks,"\n".join([
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
            self.full(page_text)

            self.step_end(f"Last composition")
            self.step_end(f"summerizing {page['title']}")
            processed_pages += f"{page['title']}\n{page_text}"

        page_text = self.summerize(processed_pages,"\n".join([
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
            callback=self.sink
            )
        self.full(page_text)

        self.step_start(f"Last composition")
        page_text = self.summerize(document_chunks,"\n".join([
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
        self.full(page_text)
        self.step_end(f"Last composition")

        if self.personality_config.output_path:
            self.save_text(page_text, Path(self.personality_config.output_path)/(page['title']+"_summary.txt"))
        return page_text, output
                    
        

    def start_scraping(self, prompt="", full_context=""):
        self.new_message("")
        if self.personality_config.search_query!="":
            self.search_and_zip(self.personality_config.search_query)
        else:
            self.info("Please put a search query in the search query setting of this personality.")



    def recover_all_rss_feeds(self, prompt="", full_context=""):
        output_folder = self.personality_config.output_folder
        if output_folder=="":
            self.personality.InfoMessage("output_folder is empty, please open the configurations of the personality and set an output path.\nThis allows me to store the data recovered from the internet so that I can recover in the future if i fail to finish.")
            return
        output_folder = Path(output_folder)
        if not output_folder.exists():
            self.personality.InfoMessage("output_folder does not exist, please open the configurations of the personality and set a valid output path.\nThis allows me to store the data recovered from the internet so that I can recover in the future if i fail to finish.")
            return
        self.new_message("")
        self.chunk("")
        if self.personality_config.rss_urls!="":
            self.step_start("Recovering rss feeds")
            rss_feeds = [feed.strip() for feed in self.personality_config.rss_urls.split(",")]
            self.step_end("Recovering rss feeds")
            links = []
            feeds = []
            nb_feeds=0
            for rss_feed in rss_feeds:
                feed = feedparser.parse(rss_feed)
                feeds.append([])
                to_remove=[]
                for p in feed.entries:
                    nb_feeds += 1
                    if nb_feeds>=self.personality_config.nb_rss_feed_pages and self.personality_config.nb_rss_feed_pages!=-1:
                        break
                    feeds[-1].append(p)
                    content = p['summary'] if 'summary' in p else p['description'] if 'description' in p else ''
                    if content!="":
                        card = f'''
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    <h3 style="margin-top: 0;">
        <a href="{p.link}" target="_blank" style="text-decoration: none; color: #333;">{p.title}</a>
    </h3>
    <p style="color: #666;">{content}</p>
</div>
                        '''
                        links.append(card)
                    else:
                        to_remove.append(p)
                for r in to_remove:
                    feed.entries.remove(r)
                if nb_feeds>=self.personality_config.nb_rss_feed_pages and self.personality_config.nb_rss_feed_pages!=-1:
                    break

            # Save
            with open(output_folder/"news_data.json","w") as f:
                feeds = [feed for feed_pack in feeds for feed in feed_pack]
                json.dump(feeds, f)
            # build output
            output = "\n".join([
                "## Internet search done:",
                "### Pages:",
            ]+links)
            ASCIIColors.yellow("Done URLs recovery")
            self.full(output)

    def fuse_articles(self, prompt="", full_context=""):
        output_folder = self.personality_config.output_folder
        if output_folder=="":
            self.personality.InfoMessage("output_folder is empty, please open the configurations of the personality and set an output path.\nThis allows me to store the data recovered from the internet so that I can recover in the future if i fail to finish.")
            return
        output_folder = Path(output_folder)
        if not output_folder.exists():
            self.personality.InfoMessage("output_folder does not exist, please open the configurations of the personality and set a valid output path.\nThis allows me to store the data recovered from the internet so that I can recover in the future if i fail to finish.")
            return
        self.new_message("")
        self.chunk("")
        self.step_start("Fusing articles")
        with open(output_folder/"news_data.json","r") as f:
            feeds = json.load(f)
        self.step_end("Fusing articles")
        total_entries = len(feeds)
        subjects = []
        processed=[]
        previous_output =""
        for index,feed in enumerate(feeds):
            if not feed in processed:
                subjects.append([feed])
                progress = (index / total_entries) * 100
                content = feed['summary'] if 'summary' in feed else feed['description'] if 'description' in feed else ''
                for second_index, second_feed in enumerate(feeds):
                    second_progress = (second_index / total_entries) * 100
                    second_content = second_feed['summary'] if 'summary' in second_feed else second_feed['description'] if 'description' in second_feed else  ''
                    answer = self.yes_no("Are those two articles talking about the same subject?",f"Article 1 :\nTitle: {feed['title']}\nContent:\n{content}\nArticle 2 :\nTitle: {second_feed['title']}\nContent:\n{second_content}\n")
                    out = f'''
<b>Processing article : {feed['title']}</b>
<div style="width: 100%; height: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
    <div style="width: {progress}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
</div>
<b>Comparing article : {second_feed['title']}</b>
<div style="width: 100%; height: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
    <div style="width: {second_progress}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
</div>    
<b>{'same' if answer else 'different'}<b><br>
'''+previous_output
                    self.full(out)
                    if answer:
                        subjects[-1].append(second_feed)
                        processed.append(feed)
                        ASCIIColors.yellow(f"{feed['title']} and {second_feed['title']} are the same.")
                previous_output = "**Fused subjects**"
                for feed in subjects[-1]:
                    content = feed['summary'] if 'summary' in feed else feed['description'] if 'description' in feed else ''
                    previous_output+=f'''
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    <h3 style="margin-top: 0;">
        <a href="{feed['link']}" target="_blank" style="text-decoration: none; color: #333;">{feed['title']}</a>
    </h3>
    <p style="color: #666;">{content}</p>
</div>
                    '''
                out = f'''
<b>Processing article : {feed['title']}</b>
<div style="width: 100%; height: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
    <div style="width: {progress}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
</div>    
'''+previous_output
                self.full(out)



    def categorize_news(self, prompt="", full_context=""):
        output_folder = self.personality_config.output_folder
        if output_folder=="":
            self.personality.InfoMessage("output_folder is empty, please open the configurations of the personality and set an output path.\nThis allows me to store the data recovered from the internet so that I can recover in the future if i fail to finish.")
            return
        output_folder = Path(output_folder)
        if not output_folder.exists():
            self.personality.InfoMessage("output_folder does not exist, please open the configurations of the personality and set a valid output path.\nThis allows me to store the data recovered from the internet so that I can recover in the future if i fail to finish.")
            return
        self.new_message("")
        self.chunk("")
        self.step_start("Categorizing articles")
        with open(output_folder/"news_data.json","r") as f:
            feeds = json.load(f)
            cats = [c.strip() for c in self.personality_config.categories.split(",")]
            categorized ={
                cat:[]
                for cat in cats
            }
            total_entries = len(feeds)
            for index,feed in enumerate(feeds):
                progress = (index / total_entries) * 100
                answer = self.multichoice_question("Determine the category that suits this article the most.", cats,f"Title: {feed['title']}\nContent:\n{feed['description'] if hasattr(feed, 'description') else ''}\n")
                categorized[cats[answer]].append(feed)
                self.full(f'''
Article classified as : {cats[answer]}
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    <h3 style="margin-top: 0;">
        <a href="{feed['link']}" target="_blank" style="text-decoration: none; color: #333;">{feed['title']}</a>
    </h3>
    <p style="color: #666;">{feed['description'] if hasattr(feed, 'description') else ''}</p>
    <div style="width: 100%; height: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
        <div style="width: {progress}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
    </div>    
</div>
                    ''')
        
        with open(output_folder/"news_data_categorized.json","w") as f:
            json.dump(categorized, f)
        self.step_end("Categorizing articles")

    def scrape_news(self, prompt="", full_context=""):
        """
        This function will search for latest news, then regroup them by category
        """
        self.recover_all_rss_feeds()
        self.fuse_articles()
        self.categorize_news()



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
        self.step_start("Understanding request")
        if self.yes_no("Is the user asking for doing internet search about a topic?", previous_discussion_text):
            self.step_end("Understanding request")
            self.personality.step_start("Crafting internet search query")
            query = self.personality.fast_gen(f"!@>discussion:\n{previous_discussion_text}\n!@>system: Read the discussion and craft a web search query suited to recover needed information to reply to last {self.personality.config.user_name} message.\nDo not answer the prompt. Do not add explanations.\n!@>current date: {datetime.now()}!@>websearch query: ", max_generation_size=256, show_progress=True, callback=self.personality.sink)
            self.personality.step_end("Crafting internet search query")

            self.personality.step_start("Scraping (this may take time, so be patient) ....")
            self.search_and_zip(query)
            self.personality.step_end("Scraping (this may take time, so be patient) ....")
        else:
            self.step_end("Understanding request")
            self.fast_gen(previous_discussion_text, callback=self.callback)
        return ""



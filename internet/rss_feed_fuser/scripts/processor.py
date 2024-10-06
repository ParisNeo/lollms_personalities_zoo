from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.utilities import PackageManager
from lollms.personality import APScript, AIPersonality
from lollms.types import MSG_OPERATION_TYPE
from lollms.internet import internet_search, scrape_and_save
from typing import Callable, Any

from lollmsvectordb.text_document_loader import TextDocumentsLoader
from lollmsvectordb.text_chunker import TextChunker
import subprocess
from pathlib import Path
from datetime import datetime
import json
from lollms.client_session import Client

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
                {"name":"nb_rss_feeds_per_source","type":"int","value":5, "help":"the maximum number of rss feed pages to search"},
                {"name":"rss_scraping_type","type":"str","value":"quick","options":["quick","deep"], "help":"quick uses only the breafs to build the summary and the deep will scrape data from the website"},
                {"name":"rss_urls","type":"text","value":"https://feeds.bbci.co.uk/news/rss.xml, http://rss.cnn.com/rss/cnn_topstories.rss, https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml, https://www.theguardian.com/world/rss, https://www.reuters.com/rssfeed/topNews, http://feeds.foxnews.com/foxnews/latest, https://www.aljazeera.com/xml/rss/all.xml, https://www.bloomberg.com/politics/feeds/site.xml", "help":"Here you can put rss feed address to recover data."},
                {"name":"categories","type":"text","value":"World News,Entertainment,Sport,Technology,Education,Medicine,Space,R&D,Politics,Music,Business,Peaple", "help":"The list of categories to help the AI organize the news."},
                {"name":"keep_only_multi_articles_subjects","type":"bool","value":False, "help":"When this option is true, only articles that have more than one source are kept"},
                
                {"name":"quick_search","type":"bool","value":False, "help":"Quick search returns only a brief summary of the webpage"},
                {"name":"summary_mode","type":"str","value":"RAG", "options":["RAG","Full Summary"], "help":"If Rag is used then the AI will search for useful data before summerizing, else it's gonna read the whole page before summary. The first is faster, but the second allows accessing the whole information without compromize."},                
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

    def generate_thumbnail_html(self, feed):
        if type(feed)==list:
            thumbnails = feed
        else:
            thumbnails = feed.get('media_thumbnail',[])
        
        thumbnail_html = ''
        for thumbnail in thumbnails:
            try:
                url = thumbnail['url']
            except:
                continue
            try:
                width = thumbnail['width']
            except:
                width = 500
            try:
                height = thumbnail['height']
            except:
                height = 200
            
            thumbnail_html += f'<img src="{url}" width="{width}" height="{height}" alt="Thumbnail" style="margin-right: 10px;">'
        
        card_html = f'''
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    {thumbnail_html}
</div>
        '''
        return card_html


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
        self.add_chunk_to_message_content("")
        if self.personality_config.rss_urls!="":
            self.step_start("Recovering rss feeds")
            rss_feeds = [feed.strip() for feed in self.personality_config.rss_urls.split(",")]
            self.step_end("Recovering rss feeds")
            links = []
            feeds = []
            for rss_feed in rss_feeds:
                nb_feeds=0
                feed = feedparser.parse(rss_feed)
                feeds.append([])
                to_remove=[]
                for p in feed.entries:
                    nb_feeds += 1
                    if nb_feeds>self.personality_config.nb_rss_feeds_per_source and self.personality_config.nb_rss_feeds_per_source!=-1:
                        break
                    feeds[-1].append(p)
                    content = p['summary'] if 'summary' in p else p['description'] if 'description' in p else ''
                    thumbnail_html = self.generate_thumbnail_html(p)
                    if content!="":
                        card = f'''
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    <h3 style="margin-top: 0;">
        <a href="{p.link}" target="_blank" style="text-decoration: none; color: #333;">{p.title}</a>
    </h3>
{thumbnail_html}    

<p style="color: #666;">{content}</p>
</div>
                        '''
                        links.append(card)
                    else:
                        to_remove.append(p)
                for r in to_remove:
                    feed.entries.remove(r)
            # Save
            with open(output_folder/"news_data.json","w") as f:
                feeds = [feed for feed_pack in feeds for feed in feed_pack]
                json.dump(feeds, f,indent=4)
            # build output
            output = "\n".join([
                "## RSS feeds recovered:",
                "### News:",
            ]+links)
            ASCIIColors.yellow("Done URLs recovery")
            self.set_message_content(output)

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
        self.add_chunk_to_message_content("")
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
                processed.append(feed)
                subjects.append([feed])
                progress = ((index+1) / total_entries) * 100
                content = feed['summary'] if 'summary' in feed else feed['brief'] if 'brief' in feed  else feed['description'] if 'description' in feed else ''
                for second_index, second_feed in enumerate(feeds):
                    if not second_feed in processed:
                        second_progress = ((second_index+1) / total_entries) * 100
                        second_content = second_feed['summary'] if 'summary' in second_feed else second_feed['brief'] if 'brief' in second_feed  else second_feed['description'] if 'description' in second_feed else ''
                        answer = self.yes_no("Based on these two articles, are they covering the same subject? Two article are talking about the same subject if they are exposing the same event or the same context. Make sure you answer the ",f"Date:{datetime.now()}\nArticle 1 :\nTitle: {feed['title']}\nContent:\n{content}\nArticle 2 :\nTitle: {second_feed['title']}\nContent:\n{second_content}\n")
                        out = f'''
<b>Processing article : {feed['title']}</b>
<div style="width: 100%; height: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
    <div style="width: {progress}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
</div>
<b>Comparing to : {second_feed['title']}</b>
<div style="width: 100%; height: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
    <div style="width: {second_progress}%; height: 100%; background-color: #4CAF50; border-radius: 5px;"></div>
</div>    
<b>{'same' if answer else 'different'}<b><br>
'''+previous_output
                        self.set_message_content(out)
                        if answer:
                            subjects[-1].append(second_feed)
                            processed.append(second_feed)
                            ASCIIColors.yellow(f"{feed['title']} and {second_feed['title']} are the same.")
                
                
                previous_output = "<b>Fused subjects</b>"
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
                self.set_message_content(out)



        try:
            if self.personality_config.keep_only_multi_articles_subjects:
                to_remove = []
                for subject in subjects:
                    if len(subject)<2:
                        to_remove.append(subject)
                for r in to_remove:
                    subjects.remove(r)
        except Exception as ex:
            ASCIIColors.warning(f"Couldn't remove some subjects: {ex}")

        themes = {}
        out = ""
        for subject_bundle in subjects:
            prompt = f"{self.config.start_header_id_template}{self.config.system_message_template}{self.config.end_header_id_template}As an authoritative figure in the designated category, provide an in-depth analysis of the document's content, summarizing key points while highlighting any notable trends, patterns, or relevant context. Consider the document's source, credibility, and potential biases, while incorporating your expertise and insights to deliver a comprehensive and engaging summary.\n"
            thumbnails = []
            urls = []
            for feed in subject_bundle:
                content = feed['summary'] if 'summary' in feed else feed['description'] if 'description' in feed else ''
                thumbnails += feed.get('media_thumbnail',[])
                urls.append(feed['link'])
                if self.personality_config.rss_scraping_type=="quick":
                    prompt+=f"Title: {feed['title']}\nContent:\n{content}\n"
                else:
                    content = scrape_and_save(feed['link'])

                    content = self.summarize_text(content,"summarize the news article. Only extract the news information, do not add iny information that does not exist in the chunk.")
                    prompt+=f"Title: {feed['title']}\nContent:\n{content}\n"

            prompt += f"Don't make any comments, just do the summary. Analyze the content of the snippets and give a clear verified and elegant article summary.\nOnly report information from the snippet.\nDon't add information that is not found in the chunks.\nDon't add any dates that are not explicitely reported in the documents.{self.config.separator_template}{self.config.start_header_id_template}Today date:{datetime.now()}{self.config.separator_template}{self.config.start_header_id_template}summary:\n"
            gen = self.fast_gen(prompt, callback=self.sink)
                
            title = self.fast_gen(f"{self.config.start_header_id_template}{self.config.system_message_template}{self.config.end_header_id_template}Generate a concise yet eye catching title for this article.\nInfo: No comments, just provide a comprehensive and informative summary.{self.config.separator_template}{self.config.start_header_id_template}Today date:{datetime.now()}{self.config.separator_template}{self.config.start_header_id_template}content:{gen}{self.config.separator_template}{self.config.start_header_id_template}title:", callback=self.sink)
            themes['title']={
                'title':title,
                'thumbnails':thumbnails,
                'content':gen
            }
            thumbnail_html = self.generate_thumbnail_html(thumbnails)
            card = f'''
<div style="width: 100%; border: 1px solid #ccc; border-radius: 5px; padding: 20px; font-family: Arial, sans-serif; margin-bottom: 20px; box-sizing: border-box;">
    <h3 style="margin-top: 0;">
        <a href="{urls[0]}" target="_blank" style="text-decoration: none; color: #333;">{title}</a>
    </h3>
{thumbnail_html}    
<p style="color: #666;">{gen}</p>
<h4>Sources:</h4>
'''
            for url in urls:
                 card+=f'''
<a href="{url}" target="_blank" style="text-decoration: none; color: #333;">{url}</a>
'''
            card+=f'''
</div>
'''
            out += self.build_a_folder_link(self.personality_config.output_folder,"Open output folder")
            out +=card
            self.set_message_content(out)
        out = "<html><header></header><body>"+"\n"+out+"</body><html>"
        with open(output_folder/"news.html","w") as f:
            f.write(out)

        with open(output_folder/"fused.json","w") as f:
            json.dump(themes,f,indent=4)

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
        self.add_chunk_to_message_content("")
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
                progress = ((index+1) / total_entries) * 100
                answer = self.multichoice_question("Determine the category that suits this article the most.", cats,f"Title: {feed['title']}\nContent:\n{feed['description'] if hasattr(feed, 'description') else ''}\n")
                categorized[cats[answer]].append(feed)
                self.set_message_content(f'''
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
            json.dump(categorized, f,indent=4)
        self.step_end("Categorizing articles")

    def scrape_news(self, command, full_context, callback, context_state, client:Client):
        """
        This function will search for latest news, then regroup them by category
        """
        self.recover_all_rss_feeds()
        self.fuse_articles()
        self.categorize_news()


    def search_and_zip(self, query,  output =""):
        self.step_start("Performing internet search")
        self.add_chunk_to_message_content("")
        pages = internet_search(query, self.personality_config.internet_nb_search_pages, buttons_to_press=self.personality_config.buttons_to_press, quick_search=self.personality_config.quick_search)
        processed_pages = ""
        if len(pages)==0:
            self.set_message_content("Failed to do internet search!!")
            self.step_end("Performing internet search",False)
            return
        self.step_end("Performing internet search")
        for page in pages:
            if self.personality_config.quick_search:
                page_text = f"page_title: {page['title']}\npage_brief:{page['brief']}"
            else:
                page_text = f"page_title: {page['title']}\npage_content:{page['content']}"
            tk = self.personality.model.tokenize(page_text)
            self.step_start(f"summerizing {page['title']}")
            if len(tk)<int(self.personality_config.zip_size) or self.personality_config.summary_mode!="RAG":
                page_text = self.summarize_text(page_text,"\n".join([
                                f"Extract from the document any information related to the query. Write the output as a short article.",
                                "The summary should contain exclusively information from the document chunk.",
                                "Do not provide opinions nor extra information that is not in the document chunk",
                                f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                                f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                                f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                                f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                                f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                                f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                                f"{'The article should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
                                f"{self.config.start_header_id_template}query: {query}"
                            ]),
                            "Document chunk"
                            )
                self.set_message_content(page_text)
            else:
                chunks = self.vectorize_and_query(page['content'], page['title'], page['url'], query)
                content = "\n".join([c.text for c in chunks])
                page_text = f"page_title:\n{page['title']}\npage_content:\n{content}"
                page_text = self.summarize_text(page_text,"\n".join([
                        f"Extract from the document any information related to the query. Write the output as a short article.",
                        "The summary should contain exclusively information from the document chunk.",
                        "Do not provide opinions nor extra information that is not in the document chunk",
                        f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                        f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                        f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                        f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                        f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                        f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                        f"{'The article should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
                        f"{self.config.start_header_id_template}query: {query}"
                    ]),
                    "Document chunk"
                    )
                self.set_message_content(page_text)
            self.set_message_content(page_text)

            self.step_end(f"Last composition")
            self.step_end(f"summerizing {page['title']}")
            processed_pages += f"{page['title']}\n{page_text}"

        page_text = self.summarize_text(processed_pages,"\n".join([
                f"Extract from the document any information related to the query. Write the output as a short article.",
                "The summary should contain exclusively information from the document chunk.",
                "Do not provide opinions nor extra information that is not in the document chunk",
                f"{'Keep the same language.' if self.personality_config.keep_same_language else ''}",
                f"{'Preserve the title of this document if provided.' if self.personality_config.preserve_document_title else ''}",
                f"{'Preserve author names of this document if provided.' if self.personality_config.preserve_authors_name else ''}",
                f"{'Preserve results if presented in the chunk and provide the numerical values if present.' if self.personality_config.preserve_results else ''}",
                f"{'Eliminate any useless information and make the summary as short as possible.' if self.personality_config.maximum_compression else ''}",
                f"{self.personality_config.contextual_zipping_text if self.personality_config.contextual_zipping_text!='' else ''}",
                f"{'The summary should be written in '+self.personality_config.translate_to if self.personality_config.translate_to!='' else ''}"
                f"{self.config.start_header_id_template}query: {query}"
            ]),
            "Document chunk",
            callback=self.sink
            )
        self.set_message_content(page_text)

        self.step_start(f"Last composition")
        page_text = self.summarize_text(page_text,"\n".join([
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
        self.set_message_content(page_text)
        self.step_end(f"Last composition")

        if self.personality_config.output_path:
            self.save_text(page_text, Path(self.personality_config.output_path)/(page['title']+"_summary.txt"))
        return page_text, output
                    

    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str | list | None, MSG_OPERATION_TYPE, str, AIPersonality| None], bool]=None, context_details:dict=None, client:Client=None):
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
        self.step_start("Understanding request")
        if self.yes_no("Is the user asking for information that requires verification using internet search?", previous_discussion_text):
            self.step_end("Understanding request")
            self.step("Decided to make an internet search")
            self.personality.step_start("Crafting internet search query")
            query = self.personality.fast_gen(f"{self.config.start_header_id_template}discussion:\n{previous_discussion_text}{self.config.separator_template}{self.config.start_header_id_template}{self.config.system_message_template}{self.config.end_header_id_template}Read the discussion and craft a web search query suited to recover needed information to reply to last {self.personality.config.user_name} message.\nDo not answer the prompt. Do not add explanations.{self.config.separator_template}{self.config.start_header_id_template}current date: {datetime.now()}{self.config.separator_template}{self.config.start_header_id_template}websearch query: ", max_generation_size=256, show_progress=True, callback=self.personality.sink).split("\n")[0]
            self.personality.step("Query: "+query)
            self.personality.step_end("Crafting internet search query")

            self.personality.step_start("Scraping (this may take time, so be patient) ....")
            self.search_and_zip(query)
            self.personality.step_end("Scraping (this may take time, so be patient) ....")
        else:
            self.step_end("Understanding request")
            self.fast_gen(previous_discussion_text, callback=self.callback)
        return ""



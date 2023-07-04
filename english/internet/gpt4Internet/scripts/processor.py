from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality

from pathlib import Path
import subprocess

def format_url_parameter(value:str):
    encoded_value = value.strip().replace("\"","")
    return encoded_value

def extract_results(url, max_num, chromedriver_path=None):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from bs4 import BeautifulSoup

    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode

    # Set path to chromedriver executable (replace with your own path)
    if chromedriver_path is None: 
        chromedriver_path = ""#"/snap/bin/chromium.chromedriver"    

    # Create a new Chrome webdriver instance
    try:
        driver = webdriver.Chrome(executable_path=chromedriver_path, options=chrome_options)
    except:
        driver = webdriver.Chrome(options=chrome_options)

    # Load the webpage
    driver.get(url)

    # Wait for JavaScript to execute and get the final page source
    html_content = driver.page_source

    # Close the browser
    driver.quit()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Detect that no outputs are found
    Not_found = soup.find("No results found")

    if Not_found : 
        return []    

    # Find the <ol> tag with class="react-results--main"
    ol_tag = soup.find("ol", class_="react-results--main")

    # Initialize an empty list to store the results
    results_list = []

    # Find all <li> tags within the <ol> tag
    li_tags = ol_tag.find_all("li")

    # Loop through each <li> tag, limited by max_num
    for index, li_tag in enumerate(li_tags):
        if index > max_num:
            break

        try:
            # Find the three <div> tags within the <article> tag
            div_tags = li_tag.find_all("div")

            # Extract the link, title, and content from the <div> tags
            links = div_tags[0].find_all("a")
            href_value = links[1].get('href')
            span = links[1].find_all("span")
            link = span[0].text.strip()

            title = div_tags[2].text.strip()
            content = div_tags[3].text.strip()

            # Add the extracted information to the list
            results_list.append({
                "link": link,
                "href": href_value,
                "title": title,
                "content": content
            })
        except Exception:
            pass

    return results_list

   
class Processor(APScript):
    """
    A class that processes model inputs and outputs.

    Inherits from APScript.
    """

    def __init__(
                 self, 
                 personality: AIPersonality
                ) -> None:
        self.queries=[]
        self.formulations=[]
        self.summaries=[]
        self.word_callback = None
        self.generate_fn = None
        
        personality_config = TypedConfig(
            ConfigTemplate([
                {"name":"chromedriver_path","type":"str","value":""},
                {"name":"num_results","type":"int","value":3, "min":2, "max":100},
                {"name":"max_query_size","type":"int","value":50, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_summery_size","type":"int","value":256, "min":10, "max":personality.model.config["ctx_size"]},
                
            ]),
            BaseConfig(config={
                'chromedriver_path'         : "",
                'num_results'               : 3,
                'max_query_size'            : 50,
                'max_summery_size'          : 256
            })
        )
        super().__init__(
                            personality,
                            personality_config
                        )
        
    def install(self):
        super().install()
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])        
        ASCIIColors.success("Installed successfully")


     
    def internet_search(self, query):
        """
        Perform an internet search using the provided query.

        Args:
            query (str): The search query.

        Returns:
            dict: The search result as a dictionary.
        """
        formatted_text = ""
        results = extract_results(f"https://duckduckgo.com/?q={format_url_parameter(query)}&t=h_&ia=web", self.personality_config.num_results, self.personality_config.chromedriver_path)
        for i, result in enumerate(results):
            title = result["title"]
            content = result["content"]
            link = result["link"]
            href = result["href"]
            formatted_text += f"index: {i+1}\nsource: {href}\ntitle: {title}\n"

        print("Searchengine results : ")
        print(formatted_text)
        return formatted_text, results

    def run_workflow(self, prompt, previous_discussion_text="", callback=None):
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
        self.word_callback = callback

        # 1 first ask the model to formulate a query
        search_formulation_prompt = f"""!!> Instructions:
Formulate a search query text out of the user prompt.
Keep all important information in the query and do not add unnecessary text.
Write a short query.
Do not explain the query.
## question:
{prompt}
### search query:
"""
        search_query = format_url_parameter(self.generate(search_formulation_prompt, self.personality_config.max_query_size)).strip()
        if search_query=="":
            search_query=prompt
        if callback is not None:
            callback("Crafted search query :"+search_query+"\nSearching...", MSG_TYPE.MSG_TYPE_FULL)
        search_result, results = self.internet_search(search_query)
        if callback is not None:
            callback("Crafted search query :"+search_query+"\nSearching... OK\nSummerizing...", MSG_TYPE.MSG_TYPE_FULL)
        prompt = f"""!!> Instructions:
Use Search engine results to answer user question by summerizing the results in a single coherant paragraph in form of a markdown text with sources citation links in the format [index](source).
Place the citation links in front of each relevant information.
### search results:
{search_result}
### question:
{prompt}
## answer:
"""
        print(prompt)
        output = self.generate(prompt, self.personality_config.max_summery_size)
        sources_text = "\n# Sources :\n"
        for result in results:
            link = result["link"]
            href = result["href"]
            sources_text += f"[source : {link}]({href})\n\n"

        output = output+sources_text
        if callback is not None:
            callback(output, MSG_TYPE.MSG_TYPE_FULL)

        return output




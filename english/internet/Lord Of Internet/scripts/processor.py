from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.types import MSG_TYPE
from lollms.personality import APScript, AIPersonality


from pathlib import Path
import subprocess
import re

def format_url_parameter(value:str):
    encoded_value = value.strip().replace("\"","")
    return encoded_value


def get_relevant_text_block(
                                url, 
                                question, 
                                max_nb_chunks=2, 
                                max_global_size=512, 
                                max_chunk_size=256, 
                                overlap_size=50,
                                callback=None):
    import requests
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if callback:
        callback("Recovering data", MSG_TYPE.MSG_TYPE_STEP_START)
        
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    # Example: Remove all <script> and <style> tags
    for script in soup(["script", "style"]):
        script.extract()

    all_text = soup.get_text()
    # Example: Remove leading/trailing whitespace and multiple consecutive line breaks
    all_text = ' '.join(all_text.strip().splitlines())
    if callback:
        callback("Recovering data", MSG_TYPE.MSG_TYPE_STEP_END)

    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)[\.\?\s]{2,}', all_text)


    # Chunk the sentences into consistent blocks with overlap
    chunks = []
    chunk=""
    for i in range(len(sentences)):
        sentence_size = len(sentences[i])
        current_chunk_size = len(chunk)+sentence_size
        if current_chunk_size<max_chunk_size:
            chunk+=sentences[i]
        else:
            chunks.append(chunk)
            
            chunk = ""
            ol = []
            j=0
            while len(chunk)<overlap_size and i-j>=0:
                ol.append(sentences[i-j])
                j+=1
                chunk=chunk+sentences[i-j]+".\n"
            chunk += ".\n".join(reversed(ol))
    if chunk!="":
        chunks.append(chunk)

    if len(chunks)==1:
        return chunks[0]
    else:
        if callback:
            callback("Vectorizing data", MSG_TYPE.MSG_TYPE_STEP_START)
        # Vectorize the chunks
        vectorizer = TfidfVectorizer()
        vectorized_text = vectorizer.fit_transform(chunks)

        if callback:
            callback("Vectorizing data", MSG_TYPE.MSG_TYPE_STEP_END)

        if callback:
            callback("Searching relevant data", MSG_TYPE.MSG_TYPE_STEP_START)

        # Vectorize the question
        vectorized_question = vectorizer.transform([question])

        # Calculate document similarity
        similarity_scores = cosine_similarity(vectorized_text, vectorized_question)


        # Retrieve relevant text chunks based on similarity threshold
        relevant_chunks = []
        # Sort similarity scores in descending order and get the indices of top n scores
        top_n_indices = similarity_scores.argsort(axis=0)[-max_nb_chunks:][::-1]

        # Retrieve the corresponding chunks
        for index in top_n_indices:
            relevant_chunks.append(chunks[index[0]])

        if callback:
            callback("Searching relevant data", MSG_TYPE.MSG_TYPE_STEP_END)

        if callback:
            callback("Preprocessing data", MSG_TYPE.MSG_TYPE_STEP_START)

        # Cap the relevant chunks to not exceed max_global_size
        capped_relevant_chunks = []
        current_size = 0
        for chunk in relevant_chunks:
            chunk_size = len(chunk)
            if current_size + chunk_size <= max_global_size:
                capped_relevant_chunks.append(chunk)
                current_size += chunk_size
            else:
                break

        # Concatenate relevant text chunks into a single text block
        relevant_text_block = ' '.join(capped_relevant_chunks)
        if callback:
            callback("Preprocessing data", MSG_TYPE.MSG_TYPE_STEP_END)

        return relevant_text_block





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

    try:
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
    except:
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
        template = ConfigTemplate([
                {"name":"craft_search_query","type":"bool","value":False},
                {"name":"chromedriver_path","type":"str","value":""},
                {"name":"num_results","type":"int","value":5, "min":2, "max":100},
                {"name":"max_query_size","type":"int","value":50, "min":10, "max":personality.model.config["ctx_size"]},
                {"name":"max_summery_size","type":"int","value":256, "min":10, "max":personality.model.config["ctx_size"]},
            ])
        config = BaseConfig.from_template(template)
        personality_config = TypedConfig(
            template,
            config
        )
        super().__init__(
                            personality,
                            personality_config
                        )
        
        #Now try to import stuff to verify that installation succeeded
        import requests
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from bs4 import BeautifulSoup
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
    def install(self):
        super().install()
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])        
        ASCIIColors.success("Installed successfully")

    def uninstall(self):
        super().uninstall()

     
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
            href = result["href"]
            content = get_relevant_text_block(href, query, callback=self.word_callback)
            formatted_text += f"link:{result['href']}\n"+"content:"+content+"\n"

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
        if self.personality_config.craft_search_query:
            # 1 first ask the model to formulate a query
            search_formulation_prompt = f"""### Instructions:
    Formulate a search query text out of the user prompt.
    Keep all important information in the query and do not add unnecessary text.
    Write a short query.
    Do not explain the query.
    ## question:
    {prompt}
    ### search query:
    """
            if callback is not None:
                callback("Crafting search query", MSG_TYPE.MSG_TYPE_STEP_START)
            search_query = format_url_parameter(self.generate(search_formulation_prompt, self.personality_config.max_query_size)).strip()
            if search_query=="":
                search_query=prompt
            if callback is not None:
                callback("Crafting search query", MSG_TYPE.MSG_TYPE_STEP_END)
        else:
            search_query = prompt
            
        search_result, results = self.internet_search(search_query)
        prompt = f"""### Instructions:
Use Search engine results to answer user question by summerizing the results in a single coherant paragraph in form of a markdown text with sources citation links in the format [index](source).
Place the citation links in front of each relevant information.
Citation is mandatory.
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




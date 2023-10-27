from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.personality import APScript, AIPersonality
from safe_store import GenericDataLoader
from safe_store import TextVectorizer, VectorizationMethod, VisualizationMethod
from pathlib import Path
import json

def find_last_file(folder_path):
    i = 0
    while True:
        file_name = f"database_{i}.json"
        file_path = Path(folder_path) / file_name
        if not file_path.exists():
            return str(f"database_{i-1}.json")
        i += 1


def find_available_file(folder_path):
    i = 0
    while True:
        file_name = f"database_{i}.json"
        file_path = Path(folder_path) / file_name
        if not file_path.exists():
            return str(file_path)
        i += 1

class Processor(APScript):
    """
    A class that processes model inputs and outputs.
    Inherits from APScript.
    """

    def __init__(
        self,
        personality: AIPersonality,
        callback=None,
    ) -> None:
        # Get the current directory
        root_dir = personality.lollms_paths.personal_path
        # We put this in the shared folder in order as this can be used by other personalities.
        shared_folder = root_dir / "shared"
        self.callback = None
        personality_config_template = ConfigTemplate(
            [
                {
                    "name": "data_folder_path",
                    "type": "str",
                    "value": "",
                    "help": "A path to a local folder containing the original data to be converted to a chat database",
                },
                {
                    "name": "questions_gen_size",
                    "type": "int",
                    "value": 512,
                    "help": "The maximum number of tokens that can be generated for each chunk of text in the questions building phase",
                },
                {
                    "name": "answer_gen_size",
                    "type": "int",
                    "value": 1024,
                    "help": "The maximum number of tokens that can be generated for each chunk of text in the questions building phase",
                },
                {
                    "name": "data_chunk_size",
                    "type": "int",
                    "value": 512,
                    "help": "The maximum number of tokens that for each vectorized data chunks",
                },
                {
                    "name": "data_overlap_size",
                    "type": "int",
                    "value": 128,
                    "help": "The overlap between data chunks in tokens",
                },
                {
                    "name": "data_vectorization_nb_chunks",
                    "type": "int",
                    "value": 2,
                    "help": "The number of chunks to recover from the database",
                },
                {
                    "name": "use_enhanced_mode",
                    "type": "bool",
                    "value": False,
                    "help": "This activates using the keyword building part of the execution diagram. Please read the paper for more details.",
                },
                
                
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
        self.data_store = TextVectorizer(
            vectorization_method=VectorizationMethod.TFIDF_VECTORIZER,  # =VectorizationMethod.BM25_VECTORIZER,
            data_visualization_method=VisualizationMethod.PCA,  # VisualizationMethod.PCA,
            save_db=False
        )

    def run_workflow(self, prompt, previous_discussion_text="", callback=None):
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
        # Preparing callback
        self.callback = callback
        
        # Preparing data
        output_folder = self.personality.lollms_paths.personal_outputs_path/self.personality.name
        output_folder.mkdir(parents=True, exist_ok=True)
        # Verify if the data_folder_path exists
        data_folder_path = Path(self.personality_config.data_folder_path)
        if not Path(data_folder_path).exists():
            self.warning("The specified data_folder_path does not exist.")
        document_files = [v for v in data_folder_path.iterdir()]
        self.step_start(f"Loading files")
        for file_path in document_files:
            document_text = GenericDataLoader.read_file(file_path)
            self.data_store.add_document(file_path, document_text, chunk_size=512, overlap_size=128)
        self.step_end(f"Loading files")
        # Index the vector store
        self.step_start(f"Indexing files")
        self.data_store.index()
        self.step_end(f"Indexing files")
        
        #processing
        if "continue" in prompt.lower():
            try:
                db_name = find_last_file(output_folder)
                with open(output_folder/f"{db_name.split('.')[0]}_q.json", 'r') as file:
                    questions_vector = json.load(file)
            except:
                output = "FAILED to continue from last process: "
                self.full(output)
                return
            output = "### Loading questions:\n"
            output += "\n".join(questions_vector)
        else:
            db_name = find_available_file(output_folder)
            output = "### Building questions:\n"
            self.full(output)
            # Iterate over all documents in data_folder_path
            processed_chunks = 0
            # Iterate over all chunks and extract text
            questions_vector = []
            total_chunks = len(self.data_store.chunks.items())
            for chunk_name, chunk in self.data_store.chunks.items():
                chunk_text = chunk["chunk_text"]
                processed_chunks += 1
                self.step_start(f"Processing chunk {chunk_name}: {processed_chunks}/{total_chunks}")
                # Build the prompt text with placeholders
                prompt_text = "!@>instruction: Generate questions or tasks that delve into the specific details and information presented in the text chunks. Please do not ask questions about the form of the text, and do not mension the text itself in your questions. Start each question with -. Do not add question enumeration.\n\n!@>chunk {{chunk_name}}: {{chunk}}\n!@>Here are some questions to explore the content of the text chunk. I will only present the questions without answering them:\n- "
                # Ask AI to generate questions
                generated_text = "- "+self.fast_gen(prompt_text, max_generation_size=self.personality_config.questions_gen_size, placeholders={"chunk": chunk_text, "chunk_name":chunk_name}, debug=True)
                # Split the generated text into lines and accumulate into questions_vector
                generated_lines = [q[2:] if q.startswith("- ") else q for q in generated_text.strip().split("\n")]
                questions_vector.extend(generated_lines)
                self.step_end(f"Processing chunk {chunk_name}: {processed_chunks}/{total_chunks}")
                output += generated_text + "\n"
                self.full(output)
            
            self.step_start(f"Saving questions for future use")
            with open(output_folder/f"{db_name.split('.')[0]}_q.json", 'w') as file:
                json.dump(questions_vector, file)
            self.step_end(f"Saving questions for future use")
        
        output += "### Building answers:\n"
        self.full(output)
        qna_list=[]
        # Perform further processing with questions_vector
        for index, question in enumerate(questions_vector):
            docs, sorted_similarities = self.data_store.recover_text(question, top_k=self.personality_config.data_vectorization_nb_chunks) 
            if self.personality_config.use_enhanced_mode:
                self.step_start(f"Verifying RAG data_{index}")
                prompt_text = """!@>chunk: {{chunk}}
!@>instruction: Is the information provided in the above chunk sufficient to answer the following question?
Valid answers:
- Yes
- No
!@>question: {{question}}
!@>answer: """
                if "yes" not in prompt_text.lower():
                    self.step_end(f"Verifying RAG data_{index}", False)
                    continue
                self.step_end(f"Verifying RAG data_{index}")

            self.step_start(f"Asking question {index}/{len(questions_vector)}")
            prompt_text = """!@>chunk: {{chunk}}
!@>instructions:
Interpret the textual data contained within the chunk thoroughly to answer the corresponding instruction/task presented alongside it.
If the information stored in this chunk does not suffice to provide categorically accurate answers, please indicate accordingly by stating "insufficient information".
All statements must be generated solely based on the available input data, discarding any assumptions beyond what has been explicitly stated. 
It is crucial to maintain strict adherence to the content delineated in each instance of interaction.
!@>question: {{question}}
!@>answer: """
            # !@>chunk: {{chunk}}\n!@>instruction: Please use the text chunks to answer the following question:\n\n!@>question: {{question}}\n\n!@>answer: "
            # Ask AI to generate an answer
            answer = self.fast_gen(prompt_text, max_generation_size=self.personality_config.answer_gen_size, placeholders={"chunk": "\nchunk: ".join(docs), "question": question})
            qna_list.append({
                "conditionning":"Act as LoLLMs expert and answer the following questions.",
                "question":question,
                "answer":answer,
                "id":0
            })
            output += f"q:{question}\na:{answer}\n"
            self.full(output)
            self.step_end(f"Asking question {index}/{len(questions_vector)}")
            with open(output_folder/db_name, 'w') as file:
                json.dump(qna_list, file)
        print("Dictionary saved as JSON successfully!")
        return ""




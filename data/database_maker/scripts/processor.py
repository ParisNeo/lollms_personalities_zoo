import subprocess
from lollms.helpers import ASCIIColors, trace_exception
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate, InstallOption
from lollms.personality import APScript, AIPersonality
from lollms.utilities import GenericDataLoader
from safe_store import TextVectorizer, VectorizationMethod, VisualizationMethod
from pathlib import Path


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
        self.callback = callback
        # Verify if the data_folder_path exists
        data_folder_path = Path(self.personality_config.data_folder_path)
        if not Path(data_folder_path).exists():
            self.warning("The specified data_folder_path does not exist.")
        # Iterate over all documents in data_folder_path
        document_files = [v for v in data_folder_path.iterdir()]
        total_chunks = len(document_files)
        processed_chunks = 0
        self.step_start(f"Loading files")
        for file_path in document_files:
            document_text = GenericDataLoader.read_file(file_path)
            self.data_store.add_document(file_path, document_text, chunk_size=512, overlap_size=128)
        self.step_end(f"Loading files")
        # Index the vector store
        self.step_start(f"Indexing files")
        self.data_store.index()
        self.step_end(f"Indexing files")
        # Iterate over all chunks and extract text
        questions_vector = []
        for chunk_name, chunk in self.data_store.chunks.items():
            chunk_text = chunk["chunk_text"]
            processed_chunks += 1
            self.step_start(f"Processing chunk {chunk_name}: {processed_chunks}/{total_chunks}")
            # Build the prompt text with placeholders
            prompt_text = f"###>instruction: Please formulate a set of questions that can be answered by reading the following chunk of text:\n\n###>chunk: {{chunk}}\n###>Here is an a set of questions that can be answered using the chunk of text you presented:\n- What is the topic of the text?\n- "
            # Ask AI to generate questions
            generated_text = "- "+self.fast_gen(prompt_text, max_generation_size=self.personality_config.questions_gen_size, placeholders={"chunk": chunk_text})
            # Split the generated text into lines and accumulate into questions_vector
            generated_lines = generated_text.strip().split("\n")
            questions_vector.extend(generated_lines)
            self.step_end(f"Processing chunk {processed_chunks}/{total_chunks}")
        # Perform further processing with questions_vector
        return ""




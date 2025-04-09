# -*- coding: utf-8 -*-
"""
ArtBot Personality for lollms

This personality is designed to generate images using a Text-to-Image (TTI) service
like Automatic1111's Stable Diffusion web UI. It guides the user through prompt
creation, style selection, and image generation, leveraging the new UI formatting
capabilities of lollms.
"""
import subprocess
import re
import webbrowser
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional, Union

from PIL import Image

# Lollms imports
from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.types import MSG_OPERATION_TYPE
from lollms.personality import APScript, AIPersonality
from lollms.utilities import PromptReshaper, discussion_path_to_url, find_first_available_file_index, add_callback
from lollms.functions.prompting.image_gen_prompts import get_image_gen_prompt, get_random_image_gen_prompt
from lollms.client_session import Client
from lollms.prompting import LollmsContextDetails

# Constants
DEFAULT_SD_ADDRESS = "http://127.0.0.1:7860"
DEFAULT_NEGATIVE_PROMPT = "(((ugly))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), ((extra arms)), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), ((watermark)), ((robot eyes))"
SUPPORTED_SAMPLERS = [
    "Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE",
    "DPM++ 2M SDE", "DPM fast", "DPM adaptive", "DPM Karras", "DPM2 Karras", "DPM2 a Karras",
    "DPM++ 2S a Karras", "DPM++ 2M Karras", "DPM++ SDE Karras", "DPM++ 2M SDE Karras", "DDIM",
    "PLMS", "UniPC", "DPM++ 3M SDE", "DPM++ 3M SDE Karras", "DPM++ 3M SDE Exponential"
]
PRODUCTION_TYPES = [
    "a photo", "an artwork", "a drawing", "a painting", "a hand drawing", "a design",
    "a presentation asset", "a presentation background", "a game asset", "a game background", "an icon"
]

class Processor(APScript):
    """
    ArtBot Processor class for handling image generation tasks.

    Inherits from APScript and implements the logic for interacting with users,
    generating prompts, and interfacing with the TTI service.
    """

    # Class attributes
    binding_name = "ArtBot"
    category = "Art"
    author = "ParisNeo"
    version = "2.0.0"
    personality_description = """
    ArtBot is a personality designed to generate images using a Text-to-Image (TTI)
    service like Stable Diffusion. It helps users craft prompts, select styles,
    and generate artwork, photos, designs, and more. It now uses the enhanced
    UI features for displaying images and information.
    """
    personality_package_path = Path(__file__).parent.parent
    dependencies = ["Pillow", "requests"] # Assuming TTI client handles its own specific dependencies

    def __init__(
        self,
        personality: AIPersonality,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Initializes the Processor instance.

        Args:
            personality (AIPersonality): The personality instance containing configuration and paths.
            callback (Optional[Callable]): A callback function for asynchronous operations.
        """
        self.callback = callback
        self.personality = personality
        self.config = self._configure(personality)
        super().__init__(
            personality,
            self.config,
            [
                {
                    "name": "idle",
                    "commands": {  # list of commands
                        "help": self.help,
                        "new_image": self.new_image,
                        "regenerate": self.regenerate,
                        "show_settings": self.show_settings,
                    },
                    "default": self.main_process
                },
            ],
            callback=callback
        )
        self.width = int(self.personality_config.width)
        self.height = int(self.personality_config.height)
        self.tti = None # Will be set/checked in `prepare`

    def _configure(self, personality: AIPersonality) -> TypedConfig:
        """
        Sets up the configuration template and initializes the TypedConfig.

        Args:
            personality (AIPersonality): The personality instance.

        Returns:
            TypedConfig: The configured TypedConfig object.
        """
        # Determine paths
        root_dir = personality.lollms_paths.personal_path
        shared_folder = root_dir / "shared"
        self.sd_folder = shared_folder / "auto_sd"
        self.sd_models_folder = self.sd_folder / "models" / "Stable-diffusion"

        # Check for SD models
        sd_models = ["Not installed"]
        if self.sd_models_folder.exists() and self.sd_models_folder.is_dir():
            models = [f.stem for f in self.sd_models_folder.iterdir() if f.is_file() and f.suffix in ['.pth', '.pt', '.ckpt', '.safetensors']]
            if models:
                sd_models = models

        # Define configuration template
        personality_config_template = ConfigTemplate([
            {"name": "activate_discussion_mode", "type": "bool", "value": True, "help": "If active, the AI will chat until explicitly asked to generate an image."},
            {"name": "examples_extraction_method", "type": "str", "value": "random", "options": ["random", "rag_based", "None"], "help": "Method to select prompt examples (None, random, or RAG-based)."},
            {"name": "number_of_examples_to_recover", "type": "int", "value": 3, "help": "Number of examples to provide to the AI."},

            {"name": "production_type", "type": "str", "value": "an artwork", "options": PRODUCTION_TYPES, "help": "Type of graphics the AI should produce."},
            {"name": "sd_model_name", "type": "str", "value": sd_models[0], "options": sd_models, "help": "Stable Diffusion model to use."},
            {"name": "sd_address", "type": "str", "value": DEFAULT_SD_ADDRESS, "help": "Address of the Stable Diffusion service (e.g., Automatic1111 web UI)."},
            {"name": "sampler_name", "type": "str", "value": "DPM++ 3M SDE", "options": SUPPORTED_SAMPLERS, "help": "Sampler for the diffusion process."},
            {"name": "steps", "type": "int", "value": 40, "min": 10, "max": 1024, "help":"Number of generation steps."},
            {"name": "scale", "type": "float", "value": 5.0, "min": 0.1, "max": 100.0, "help":"Guidance scale (CFG scale)."},

            {"name": "imagine", "type": "bool", "value": True, "help": "Generate the image after crafting the prompt."},
            {"name": "build_title", "type": "bool", "value": True, "help": "Generate a title for the artwork."},
            {"name": "paint", "type": "bool", "value": True, "help": "Execute the image generation."}, # Redundant with imagine? Clarify purpose. Seems like 'imagine' is prompt gen, 'paint' is actual generation.
            {"name": "use_fixed_negative_prompts", "type": "bool", "value": True, "help": "Use a predefined negative prompt."},
            {"name": "fixed_negative_prompts", "type": "str", "value": DEFAULT_NEGATIVE_PROMPT, "help": "The fixed negative prompt to use if the above is checked."},
            {"name": "show_infos", "type": "bool", "value": True, "help": "Show generation parameters and info after completion."},
            {"name": "continuous_discussion", "type": "bool", "value": True, "help": "Use previous discussion context for generating the next image."},
            {"name": "automatic_resolution_selection", "type": "bool", "value": False, "help": "Let ArtBot choose the image resolution."},
            {"name": "add_style", "type": "bool", "value": False, "help": "Let ArtBot choose and add a specific style to the prompt."},

            {"name": "continue_from_last_image", "type": "bool", "value": False, "help": "Use the last generated/uploaded image as input for img2img."},
            {"name": "img2img_denoising_strength", "type": "float", "value": 0.75, "min": 0.01, "max": 1.0, "help": "Img2img denoising strength (if using continue_from_last_image)."},
            {"name": "restore_faces", "type": "bool", "value": True, "help": "Enable face restoration if supported by the TTI service."},
            {"name": "caption_received_files", "type": "bool", "value": False, "help": "Automatically generate captions for uploaded files using BLIP."},

            {"name": "width", "type": "int", "value": 1024, "min": 64, "max": 4096, "help":"Default image width."},
            {"name": "height", "type": "int", "value": 1024, "min": 64, "max": 4096, "help":"Default image height."},

            # Removed thumbneil_ratio as it's related to the old HTML generation
            # {"name":"thumbneil_ratio","type":"int","value":2, "min":1, "max":5},

            # Automatic image size is covered by automatic_resolution_selection
            # {"name":"automatic_image_size","type":"bool","value":False,"help":"If true, artbot will select the image resolution"},

            {"name": "skip_grid", "type": "bool", "value": True, "help": "Skip generating a grid image if the TTI service supports it."},
            {"name": "batch_size", "type": "int", "value": 1, "min": 1, "max": 100, "help": "Number of images per batch (higher needs more VRAM)."},
            {"name": "num_images", "type": "int", "value": 1, "min": 1, "max": 100, "help": "Total number of images/batches to generate."},
            {"name": "seed", "type": "int", "value": -1, "help": "Generation seed (-1 for random)."},
            {"name": "max_generation_prompt_size", "type": "int", "value": 512, "min": 10, "max": personality.config["ctx_size"], "help": "Max tokens for LLM generating the prompts."},
        ])

        personality_config_vals = BaseConfig.from_template(personality_config_template)
        return TypedConfig(personality_config_template, personality_config_vals)

    def get_css(self) -> str:
        """
        Returns the CSS link for styling the personality's output.
        Uses Tailwind CSS for styling.
        """
        return '<link rel="stylesheet" href="/personalities/art/artbot/assets/tailwind.css">'

    def print_prompt(self, title: str, prompt: str) -> None:
        """
        Prints a formatted prompt to the console for debugging.

        Args:
            title (str): The title for the prompt section.
            prompt (str): The prompt text to print.
        """
        ASCIIColors.red(f"*-*-*-*-*-*-*-* {title} *-*-*-*-*-*-*-*", end="")
        ASCIIColors.yellow(prompt)
        ASCIIColors.red(" *-*-*-*-*-*-*-*")

    def install(self) -> None:
        """
        Installs dependencies for the personality.
        Runs pip install based on the requirements.txt file.
        """
        super().install()
        requirements_file = self.personality.personality_package_path / "requirements.txt"
        try:
            subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)], check=True)
            self.personality.info("ArtBot dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            self.personality.error(f"Failed to install dependencies: {e}")
            ASCIIColors.error("Failed to install ArtBot dependencies.")
        except FileNotFoundError:
            self.personality.warning("requirements.txt not found. Skipping pip install.")
            ASCIIColors.warning("requirements.txt not found for ArtBot. Skipping pip install.")


    def remove_image_links(self, markdown_text: str) -> str:
        """
        Removes Markdown image links from text. Used to clean context for the LLM.

        Args:
            markdown_text (str): The text containing potential Markdown image links.

        Returns:
            str: The text with image links removed.
        """
        # Regular expression pattern to match image links in Markdown: ![alt text](url)
        image_link_pattern = r"!\[.*?\]\(.*?\)"
        return re.sub(image_link_pattern, "", markdown_text)

    # --------------------- Commands -----------------------------

    def help(self, prompt: str = "", full_context: str = "", client: Optional[Client] = None) -> None:
        """
        Displays the help message for the personality.

        Args:
            prompt (str): The user prompt (unused).
            full_context (str): The full conversation context (unused).
            client (Optional[Client]): The client instance (unused).
        """
        self.personality.InfoMessage(self.personality.help)

    def new_image(self, prompt: str = "", full_context: str = "", client: Optional[Client] = None) -> None:
        """
        Clears the list of images used for img2img context.

        Args:
            prompt (str): The user prompt (unused).
            full_context (str): The full conversation context (unused).
            client (Optional[Client]): The client instance (unused).
        """
        # self.personality.image_files = [] # image_files is now managed by Client object/discussion
        if client:
            client.discussion.image_files = [] # Assuming this is how context images are stored
            self.personality.info("Cleared image context. Starting fresh :)")
        else:
             self.personality.warning("No client provided, cannot clear image context.")


    def show_settings(self, prompt: str = "", full_context: str = "", client: Optional[Client] = None) -> None:
        """
        Opens the Stable Diffusion web UI settings page in a browser.

        Args:
            prompt (str): The user prompt (unused).
            full_context (str): The full conversation context (unused).
            client (Optional[Client]): The client instance (unused).
        """
        try:
            sd_url = self.config.sd_address
            if not sd_url.startswith(('http://', 'https://')):
                sd_url = 'http://' + sd_url
            webbrowser.open(sd_url + "/?__theme=dark")
            self.step("Opened Stable Diffusion settings UI in your browser.", client=client)
        except Exception as e:
            self.error(f"Could not open Stable Diffusion UI: {e}", client=client)

    def show_last_image(self, prompt: str = "", full_context: str = "", client: Optional[Client] = None) -> None:
        """
        Displays the last generated or uploaded image in the chat.

        Args:
            prompt (str): The user prompt (unused).
            full_context (str): The full conversation context (unused).
            client (Optional[Client]): The client instance.
        """
        if client and client.discussion.image_files:
            last_image_path = client.discussion.image_files[-1]
            try:
                img_url = discussion_path_to_url(last_image_path)
                # Use the new set_message_html with the media class
                html_content = f'<img src="{img_url}" class="media" alt="Last image">'
                self.set_message_html(html_content, client=client)
            except Exception as e:
                self.error(f"Error creating URL for image: {e}", client=client)
        else:
            self.warning("No last image available to show.", client=client)

    # --------------------- File Handling -----------------------------

    def add_file(self, path: Path, client: Client, callback: Optional[Callable] = None) -> Union[str, None]:
        """
        Handles file uploads, displaying the image and optionally captioning it.

        Args:
            path (Path): The path to the uploaded file.
            client (Client): The client instance associated with the upload.
            callback (Optional[Callable]): An optional callback function.

        Returns:
            Union[str, None]: HTML content string or None if processing fails.
        """
        
        active_callback = callback if callback else self.callback
        if not client:
            self.error("Client information is missing, cannot process file.")
            return None

        try:
            # Ensure the path is absolute and exists
            abs_path = Path(path)
            if not abs_path.is_absolute():
                 abs_path = self.personality.lollms_paths.personal_outputs_path/client.discussion.discussion_folder_name/ path
            if not abs_path.exists():
                 abs_path = self.personality.lollms_paths.personal_outputs_path/ path
            if not abs_path.exists():
                 abs_path = self.personality.lollms_paths.uploads_path/ path


            if not abs_path.exists():
                self.error(f"Uploaded file not found at expected paths: {path}", client=client)
                return None

            # Add the file path to the discussion context (use absolute path)
            if not hasattr(client.discussion, 'image_files'):
                client.discussion.image_files = []
            client.discussion.image_files.append(str(abs_path)) # Store as string

            # Generate web-accessible URL
            file_url = discussion_path_to_url(abs_path, client_id=client.client_id)
            if not file_url:
                self.error(f"Could not generate URL for path: {abs_path}", client=client)
                return None

            # Prepare initial HTML for the image
            image_html = f'<div><img src="{file_url}" class="media" alt="Uploaded image: {abs_path.name}"></div>'
            caption_html = ""

            if self.config.caption_received_files:
                self.step_start("Understanding the image...", client=client, callback=active_callback)
                try:
                    if not self.personality.model:
                        self.error("Cannot caption image: No model loaded.", client=client, callback=active_callback)
                        self.step_end("Understanding the image...", status='failed', client=client, callback=active_callback)
                    else:
                        img = Image.open(abs_path).convert("RGB")
                        # Assuming interrogate_blip is available on the model object
                        descriptions = self.personality.model.interrogate_blip([img])
                        description = descriptions[0] if descriptions else "Could not describe the image."
                        self.print_prompt("BLIP description", description)
                        caption_html = f'<div class="mt-2 text-sm text-gray-600 dark:text-gray-300"><b>Image description:</b><br>{description}</div>'
                        self.step_end("Understanding the image.", client=client, callback=active_callback)

                except Exception as e:
                    self.error(f"Failed to caption image: {e}", client=client, callback=active_callback)
                    self.step_end("Understanding the image...", status='failed', client=client, callback=active_callback)
                    caption_html = '<div class="mt-2 text-sm text-red-500">Failed to generate description.</div>'

            # Combine HTML and send using set_message_html
            final_html = f'<div class="border p-2 rounded-lg">{image_html}{caption_html}</div>'
            self.set_message_html(final_html, client=client, callback=active_callback)
            return final_html # Return the generated HTML

        except Exception as e:
            self.error(f"Error processing file {path}: {e}", client=client, callback=active_callback)
            return None # Return None on failure

    # --------------------- Core Processing -----------------------------

    def prepare(self, client: Optional[Client]=None) -> bool:
        """
        Checks if the Text-to-Image (TTI) service is available and configured.

        Args:
            client (Optional[Client]): The client instance.

        Returns:
            bool: True if the TTI service is ready, False otherwise.
        """
        if self.personality.app.tti is None:
            self.error(
                "TTI service is not configured. Please go to Settings > Services Zoo "
                "and configure a Text-to-Image service (e.g., Automatic1111).",
                client=client
            )
            return False
        self.tti = self.personality.app.tti # Store reference if needed later
        return True

    def regenerate(self, prompt: str = "", full_context: str = "", client: Optional[Client] = None) -> None:
        """
        Regenerates an image using the parameters from the previous generation.

        Args:
            prompt (str): The user prompt (unused).
            full_context (str): The full conversation context (unused).
            client (Optional[Client]): The client instance.
        """
        if not client:
            self.error("Client information is missing, cannot regenerate.")
            return

        if not self.prepare(client):
            return

        metadata = client.discussion.get_metadata()
        positive_prompt = metadata.get("positive_prompt")
        negative_prompt = metadata.get("negative_prompt")
        sd_title = metadata.get("sd_title", "Regenerated Image")
        width = metadata.get("width", self.config.width)
        height = metadata.get("height", self.config.height)
        seed = metadata.get("seed", self.config.seed) # Use previous seed? Or new one? Let's use a new one.
        steps = metadata.get("steps", self.config.steps)
        scale = metadata.get("scale", self.config.scale)
        sampler = metadata.get("sampler_name", self.config.sampler_name)
        img2img_strength = metadata.get("img2img_denoising_strength", self.config.img2img_denoising_strength)

        if not positive_prompt:
            self.warning("No previous prompt found in metadata. Please generate an image first.", client=client)
            return

        self.step_start("Regenerating image...", client=client)
        initial_md = f"### Regenerating with previous settings:\n"
        initial_md += f"- **Positive Prompt:** `{positive_prompt}`\n"
        initial_md += f"- **Negative Prompt:** `{negative_prompt}`\n" if negative_prompt else ""
        initial_md += f"- **Dimensions:** {width}x{height}\n"
        initial_md += f"- **Sampler:** {sampler}\n"
        initial_md += f"- **Steps:** {steps}, **Scale:** {scale}\n"

        self.set_message_content(initial_md, client=client) # Show parameters being used

        # Call the paint function with retrieved metadata
        self._paint(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt or "", # Ensure it's a string
            sd_title=sd_title,
            width=width,
            height=height,
            sampler_name=sampler,
            seed=-1, # Use new seed for variation
            scale=scale,
            steps=steps,
            img2img_denoising_strength=img2img_strength,
            client=client,
            metadata_info_md=initial_md # Pass initial markdown to append results
        )
        self.step_end("Regenerating image.", client=client)

    def get_styles(self, initial_prompt: str, full_context: str, client: Optional[Client] = None) -> str:
        """
        Uses the LLM to select appropriate styles for the user's prompt.

        Args:
            initial_prompt (str): The user's initial request.
            full_context (str): The conversation history.
            client (Optional[Client]): The client instance.

        Returns:
            str: A comma-separated string of selected styles, or an empty string.
        """
        self.step_start("Selecting style...", client=client)
        styles_options = [
            "Oil painting", "Octane rendering", "Cinematic", "Art deco", "Enameled", "Etching",
            "Arabesque", "Cross Hatching", "Calligraphy", "Vector art", "Vexel art", "Cartoonish",
            "Cubism", "Surrealism", "Pop art", "Pop surrealism", "Rorschach Inkblot", "Flat icon",
            "Material Design Icon", "Skeuomorphic Icon", "Glyph Icon", "Outline Icon", "Gradient Icon",
            "Neumorphic Icon", "Vintage Icon", "Abstract Icon"
        ]
        styles_list_str = ", ".join(styles_options)

        prompt = self.build_prompt([
            f"{self.system_full_header}Select suitable style(s) for the following request.",
            f"{self.system_custom_header('Available Styles')}{styles_list_str}",
            f"{self.system_custom_header('User Request')}{initial_prompt}",
            f"{self.system_custom_header('Conversation Context')}{self.remove_image_links(full_context)}",
            f"Based on the request and context, which style(s) from the list are most appropriate for creating {self.config.production_type}?",
            "List only the selected styles, separated by commas.",
            self.ai_full_header + "Selected styles:",
        ], few_shot_prompt=2) # Assuming few_shot_prompt maps to context window slices

        try:
            llm_styles = self.generate(prompt, self.config.max_generation_prompt_size).strip()
            # Filter LLM output to only include valid styles
            selected_styles = [s for s in styles_options if s.lower() in llm_styles.lower()]
            result = ", ".join(selected_styles)
            self.step_end(f"Selected style: {result}" if result else "No specific style selected.", client=client)
            return result
        except Exception as e:
            self.error(f"Error during style selection: {e}", client=client)
            self.step_end("Selecting style...", status='failed', client=client)
            return ""

    def get_resolution(self, initial_prompt: str, full_context: str, default_resolution: List[int], client: Optional[Client] = None) -> List[int]:
        """
        Uses the LLM to determine a suitable image resolution based on the prompt.

        Args:
            initial_prompt (str): The user's initial request.
            full_context (str): The conversation history.
            default_resolution (List[int]): The default [width, height] to use if detection fails.
            client (Optional[Client]): The client instance.

        Returns:
            List[int]: The selected [width, height].
        """
        self.step_start("Choosing resolution...", client=client)

        def extract_resolution(text: str, default: List[int]) -> List[int]:
            # Enhanced pattern to match (w, h), w x h, w by h etc., ignoring case
            pattern = r'\(?\s*(\d+)\s*(?:,|\s*x\s*|\s*by\s*)\s*(\d+)\s*\)?'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    width = int(match.group(1))
                    height = int(match.group(2))
                    # Basic sanity check for resolution ranges
                    if 64 <= width <= 8192 and 64 <= height <= 8192:
                         # Check if proportional to common aspect ratios, adjust if needed? (Future enhancement)
                        return [width, height]
                    else:
                         self.warning(f"Extracted resolution {width}x{height} out of bounds, using default.", client=client)
                         return default
                except ValueError:
                     self.warning(f"Could not parse extracted resolution: {match.group(0)}", client=client)
                     return default
            else:
                self.info(f"No resolution pattern found in LLM output: '{text[:100]}...'. Using default.", client=client)
                return default

        prompt = self.build_prompt([
            f"{self.system_full_header}Determine the optimal image resolution (width, height) for the user's request.",
            f"Consider common aspect ratios (e.g., 1:1, 16:9, 9:16, 4:3, 3:4, 2:1, 1:2).",
            f"The default resolution is ({default_resolution[0]}, {default_resolution[1]}).",
            f"{self.system_custom_header('User Request')}{initial_prompt}",
            f"{self.system_custom_header('Conversation Context')}{self.remove_image_links(full_context)}",
            "Output only the selected resolution in the format (width, height).",
            self.ai_full_header + "Selected resolution:",
        ], few_shot_prompt=2)

        try:
            llm_output = self.generate(prompt, self.config.max_generation_prompt_size).strip()
            resolution = extract_resolution(llm_output, default_resolution)
            self.width, self.height = resolution # Update instance variables
            self.step_end(f"Chosen resolution: {self.width}x{self.height}", client=client)
            return resolution
        except Exception as e:
            self.error(f"Error during resolution selection: {e}", client=client)
            self.step_end("Choosing resolution...", status='failed', client=client)
            return default_resolution

    def _paint(
        self,
        positive_prompt: str,
        negative_prompt: str,
        sd_title: str,
        width: int,
        height: int,
        sampler_name: str,
        seed: int,
        scale: float,
        steps: int,
        img2img_denoising_strength: float,
        client: Client,
        metadata_info_md: str = "" # Accumulates markdown info
    ) -> Optional[Dict[str, Any]]:
        """
        Internal method to perform the actual image generation using the TTI service.

        Args:
            positive_prompt (str): The positive prompt.
            negative_prompt (str): The negative prompt.
            sd_title (str): Title for the image (used in alt text).
            width (int): Image width.
            height (int): Image height.
            sampler_name (str): The sampler to use.
            seed (int): Generation seed.
            scale (float): CFG scale.
            steps (int): Number of steps.
            img2img_denoising_strength (float): Denoising strength for img2img.
            client (Client): The client instance.
            metadata_info_md (str): Existing markdown content to append to.

        Returns:
            Optional[Dict[str, Any]]: Generation info dictionary from the TTI service, or None on failure.
        """
        if not self.prepare(client): # Ensure TTI is ready
            return None

        generated_images_html = []
        last_infos = None
        output_folder = client.discussion.discussion_folder # Use discussion-specific folder

        for i in range(self.config.num_images):
            step_msg = f"Generating image {i + 1}/{self.config.num_images}"
            self.step_start(step_msg, client=client)

            current_seed = seed if seed != -1 else -1 # Use fixed seed if provided, else random for each image
            
            # Determine if img2img is applicable
            use_img2img = self.config.continue_from_last_image and client.discussion.image_files
            input_images = [client.discussion.image_files[-1]] if use_img2img else None

            try:
                gen_params = {
                    "prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "sampler_name": sampler_name,
                    "seed": current_seed,
                    "cfg_scale": scale,
                    "steps": steps,
                    "width": width,
                    "height": height,
                    "output_path": output_folder,
                    # Add conditional parameters
                    "restore_faces": self.config.restore_faces,
                    "tiling": False, # Add if configurable later
                    "skip_grid": self.config.skip_grid,
                    "batch_size": self.config.batch_size, # TTI needs to handle batch>1 correctly
                     # img2img specific
                    "image_paths": input_images, # Will be None if not img2img
                    "denoising_strength": img2img_denoising_strength if use_img2img else None,
                }
                # Filter out None values before calling TTI
                gen_params = {k: v for k, v in gen_params.items() if v is not None}

                # Call the appropriate TTI method
                # Assuming paint method handles both txt2img and img2img based on image_paths
                generated_files_infos = self.tti.paint(**gen_params)


                # Process results (assuming TTI returns a list of tuples (filepath, info_dict))
                if not generated_files_infos:
                    self.warning(f"TTI service did not return any files for image {i+1}.", client=client)
                    self.step_end(step_msg, status='failed', client=client)
                    continue # Try next image if any

                # Handle single or multiple results from TTI (e.g., if batch > 1)
                for file_path_str, infos in generated_files_infos:
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        escaped_url = discussion_path_to_url(file_path, client_id=client.client_id)
                        if escaped_url:
                            alt_text = f"{sd_title} - Image {i+1}"
                            # Add info to alt text or title if needed? For now, keep it simple.
                            html_img = f'<img src="{escaped_url}" class="media" alt="{alt_text}" title="{alt_text}">'
                            generated_images_html.append(html_img)
                            last_infos = infos # Store info from the last successful generation
                            # Add generated file to discussion context if needed for future steps
                            if not hasattr(client.discussion, 'image_files'):
                                client.discussion.image_files = []
                            client.discussion.image_files.append(str(file_path))

                        else:
                            self.warning(f"Could not create URL for generated image: {file_path}", client=client)
                    else:
                        self.warning(f"Generated image file not found: {file_path}", client=client)

                self.step_end(step_msg, client=client)

            except Exception as e:
                self.error(f"Error during image generation {i + 1}: {e}", client=client)
                self.step_end(step_msg, status='failed', client=client)
                # Optionally break the loop on error, or continue with next image
                # break

            # Update UI progressively with generated images
            if generated_images_html:
                 # Combine metadata text with the gallery of images
                current_output_html = f"<div>{metadata_info_md}</div>\n" + "\n".join(generated_images_html)
                self.set_message_html(current_output_html, client=client) # Update HTML view

        return last_infos # Return info from the last generated image (or None)

    def main_process(self, initial_prompt: str, full_context: str, context_details: LollmsContextDetails, client: Client) -> None:
        """
        Main processing logic for generating an image based on user input.

        Args:
            initial_prompt (str): The user's latest message/prompt.
            full_context (str): The entire conversation history.
            context_details (LollmsContextDetails): Context details object.
            client (Client): The client instance.
        """
        if not client:
            self.error("Client information is missing, cannot process request.")
            return

        metadata = client.discussion.get_metadata()
        sd_title = metadata.get("sd_title", "Untitled Artwork")
        metadata_info_md = "" # Accumulates markdown formatted info for the final message

        # If in discussion mode, check if user is asking to generate
        if self.config.imagine and self.config.activate_discussion_mode:
            self.step_start("Analyzing request...", client=client)
            classification = self.multichoice_question(
                "Classify the user's intention:",
                [
                    "The user is making an affirmation or statement.",
                    "The user is asking a question.",
                    "The user is requesting to generate, create, build, or make something visual.",
                    "The user is requesting to modify or change something previously generated.",
                    "The user is giving instructions unrelated to immediate generation (e.g., setting preferences)."
                ],
                f"User prompt: '{initial_prompt}'\nConversation context:\n{self.remove_image_links(full_context)}"
            )
            self.step_end("Analyzing request.", client=client)

            # If not a generation request (index 2 or 3), engage in discussion
            if classification not in [2, 3]:
                self.step_start("Generating chat response...", client=client)
                prompt_chat = self.build_prompt([
                    f"{self.system_full_header}You are ArtBot, an AI assistant specializing in art and image generation discussion. Engage in conversation, answer questions, and provide information about art concepts. Only generate image prompts when explicitly asked to create or generate.",
                    f"{self.system_custom_header('Discussion History')}{self.remove_image_links(full_context)}",
                    f"{self.user_full_header}{initial_prompt}",
                    self.ai_full_header,
                ], few_shot_prompt=len(context_details.discussion_messages) if context_details.discussion_messages else 2) # Adapt context usage

                response = self.generate(prompt_chat, self.config.max_generation_prompt_size).strip()
                self.set_message_content(response, client=client) # Send chat response
                self.step_end("Generating chat response.", client=client)
                return # Stop further processing

        # Proceed with image generation logic
        self.set_message_content("### Starting image generation process...", client=client)

        # 1. Determine Resolution
        if self.config.imagine: # Only do this if we intend to generate
            if self.config.automatic_resolution_selection:
                self.get_resolution(initial_prompt, full_context, [self.config.width, self.config.height], client)
            else:
                self.width = self.config.width
                self.height = self.config.height
            metadata_info_md += f"- **Resolution:** {self.width}x{self.height}\n"
            self.set_message_content(f"### Generation Parameters:\n{metadata_info_md}", client=client) # Update UI

        # 2. Determine Style
        styles = ""
        if self.config.imagine and self.config.add_style:
            styles = self.get_styles(initial_prompt, full_context, client)
            if styles:
                metadata_info_md += f"- **Style:** {styles}\n"
                self.set_message_content(f"### Generation Parameters:\n{metadata_info_md}", client=client) # Update UI

        # 3. Generate Positive Prompt
        positive_prompt = initial_prompt # Default if imagine is off
        if self.config.imagine:
            self.step_start("Crafting positive prompt...", client=client)
            past_discussion = self.remove_image_links(full_context) if self.config.continuous_discussion else ""
            style_instruction = f"{self.system_custom_header('Style Hint')}{styles}\n" if styles else ""

            # Get examples for prompt generation
            examples_md = ""
            try:
                if self.config.examples_extraction_method == "random":
                    examples = get_random_image_gen_prompt(self.config.number_of_examples_to_recover)
                elif self.config.examples_extraction_method == "rag_based":
                    examples = get_image_gen_prompt(initial_prompt, self.config.number_of_examples_to_recover)
                else:
                    examples = []

                if examples:
                    examples_md = f"{self.system_custom_header('Prompt Examples')}\n" + "\n".join([f"- `{ex}`" for ex in examples]) + "\n"

            except Exception as e:
                self.warning(f"Could not retrieve prompt examples: {e}", client=client)


            prompt_build_positive = self.build_prompt([
                f"{self.system_full_header}You are ArtBot, an expert AI prompt generator for text-to-image models.",
                f"Generate a concise, detailed, and effective positive prompt based on the user's request and discussion context.",
                f"Incorporate the requested production type: **{self.config.production_type}**.",
                "Focus on visual details, composition, lighting, and artistic style.",
                "Avoid conversational phrases. Output only the prompt itself.",
                f"{self.system_custom_header('Discussion Context')}{past_discussion}",
                style_instruction,
                examples_md, # Add examples if available
                f"{self.user_full_header}{initial_prompt}",
                f"{self.ai_full_header}Generated Positive Prompt:",
            ], few_shot_prompt=4) # Adapt context usage

            self.print_prompt("Positive Prompt Generation", prompt_build_positive)
            try:
                positive_prompt = self.generate(prompt_build_positive, self.config.max_generation_prompt_size, callback=self.sink).strip()
                # Basic prompt cleaning (remove potential markdown, etc.)
                positive_prompt = re.sub(r'^Prompt:\s*', '', positive_prompt, flags=re.IGNORECASE)
                positive_prompt = positive_prompt.replace("`","") # Remove backticks often added by LLMs

                self.step_end("Crafting positive prompt.", client=client)
                metadata_info_md += f"- **Positive Prompt:** `{positive_prompt}`\n"
                self.set_message_content(f"### Generation Parameters:\n{metadata_info_md}", client=client)
            except Exception as e:
                 self.error(f"Failed to generate positive prompt: {e}", client=client)
                 self.step_end("Crafting positive prompt.", status='failed', client=client)
                 return # Cannot proceed without a prompt

        # 4. Generate Negative Prompt
        negative_prompt = ""
        if self.config.imagine:
             if self.config.use_fixed_negative_prompts:
                 negative_prompt = self.config.fixed_negative_prompts
                 self.info("Using fixed negative prompt.", client=client)
             else:
                self.step_start("Crafting negative prompt...", client=client)
                past_discussion = self.remove_image_links(full_context) if self.config.continuous_discussion else ""
                prompt_build_negative = self.build_prompt([
                    f"{self.system_full_header}You are ArtBot. Generate a negative prompt to avoid common image generation issues.",
                    "Focus on keywords like: text, signature, watermark, blurry, deformed, extra limbs, bad anatomy, ugly, duplicate.",
                    "Use weighted terms like `(((word)))` for emphasis.",
                    f"{self.system_custom_header('Positive Prompt')}{positive_prompt}",
                    f"{self.system_custom_header('Discussion Context')}{past_discussion}",
                    "Output only the negative prompt keywords, comma-separated.",
                    f"{self.ai_full_header}Generated Negative Prompt:",
                ], few_shot_prompt=4)

                self.print_prompt("Negative Prompt Generation", prompt_build_negative)
                try:
                    negative_prompt = self.generate(prompt_build_negative, self.config.max_generation_prompt_size, callback=self.sink).strip()
                    # Clean negative prompt
                    negative_prompt = re.sub(r'^Negative Prompt:\s*', '', negative_prompt, flags=re.IGNORECASE)
                    negative_prompt = negative_prompt.replace("`","") # Remove backticks
                    if "morbid" not in negative_prompt.lower(): # Ensure a common term is present
                         negative_prompt = "((morbid)), " + negative_prompt

                    self.step_end("Crafting negative prompt.", client=client)
                except Exception as e:
                    self.error(f"Failed to generate negative prompt: {e}", client=client)
                    self.step_end("Crafting negative prompt.", status='failed', client=client)
                    # Continue with empty or default negative prompt? Let's use default.
                    negative_prompt = DEFAULT_NEGATIVE_PROMPT
                    self.warning("Using default negative prompt due to error.", client=client)


             metadata_info_md += f"- **Negative Prompt:** `{negative_prompt}`\n"
             self.set_message_content(f"### Generation Parameters:\n{metadata_info_md}", client=client)


        # 5. Generate Title
        if self.config.imagine and self.config.build_title:
            self.step_start("Making up a title...", client=client)
            prompt_build_title = self.build_prompt([
                 f"{self.system_full_header}Generate a short, evocative title for an artwork based on its prompts.",
                 f"{self.system_custom_header('Positive Prompt')}{positive_prompt}",
                 f"{self.system_custom_header('Negative Prompt')}{negative_prompt}",
                 "Output only the title.",
                 f"{self.ai_full_header}Title:",
            ], few_shot_prompt=2)

            self.print_prompt("Title Generation", prompt_build_title)
            try:
                sd_title = self.generate(prompt_build_title, max_size=100).strip() # Smaller max size for title
                sd_title = sd_title.replace('"', '').replace("`", "").replace("*", "") # Clean title
                self.step_end("Making up a title.", client=client)
                metadata_info_md += f"- **Title:** {sd_title}\n"
                self.set_message_content(f"### Generation Parameters:\n{metadata_info_md}", client=client)
            except Exception as e:
                 self.error(f"Failed to generate title: {e}", client=client)
                 self.step_end("Making up a title.", status='failed', client=client)
                 sd_title = "Untitled Artwork" # Fallback title

        # Store metadata before painting
        metadata["positive_prompt"] = positive_prompt
        metadata["negative_prompt"] = negative_prompt
        metadata["sd_title"] = sd_title
        metadata["width"] = self.width
        metadata["height"] = self.height
        metadata["steps"] = self.config.steps
        metadata["scale"] = self.config.scale
        metadata["seed"] = self.config.seed # Store the config seed, actual generation might use -1
        metadata["sampler_name"] = self.config.sampler_name
        metadata["img2img_denoising_strength"] = self.config.img2img_denoising_strength
        client.discussion.set_metadata(metadata)

        # 6. Paint the image
        generation_infos = None
        if self.config.paint:
             if not self.prepare(client):
                 return # Stop if TTI is not ready

             generation_infos = self._paint(
                 positive_prompt=positive_prompt,
                 negative_prompt=negative_prompt,
                 sd_title=sd_title,
                 width=self.width,
                 height=self.height,
                 sampler_name=self.config.sampler_name,
                 seed=self.config.seed,
                 scale=self.config.scale,
                 steps=self.config.steps,
                 img2img_denoising_strength=self.config.img2img_denoising_strength,
                 client=client,
                 metadata_info_md=metadata_info_md # Pass the accumulated MD to _paint
             )
        else:
             self.info("Image generation (painting) is disabled in settings.", client=client)
             # Even if not painting, show the final parameters
             self.set_message_content(f"### Generation Parameters:\n{metadata_info_md}\n*Painting is disabled. Parameters are ready.*", client=client)


        # 7. Show Infos (Optional)
        if self.config.show_infos and generation_infos:
            try:
                # Format infos nicely using JSON or key-value pairs in markdown
                 info_md = "\n### Generation Details:\n```json\n" + \
                           f"{json.dumps(generation_infos, indent=2)}\n```"
                 # Append this to the existing message HTML, or send as a new message? Append.
                 final_html = self.get_message_html() # Assuming we can get the current HTML
                 if final_html:
                     self.set_message_html(final_html + info_md, client=client)
                 else: # Fallback if getting current HTML isn't straightforward
                     self.json("Generation Info", generation_infos, client=client)
            except Exception as e:
                 self.warning(f"Could not display generation info: {e}", client=client)


    async def handle_request(self, data: dict, client: Client) -> Dict[str, Any]:
        """
        Handles custom websocket requests for actions like variations.
        NOTE: The triggering mechanism for these actions (e.g., buttons in the UI)
        needs to be implemented separately, as the old JS-based image click
        is removed in this refactor.

        Args:
            data (dict): A dictionary containing the request data (e.g., operation, image path).
            client (Client): The client instance making the request.

        Returns:
            dict: A dictionary containing the response status and message.
        """
        operation = data.get("operation") # Renamed from 'name' for clarity
        image_path_str = data.get("imagePath") # Relative path as sent by client?
        prompt = data.get("prompt") # Optional new prompt for variation
        negative_prompt = data.get("negative_prompt") # Optional new negative prompt

        self.personality.info(f"Received handle_request: operation={operation}, imagePath={image_path_str}")

        if not client:
            return {"status": False, "message": "Client information missing."}

        if not image_path_str:
             return {"status": False, "message": "Missing imagePath in request."}

        # Construct the full path relative to the discussion output folder
        # This assumes imagePath is just the filename or relative within discussion/sd/
        base_folder = client.discussion.discussion_folder
        # Try resolving the path robustly
        possible_paths = [
            base_folder / image_path_str,
            base_folder / "sd" / image_path_str.split('/')[-1], # If path includes 'sd/' prefix
             self.personality.lollms_paths.personal_outputs_path/client.discussion.discussion_folder_name/image_path_str,
             self.personality.lollms_paths.personal_outputs_path/client.discussion.discussion_folder_name/"sd"/image_path_str.split('/')[-1]
        ]
        image_path : Optional[Path] = None
        for p in possible_paths:
            if p.exists():
                image_path = p
                break

        if not image_path:
            self.error(f"Could not find image file for operation '{operation}': {image_path_str}", client=client)
            return {"status": False, "message": f"Image file not found: {image_path_str}"}

        if operation == "variate":
            self.info(f"Variation requested for file: {image_path}", client=client)

            # Set the image as the source for img2img
            client.discussion.image_files = [str(image_path)]
            metadata = client.discussion.get_metadata()

            # Use provided prompts if available, otherwise fallback to metadata
            current_prompt = prompt or metadata.get("positive_prompt", "")
            current_neg_prompt = negative_prompt or metadata.get("negative_prompt", "")

            if not current_prompt:
                return {"status": False, "message": "Cannot create variation without a positive prompt."}

            # Update metadata with potentially new prompts
            metadata["positive_prompt"] = current_prompt
            metadata["negative_prompt"] = current_neg_prompt
            client.discussion.set_metadata(metadata)

            # Set config for img2img mode for the next generation
            self.config.continue_from_last_image = True # Ensure img2img is active

            self.info(f"Generating {self.config.num_images} variation(s)...", client=client)

            # Call regenerate, which now uses metadata correctly
            # Need to wrap this in async if regenerate becomes async
            # For now, assuming synchronous call works within async handler
            self.regenerate(client=client) # Regenerate handles the paint call

            return {"status": True, "message": f"Variation generation started for {image_path.name}."}

        elif operation == "set_as_current":
            self.info(f"Setting image as current for img2img: {image_path}", client=client)
            client.discussion.image_files = [str(image_path)]
            self.config.continue_from_last_image = True # Enable img2img mode
            self.personality.info(f"Image '{image_path.name}' is now set as the input for the next generation (img2img).", client=client)
            return {"status": True, "message": f"Image '{image_path.name}' set for img2img."}

        else:
            self.warning(f"Unknown operation requested: {operation}", client=client)
            return {"status": False, "message": f"Unknown operation: {operation}"}


    def run_workflow(self, context_details: LollmsContextDetails, client: Client, callback: Optional[Callable] = None):
        """
        Entry point for executing the personality's workflow.

        Args:
            context_details (LollmsContextDetails): Details about the context, including prompt and messages.
            client (Client): The client instance initiating the workflow.
            callback (Optional[Callable]): An optional callback function for progress updates.

        Returns:
            This method typically orchestrates actions and sends output via callbacks
            or side effects (like updating UI), rather than returning a direct value.
        """
        self.callback = callback # Store callback for potential use in async operations
        prompt = context_details.prompt # The user's latest input
        full_context = context_details.get_discussion_to_({}) # Get formatted discussion history

        # Make sure client is attached to the context details or passed correctly
        if not client:
             # Attempt to get client from context_details if available, or raise error
             # client = context_details.get_client() # Hypothetical method
             self.error("Workflow cannot run without client information.")
             # Optionally call callback with error status
             if self.callback:
                 add_callback(self.callback, "", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return


        try:
            self.main_process(prompt, full_context, context_details, client)
        except Exception as e:
            self.error(f"An error occurred during the workflow: {e}", client=client)
            # Optionally call callback with error status
            if self.callback:
                add_callback(self.callback, f"Workflow error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)

        # Workflow completion signal (optional)
        # self.info("ArtBot workflow finished.", client=client)

        # The run_workflow usually doesn't return content directly,
        # output is handled by set_message_content/html within main_process
        # Returning empty string or None is common.
        return None # Indicate completion, actual output sent via UI methods
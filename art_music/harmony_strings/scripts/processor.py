"""
Project: LoLLMs
Personality: # Placeholder: Personality name (e.g., "Science Enthusiast")
Author: # Placeholder: Creator name (e.g., "ParisNeo")
Description: # Placeholder: Personality description (e.g., "A personality designed for enthusiasts of science and technology, promoting engaging and informative interactions.")
"""

from lollms.helpers import ASCIIColors
from lollms.config import TypedConfig, BaseConfig, ConfigTemplate
from lollms.personality import APScript, AIPersonality, MSG_TYPE
import subprocess
from typing import Callable


import matplotlib
matplotlib.use('TkAgg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import json
from datetime import datetime

class GuitarLearningDB:
    def __init__(self, db_path='guitar_learning.db'):
        self.db_path = db_path
        self.create_tables()

    def create_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS UserProfile (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                current_level TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ProgressTrack (
                progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                level TEXT NOT NULL,
                chords TEXT,
                scales TEXT,
                songs TEXT,
                techniques TEXT,
                challenge_completed BOOLEAN,
                FOREIGN KEY (user_id) REFERENCES UserProfile (user_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS AIState (
                state_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                ai_memory TEXT,
                FOREIGN KEY (user_id) REFERENCES UserProfile (user_id)
            )
        ''')
        conn.commit()
        conn.close()

    def add_user_profile(self, name, email, current_level):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO UserProfile (name, email, current_level)
            VALUES (?, ?, ?)
        ''', (name, email, current_level))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id

    def get_user_id_by_name(self, name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id FROM UserProfile
            WHERE name = ?
        ''', (name,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def list_all_profiles(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, name, email, current_level FROM UserProfile
        ''')
        results = cursor.fetchall()
        conn.close()
        return results

    def remove_user_profile(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM UserProfile WHERE user_id = ?
        ''', (user_id,))
        cursor.execute('''
            DELETE FROM ProgressTrack WHERE user_id = ?
        ''', (user_id,))
        cursor.execute('''
            DELETE FROM AIState WHERE user_id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()

    def get_user_profile(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT name, email, current_level FROM UserProfile
            WHERE user_id = ?
        ''', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result

    def update_user_profile(self, user_id, name=None, email=None, current_level=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE UserProfile
            SET name = COALESCE(?, name),
                email = COALESCE(?, email),
                current_level = COALESCE(?, current_level)
            WHERE user_id = ?
        ''', (name, email, current_level, user_id))
        conn.commit()
        conn.close()

    def log_progress(self, user_id, level, chords, scales, songs, techniques, challenge_completed):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ProgressTrack (user_id, level, chords, scales, songs, techniques, challenge_completed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, level, chords, scales, songs, techniques, challenge_completed))
        conn.commit()
        conn.close()

    def get_last_ai_state(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ai_memory FROM AIState
            WHERE user_id = ?
            ORDER BY state_id DESC
            LIMIT 1
        ''', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def set_ai_state(self, user_id, ai_memory):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO AIState (user_id, ai_memory)
            VALUES (?, ?)
        ''', (user_id, ai_memory))
        conn.commit()
        conn.close()

    def get_user_overall_progress(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT level, chords, scales, songs, techniques, challenge_completed FROM ProgressTrack
            WHERE user_id = ?
            ORDER BY progress_id DESC
            LIMIT 1
        ''', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return {
            'level': result[0],
            'chords': result[1],
            'scales': result[2],
            'songs': result[3],
            'techniques': result[4],
            'challenge_completed': result[5],
        } if result else {}


    def plot_user_progress(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT date, level, chords, scales, songs, techniques FROM ProgressTrack
            WHERE user_id = ?
            ORDER BY date
        ''', (user_id,))
        results = cursor.fetchall()
        conn.close()

        if not results:
            print("No progress data to plot.")
            return

        # Extracting data for plotting
        dates = [datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S') for row in results]
        levels = [row[1] for row in results]
        chords = [row[2] for row in results]
        scales = [row[3] for row in results]
        songs = [row[4] for row in results]
        techniques = [row[5] for row in results]

        # Convert levels to numeric values if they are not already numeric
        levels_numeric = [float(level) if level.isdigit() else float(index) for index, level in enumerate(levels)]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(dates, levels_numeric, label='Level')
        plt.plot(dates, chords, label='Chords learned')
        plt.plot(dates, scales, label='Scales learned')
        plt.plot(dates, songs, label='Songs learned')
        plt.plot(dates, techniques, label='Techniques learned')

        # Formatting the plot
        plt.gcf().autofmt_xdate()  # Auto format the date on x-axis
        date_format = mdates.DateFormatter('%Y-%m-%d')
        plt.gca().xaxis.set_major_formatter(date_format)

        plt.title('User Progress Over Time')
        plt.xlabel('Date')
        plt.ylabel('Progress')
        plt.legend()
        plt.tight_layout()
        plt.show()


class Processor(APScript):
    """
    Defines the behavior of a personality in a programmatic manner, inheriting from APScript.
    
    Attributes:
        callback (Callable): Optional function to call after processing.
    """
    
    def __init__(
                 self, 
                 personality: AIPersonality,
                 callback: Callable = None,
                ) -> None:
        """
        Initializes the Processor class with a personality and an optional callback.

        Parameters:
            personality (AIPersonality): The personality instance.
            callback (Callable, optional): A function to call after processing. Defaults to None.
        """
        
        self.callback = callback
        
        # Configuration entry examples and types description:
        # Supported types: int, float, str, string (same as str, for back compatibility), text (multiline str),
        # btn (button for special actions), bool, list, dict.
        # An 'options' entry can be added for types like string, to provide a dropdown of possible values.
        personality_config_template = ConfigTemplate(
            [
                # Boolean configuration for enabling scripted AI
                #{"name":"make_scripted", "type":"bool", "value":False, "help":"Enables a scripted AI that can perform operations using python scripts."},
                
                # String configuration with options
                {"name":"user_profile_name", "type":"string", "value":"", "help":"The profile name of the user. Used to store progress data."},
                {"name":"user_level", "type":"string", "value":"beginner", "options":["beginner"], "help":"The profile name of the user. Used to store progress data."},
                 
                
                # Integer configuration example
                #{"name":"max_attempts", "type":"int", "value":3, "help":"Maximum number of attempts for retryable operations."},
                
                # List configuration example
                #{"name":"favorite_topics", "type":"list", "value":["AI", "Robotics", "Space"], "help":"List of favorite topics for personalized responses."}
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
                            states_list=[
                                {
                                    "name": "idle",
                                    "commands": {
                                        "help": self.help, # Command triggering the help method
                                    },
                                    "default": None
                                },                           
                            ],
                            callback=callback
                        )
        self.evolution_db = self.personality.personality_output_folder/"evolution.db"
        self.user_profile_db = GuitarLearningDB(str(self.evolution_db))

    def get_or_create_user_profile(self):
        if self.personality_config.user_profile_name=="":
            return False
        user_id = self.user_profile_db.get_user_id_by_name(self.personality_config.user_profile_name)
        
        if user_id is None:
            # If the user doesn't exist, create a new profile with the initial level from the configuration
            initial_level = self.personality_config.user_level
            user_id = self.user_profile_db.add_user_profile(self.personality_config.user_profile_name, '', initial_level)
            status_text = f"Created a new user profile for {self.personality_config.user_profile_name} with initial level {initial_level}."
        else:
            # If the user exists, fetch the current status
            user_profile = self.user_profile_db.get_user_profile(user_id)
            status_text = f"User {user_profile[0]} is currently at level {user_profile[2]}."

        # Retrieve the last AI state for the user
        ai_memory = self.user_profile_db.get_last_ai_state(user_id)
        if ai_memory:
            status_text += " Last session data has been retrieved."
        else:
            status_text += " No previous session data found."

        return status_text

    def get_welcome(self):
        user_level = self.get_or_create_user_profile()
        if user_level:
            return self.fast_gen(f"!@>system: Build a better  welcome message for the user.\n!@>current_welcome_message: {self.personality.welcome_message}\n!@>last session data: {user_level}\n!@>adapted welcome message:")
        else:
            return self.personality.welcome_message+"\nI see that you did not specify a profile in my settings. Please specify a profile name.\nYou need to press my icon in the chatbar and you'll see my configuration window. Type your profile name and your level, then make a new discussion.\n"

    def help(self):
        """
        Provides help information about the personality and its commands.
        """
        # Implementation of the help method
        pass

    # Note: Remember to add command implementations and additional states as needed.

    def install(self):
        """
        Install the necessary dependencies for the personality.

        This method is responsible for setting up any dependencies or environment requirements
        that the personality needs to operate correctly. It can involve installing packages from
        a requirements.txt file, setting up virtual environments, or performing initial setup tasks.
        
        The method demonstrates how to print a success message using the ASCIIColors helper class
        upon successful installation of dependencies. This step can be expanded to include error
        handling and logging for more robust installation processes.

        Example Usage:
            processor = Processor(personality)
            processor.install()
        
        Returns:
            None
        """        
        super().install()
        # Example of implementing installation logic. Uncomment and modify as needed.
        # requirements_file = self.personality.personality_package_path / "requirements.txt"
        # subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])      
        ASCIIColors.success("Installed successfully")

    def help(self, prompt="", full_context=""):
        """
        Displays help information about the personality and its available commands.

        This method provides users with guidance on how to interact with the personality,
        detailing the commands that can be executed and any additional help text associated
        with those commands. It's an essential feature for enhancing user experience and
        ensuring users can effectively utilize the personality's capabilities.

        Args:
            prompt (str, optional): A specific prompt or command for which help is requested.
                                    If empty, general help for the personality is provided.
            full_context (str, optional): Additional context information that might influence
                                          the help response. This can include user preferences,
                                          historical interaction data, or any other relevant context.

        Example Usage:
            processor = Processor(personality)
            processor.help("How do I use the 'add_file' command?")
        
        Returns:
            None
        """
        # Example implementation that simply calls a method on the personality to get help information.
        # This can be expanded to dynamically generate help text based on the current state,
        # available commands, and user context.
        self.full(self.personality.help)


    from lollms.client_session import Client
    def run_workflow(self, prompt:str, previous_discussion_text:str="", callback: Callable[[str, MSG_TYPE, dict, list], bool]=None, context_details:dict=None, client:Client=None):
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
        user_id = self.user_profile_db.get_user_id_by_name(self.personality_config.user_profile_name)
        self.personality.info("Generating")
        memory_data = self.user_profile_db.get_last_ai_state(user_id)
        prompt = self.build_prompt([
            "!@>system:\n"+context_details["conditionning"] if context_details["conditionning"] else "",
            "!@>documentation:\n"+context_details["documentation"] if context_details["documentation"] else "",
            "!@>knowledge:\n"+context_details["knowledge"] if context_details["knowledge"] else "",
            "!@>user_description:\n"+context_details["user_description"] if context_details["user_description"] else "",
            "!@>positive_boost:\n"+context_details["positive_boost"] if context_details["positive_boost"] else "",
            "!@>negative_boost:\n"+context_details["negative_boost"] if context_details["negative_boost"] else "",
            "!@>current_language:\n"+context_details["current_language"] if context_details["current_language"] else "",
            "!@>fun_mode:\n"+context_details["fun_mode"] if context_details["fun_mode"] else "",
            "!@>discussion_window:\n"+context_details["discussion_messages"] if context_details["discussion_messages"] else "",
            "!@>memory_data:\n"+memory_data if memory_data is not None else "",
            "!@>"+context_details["ai_prefix"].replace("!@>","")+":"
        ], 
        8)
        self.callback = callback
        out = self.multichoice_question("classify last user prompt.",[
            "asking a question or clarification",
            "asking to visualize his progress",
            "returning feedback on his progress",
        ],prompt)
        if out==0:
            prompt = self.build_prompt([
                "!@>system:\n"+context_details["conditionning"] if context_details["conditionning"] else "",
                "!@>documentation:\n"+context_details["documentation"] if context_details["documentation"] else "",
                "!@>knowledge:\n"+context_details["knowledge"] if context_details["knowledge"] else "",
                "!@>user_description:\n"+context_details["user_description"] if context_details["user_description"] else "",
                "!@>positive_boost:\n"+context_details["positive_boost"] if context_details["positive_boost"] else "",
                "!@>negative_boost:\n"+context_details["negative_boost"] if context_details["negative_boost"] else "",
                "!@>current_language:\n"+context_details["current_language"] if context_details["current_language"] else "",
                "!@>fun_mode:\n"+context_details["fun_mode"] if context_details["fun_mode"] else "",
                "!@>discussion_window:\n"+context_details["discussion_messages"] if context_details["discussion_messages"] else "",
                "!@>memory_data:\n"+memory_data if memory_data is not None else "",
                "!@>"+context_details["ai_prefix"].replace("!@>","")+":"
            ], 
            8)
            self.callback = callback
            out = self.fast_gen(prompt)
            self.full(out)
        elif out==1:
            self.user_profile_db.plot_user_progress(user_id)
        elif out==2:
            current_progress = self.user_profile_db.get_user_overall_progress(user_id)
            if not current_progress:
                current_progress = {
                    "level":0,
                    "chords":0,
                    "scales":0,
                    "songs":0,
                    "techniques":0,
                    "challenge_completed":0
                }
            prompt = self.build_prompt([
                "!@>system:\n"+context_details["conditionning"] if context_details["conditionning"] else "",
                "!@>documentation:\n"+context_details["documentation"] if context_details["documentation"] else "",
                "!@>knowledge:\n"+context_details["knowledge"] if context_details["knowledge"] else "",
                "!@>user_description:\n"+context_details["user_description"] if context_details["user_description"] else "",
                "!@>positive_boost:\n"+context_details["positive_boost"] if context_details["positive_boost"] else "",
                "!@>negative_boost:\n"+context_details["negative_boost"] if context_details["negative_boost"] else "",
                "!@>current_language:\n"+context_details["current_language"] if context_details["current_language"] else "",
                "!@>fun_mode:\n"+context_details["fun_mode"] if context_details["fun_mode"] else "",
                "!@>discussion_window:\n"+context_details["discussion_messages"] if context_details["discussion_messages"] else "",
                "!@>memory_data:\n"+memory_data if memory_data is not None else "",
                f"!@>current_user_level:\n{current_progress}",
                "!@>upgrading: Let's update our memory database.\nTo do that we need to issue a function call using these parameters as a json in a json markdown tag:",
                "level (float from 0 to 100), chords (float from 0 to 100), scales (float from 0 to 100), songs (float from 0 to 100), techniques (float from 0 to 100), challenge_completed (int from 0 to the number of challenges)",
                "!@>"+context_details["ai_prefix"].replace("!@>","")+":"
            ], 
            8)
            self.step_start("Building new grades")
            out = self.fast_gen(prompt, callback=self.sink)
            self.step_end("Building new grades")
            code = self.extract_code_blocks(out)
            if len(code)>0:
                parameters = json.loads(code[0]["content"])
                self.user_profile_db.log_progress(
                                                    user_id, 
                                                    parameters.get('level',current_progress['level']),
                                                    parameters.get('chords',current_progress['chords']),
                                                    parameters.get('scales',current_progress['scales']),
                                                    parameters.get('songs',current_progress['songs']),
                                                    parameters.get('techniques',current_progress['techniques']),
                                                    parameters.get('challenge_completed',current_progress['challenge_completed']),
                                                )
            current_progress_2 = self.user_profile_db.get_user_overall_progress(user_id)
            prompt = self.build_prompt([
                "!@>system:\n"+context_details["conditionning"] if context_details["conditionning"] else "",
                "!@>documentation:\n"+context_details["documentation"] if context_details["documentation"] else "",
                "!@>knowledge:\n"+context_details["knowledge"] if context_details["knowledge"] else "",
                "!@>user_description:\n"+context_details["user_description"] if context_details["user_description"] else "",
                "!@>positive_boost:\n"+context_details["positive_boost"] if context_details["positive_boost"] else "",
                "!@>negative_boost:\n"+context_details["negative_boost"] if context_details["negative_boost"] else "",
                "!@>current_language:\n"+context_details["current_language"] if context_details["current_language"] else "",
                "!@>fun_mode:\n"+context_details["fun_mode"] if context_details["fun_mode"] else "",
                "!@>discussion_window:\n"+context_details["discussion_messages"] if context_details["discussion_messages"] else "",
                "!@>memory_data:\n"+memory_data if memory_data is not None else "",
                f"!@>current_user_level:\n{current_progress}",
                "!@>upgrading: "+context_details["ai_prefix"].replace("!@>","")+" needs to build a memory note for himself so that in the next session he can remember. The memory note is in form of plauin text written inside a markdown code tag.",
                "!@>"+context_details["ai_prefix"].replace("!@>","")+":"
            ], 
            8)
            self.step_start("Building memory notes")
            out = self.fast_gen(prompt, callback=self.sink)
            self.step_end("Building memory notes")
            code = self.extract_code_blocks(out)
            self.full(f"User level updated:\nprevious_user_level:\n{current_progress}\ncurrent_user_level:\n{current_progress_2}")
            if len(code)>0:
                memory = code[0]["content"]
                self.user_profile_db.set_ai_state(user_id, memory)
                self.full(f"User level updated:\nprevious_user_level:\n{current_progress}\ncurrent_user_level:\n{current_progress_2}\nMemory information: {memory}")
        return out


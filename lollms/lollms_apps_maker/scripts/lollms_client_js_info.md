# Lollms library information
To import lollms in html:
<script src="/lollms_js"></script>
// Lollms library requires axios
// In the html don't forget to import axios.min.js cdn
// <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
First make sure you have create a variable ctx_size which contains the size of the context of the llm.
This can be either fixed or modifiable depending on the user request. You may need to store it in local store. Default value is 4096.
You can also use max_gen_size which is the maximum number of tokens that the model can generate in one go. The default value is 4096

in javascript you can generate text using the following:

```javascript
// Build lollms client
lc = new LollmsClient(
      host_address = null,// the host address (if null it will be http://localhost:9600)
      model_name = null,// The name of the model to be used. Default model will be used if this is null
      ctx_size = 4096, // The context size
      personality = -1,
      n_predict = 4096, // Default maximum number of tokens to be predicted
      temperature = 0.1,
      top_k = 50,
      top_p = 0.95,
      repeat_penalty = 0.8,
      repeat_last_n = 40,
      seed = null,
      n_threads = 8,
      service_key = "",
      default_generation_mode = ELF_GENERATION_FORMAT.LOLLMS
    )
// Now we can use it to generate text from a prompt
// First we build the prompt
// We can instruct the AI to do something in this way
system_prompt= ""// here put the instruction required
prompt = lc.system_message() + system_prompt + lc.template.separator_template + lc.ai_message()
// if you need to use chat mode then
user_prompt= ""// here put the user prompt if applicable
prompt = lc.system_message() + system_prompt + lc.template.separator_template + lc.user_message() + user_prompt + lc.template.separator_template + lc.ai_message()
// Adapt the prompt to the application
// Now to generate text, you can use:
const generated_text = await lc.generate(prompt);
```
We can also use TasksLibrary which provides advanced uses like summary, yes_no questions, multi chioce questions etc.
To build a task library we use this construcor

```javascript
tl =  new TasksLibrary(lc) // lc is a LollmsClient object
``` 
Now we can use one of these methods:

```javascript
async summarizeText(textChunk, summaryLength = "short", host_address = null, model_name = null, temperature = 0.1, maxGenerationSize = 1000)// Don't use host_address as it will use the lc one, the same goes for model_name

extractCodeBlocks(text)
    /**
     * This function extracts code blocks from a given text.
     *
     * @param {string} text - The text from which to extract code blocks. Code blocks are identified by triple backticks (```).
     * @returns {Array<Object>} - A list of objects where each object represents a code block and contains the following keys:
     *     - 'index' (number): The index of the code block in the text.
     *     - 'file_name' (string): An empty string. This field is not used in the current implementation.
     *     - 'content' (string): The content of the code block.
     *     - 'type' (string): The type of the code block. If the code block starts with a language specifier (like 'python' or 'java'), this field will contain that specifier. Otherwise, it will be set to 'language-specific'.
     *
     * Note: The function assumes that the number of triple backticks in the text is even.
     * If the number of triple backticks is odd, it will consider the rest of the text as the last code block.
     */

yesNo(question, context = "", maxAnswerLength = 50, conditioning = "") {
        /**
         * Analyzes the user prompt and answers whether it is asking to generate an image.
         *
         * @param {string} question - The user's message.
         * @param {string} context - The context for the question.
         * @param {number} maxAnswerLength - The maximum length of the generated answer.
         * @param {string} conditioning - An optional system message to put at the beginning of the prompt.
         * @returns {boolean} True if the user prompt is asking to generate an image, False otherwise.
         */
```
to use extractCodeBlocks, just do:
codes = tl.extractCodeBlocks(text to parse)
then test the length of codes and extract the content if required.
To use yesNo, just do:
if(tl.yesNo("question","context about which the question is asked"))


lollms can only process text, so if you need to do any operation that is not text generation, use the text output then parse it and perform the operation. if the user asks to access internet or recover the content of a page then do it manually and send the extracted text to lollms. use a context sie of 128000 and if possible make it configurable just as the host path.
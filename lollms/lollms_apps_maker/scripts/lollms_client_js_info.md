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
```


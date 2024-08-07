---

### Lollms Library Information

The Lollms library provides a robust framework for building applications that interact with Lollms as a client. This documentation will guide you through the essential steps to integrate and utilize the Lollms library effectively within your projects.

#### Importing Lollms in HTML

To start using the Lollms library in your HTML file, you need to include the following script tags:

```html
<script src="/lollms_js"></script>
<script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
```

> **Note:** The Lollms library requires Axios for making HTTP requests. Ensure that Axios is included in your HTML.

#### Initial Setup

Before using the Lollms client, you need to define a variable `ctx_size` which represents the size of the context for the LLM. This can be a fixed or modifiable value based on user requirements. It is recommended to store this value in local storage. The default context size is 4096 tokens.

Additionally, you can define `max_gen_size`, which indicates the maximum number of tokens the model can generate in one go. The default value for `max_gen_size` is also 4096 tokens.

#### Using Lollms Client in JavaScript

To generate text using the Lollms client, follow these steps:

1. **Build the Lollms Client:**

```javascript
const lc = new LollmsClient(
    host_address = null,  // Host address (default: http://localhost:9600 if null)
    model_name = null,    // Model name (default model used if null)
    ctx_size = 4096,      // Context size
    personality = -1,
    n_predict = 4096,     // Max tokens to be predicted
    temperature = 0.1,
    top_k = 50,
    top_p = 0.95,
    repeat_penalty = 0.8,
    repeat_last_n = 40,
    seed = null,
    n_threads = 8,
    service_key = "",
    default_generation_mode = ELF_GENERATION_FORMAT.LOLLMS
);
```

2. **Generate Text from a Prompt:**

To generate text, construct the prompt and use the `generate` method:

```javascript
// Build the prompt
const system_prompt = ""; // Instruction for the AI
const user_prompt = "";   // User prompt if applicable

let prompt = lc.system_message() + system_prompt + lc.template.separator_template + lc.ai_message();

if (user_prompt) {
    prompt = lc.system_message() + system_prompt + lc.template.separator_template + lc.user_message() + user_prompt + lc.template.separator_template + lc.ai_message();
}

// Generate text
const generated_text = await lc.generate(prompt);
```
if you want to send one or multiple images to the AI then use lc.generate_with_images instead of generate:
```javascript
// Generate text from a prompt and a list of images encoded in base64
const generated_text = await lc.generate(prompt, images);
```


#### Tokenization Functions

The `LollmsClient` also provides functions for tokenization and detokenization, enabling you to convert prompts to tokens and vice versa.

1. **Tokenize a Prompt:**

```javascript
async tokenize(prompt) {
    /**
     * Tokenizes the given prompt using the model's tokenizer.
     *
     * @param {string} prompt - The input prompt to be tokenized.
     * @returns {Array} A list of tokens representing the tokenized prompt.
     */
    const output = await axios.post("/lollms_tokenize", {"prompt": prompt});
    console.log(output.data.named_tokens);
    return output.data.named_tokens; // Returns a list of named tokens
}
```

- The `tokenize` function sends a prompt to the Lollms API and receives a response containing two types of tokens:
  - **raw_tokens:** A list of integer token IDs.
  - **named_tokens:** A list of lists, where each inner list contains a token ID and its corresponding string representation.

2. **Detokenize a List of Tokens:**

```javascript
async detokenize(tokensList) {
    /**
     * Detokenizes the given list of tokens using the model's tokenizer.
     *
     * @param {Array} tokensList - A list of tokens to be detokenized.
     * @returns {string} The detokenized text as a string.
     */
    const output = await axios.post("/lollms_detokenize", {"tokens": tokensList});
    console.log(output.data.text);
    return output.data.text; // Returns the detokenized text
}
```

- The `detokenize` function takes a list of token IDs and sends it to the Lollms API, which returns the corresponding text string.
Sure, here's the updated documentation with the `updateCode()` function added:

#### Advanced Usage with TasksLibrary

The `TasksLibrary` provides advanced functionalities such as summarization, yes/no questions, multi-choice questions, and code updating.

1. **Initialize TasksLibrary:**

```javascript
const tl = new TasksLibrary(lc); // lc is a LollmsClient object
```

2. **Summarize Text:**

```javascript
async function summarizeText(textChunk, summaryLength = "short") {
    return await tl.summarizeText(textChunk, summaryLength, lc.host_address, lc.model_name, lc.temperature, 1000);
}
```

3. **Extract Code Blocks:**

```javascript
const codes = tl.extractCodeBlocks(text);

// Process extracted code blocks
if (codes.length > 0) {
    codes.forEach(code => {
        console.log(code.content);
    });
}
```

4. **Yes/No Questions:**

```javascript
const isYes = await tl.yesNo("Is this a question?", "Context about the question", 50, "Optional conditioning message");

if (isYes) {
    console.log("The answer is yes.");
} else {
    console.log("The answer is no.");
}
```

5. **Update Code:**

```javascript
  updateCode(originalCode, queryString)
```

You can use the `updateCode()` function in your `TasksLibrary` class like this:

```javascript
const tl = new TasksLibrary();
const originalCode = `
def my_function(a, b):
    return a + b
my_function(2,5)
`;

const queryString = `
# REPLACE
# ORIGINAL
def my_function(a, b):
    return a + b
# SET
def my_function(x, y):
    return x * y
`;

const updatedCode = tl.updateCode(originalCode, queryString);
console.log(updatedCode);
```

This will output:

```python
def my_function(x, y):
    return x * y
my_function(2,5)
```

#### Important Considerations

- **Text Processing Only:** Lollms can only process text. For operations that are not text generation (e.g., accessing the internet, recovering the content of a page), manually perform the operation and send the extracted text to Lollms.
- **Context Size Configuration:** Use a context size of 128000 if required, and make it configurable along with the host path.
---

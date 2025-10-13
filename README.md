# MAT496_LangsmithAcademyCourse

IMPORTANT NOTES:

1. All relevant screenshots from the Langsmith dashboard etc are included in all the jupyter notebook files themselves. Please check them for thorough evauluation of work. 
2. Several Changes have been made and in summary:
    a) The most persistent change as seen in all files of Module 1 and 2 is that I have reworked the entire code to work with Google Gemini via Langchain instead of anything openai related as I did not want to pay for the openai API. Langchain integrations sometimes require specific things which lead to minor changes here and there all over the code to make it compatible for ex. difference in tool calling methods. 
    b) For module 2, instead of the dataset being the langchain documentation, i have incorporated a custom dataset which reflects in all files including app.py. The custom dataset is on the topic of high fantasy books. All experiments and evaluations have been done on this custom dataset and the code has been modified as such from the prompts to how I used the langsmith dashboard. I also made a custom LLM as Evaluator, and made custom splits for experimenting. All graphs and relevant results are attatched as screenshots in the jupyter notebook files alongside the code. 
    c) For Module 2, I also opted to use HuggingFace embeddings instead of openai embeddings or GoogleGenerativeAIEmbeddings (which i was using previously in place of openai) as soehow my embeddings quota ran out and wasnt working from different accounts either. Relevant changes have been made to incorporate the change in the embedding system.

## MODULE 1

### Lesson 1: Introduction to Tracing

[tracing_basicsVIDEO1MOD1.ipynb](./tracing_basicsVIDEO1MOD1.ipynb)

Projects serve as organizational containers that group related traces together. In this case, the project is a RAG (Retrieval-Augmented Generation) application. Runs represent individual units of work or operations within an LLM application, such as invoking a language model, querying a retriever, calling a tool, or executing a sub-chain. Traces are collections of interconnected runs that capture the complete end-to-end execution flow of a single operation or user request.

I used the `@traceable` decorator to monitor and trace function executions in LangSmith, gaining insight into how each component of the application operates. Through this process, I understood the fundamental concept of tracing.

---

### Lesson 2: Types of Runs

[types_of_runsVID2MOD1.ipynb](./types_of_runsVID2MOD1.ipynb)

This video explores the various run types encountered in LLM applications and their representation within traces.

**Run Types:**

* **LLM Runs:** Direct calls to language models for text generation
* **Retriever Runs:** Operations that fetch relevant documents or context from vector stores
* **Tool Runs:** Executions of external tools or functions (e.g., web search, calculations)
* **Chain Runs:** High-level workflows that orchestrate multiple operations together
* **Prompt Runs:** Template rendering and prompt construction operations

---

### Lesson 3: Alternative Tracing Methods

[alternative_tracing_methodsVID3MOD1.ipynb](./alternative_tracing_methodsVID3MOD1.ipynb)

This video examines various approaches to implementing tracing in LLM applications beyond the standard `@traceable` decorator.

**Tracing Methods Overview:**

* **@traceable (Decorator)**

  * The default and recommended method for setting up tracing
  * Automatically manages the RunTree structure, inputs, and outputs
  * Simplest implementation with minimal code changes

* **LangChain/LangGraph Integration**

  * Provides out-of-the-box tracing without manual configuration
  * Automatically traces all operations in a graph-like execution sequence
  * Best for applications already built with LangChain ecosystem

* **trace() Context Manager**

  * Offers granular control over which specific inputs and outputs get logged
  * Useful when you cannot use decorators or wrappers (e.g., not tracing a function directly)
  * Provides more flexibility than decorators for selective tracing

* **wrap_openai() Function**

  * Specialized wrapper for direct OpenAI SDK calls
  * Automatically tracks token usage, latency, and other OpenAI-specific metrics
  * Ideal when working with OpenAI API outside of LangChain abstractions

* **RunTree API**

  * Provides the lowest-level, most granular control over tracing configuration
  * Advanced option for complex custom implementations
  * Requires `LANGCHAIN_TRACING_V2=false` environment variable
  * Useful when you need access to `run_id` for purposes like adding user feedback or integrating with external logging services

---

### Lesson 4: Conversational Threads

[conversational_threadsVID4MOD1.ipynb](./conversational_threadsVID4MOD1.ipynb)

This video covers tracing in conversational applications where maintaining context across multiple turns is critical.

A **Thread** is an organizational tool that groups related traces together using a unique identifier (often generated with Python's `UUID`). Threads help visualize conversation flow across multiple user inputs and model responses.

**Key Benefits:**

* Debug issues that only appear in multi-turn conversations
* View the entire conversational flow rather than isolated messages
* Track how context is maintained throughout user interactions

---

## MODULE 2

### Video 1: Dataset Upload

[dataset_uploadVIDEO1MOD2.ipynb](./dataset_uploadVIDEO1MOD2.ipynb)

This video introduces datasets as a systematic approach to testing and benchmarking LLM applications.

**What are Datasets?**
Datasets are curated collections of input-output pairs used to evaluate LLM performance. Each pair typically consists of a test input (e.g., a question) and an expected output (reference answer).

**Key Benefits:**

* Organize test cases in a structured, reusable format
* Enable consistent, repeatable experiments across different model versions or configurations
* Track performance improvements and regressions over time
* Facilitate systematic benchmarking and comparison

**Dataset Creation Methods:**

* Import from CSV files for bulk upload
* Create programmatically using the LangSmith API
* Manually add examples through the LangSmith UI

---

### Video 2: Evaluators

[evaluatiorsVID2MOD2.ipynb](./evaluatorsVID2MOD2.ipynb)

This video covers **evaluators**, which are automated functions that assess LLM output quality.

Evaluators take a Run (actual output) and an Example (reference output) to calculate performance metrics, automating quality checks. In practice, we used evaluators to score LLM responses against reference answers on a 1–10 scale, measuring how accurately the model answered questions.

**Types of Evaluators:**

* **Exact Match:** Checks for identical outputs
* **Semantic Similarity:** Measures meaning similarity (e.g., 1–10 scale)
* **Custom Evaluators:** User-defined functions for specific needs (conciseness, tone, format)

---

### Video 3: Experiments

[experimentsVIDEO3MOD2.ipynb](./experimentsVIDEO3MOD2.ipynb)

This video explains how to set up and run experiments to systematically compare different LLM configurations using datasets and evaluators.

Experiments provide a flexible framework for evaluating datasets across different:

* Model versions (e.g., GPT-4 vs. Gemini)
* Prompt variations
* Dataset versions or splits
* Application configurations

**Features of Experiments:**

* **Multiple Runs:** Measure response consistency and variability
* **Concurrent Execution:** Run evaluations in parallel for faster results
* **Version Control:** Track which changes improve performance and which introduce regressions
* **Fair Comparisons:** Evaluate all variants on the same dataset for consistent comparisons

---

### Video 4: Analyzing Results of Experiments

This section focuses on interpreting and analyzing experiment results through the **LangSmith dashboard UI**, so there’s no uploaded matching Jupyter notebook file for this.

The LangSmith dashboard provides built-in metrics and visualizations to help you understand model performance across experiments. Through the UI, you can:

* Review aggregate performance metrics (accuracy, similarity scores, etc.)
* Compare results across different model versions or configurations
* Identify strengths and weaknesses in your LLM application

**Data Visualization:**

* Use charts and graphs to visualize performance trends
* Compare models side-by-side with visual metrics

## MODULE 3
[MODULE_3](./MODULE_3)

### Video 1: Playground Experiments

Prompts are of 2 types. Hard-coded prompts are the ones that we usually think of that are a sequence of strings. Prompt Templates give more flexibility because they have templated variables where information is filled in due to the user at run time.

Langsmith dashboard Playground specifically gives us the option to play around with prompts and prompt templates. Here I have played around with various system prompts and seen how the output drasticaly changes even with the same input Human question. 

We can change model providors and the models themselves along with setting parameters of chosen models such as the temperature. It is easy to track token usage etc along with being able to see how much of a different output different System Prompts have. We can also compare the performance of 2 different models.

To test the concept of output schemas I have created my own output schema that is an array of objects with each object being a book recommendation with other important fields. I have also implemented the same concept in the notebook wil the help of adding a custom made tool in the system prompt. I have also tested it over my custom dataset that I made. All of the screenshots and more details about the process are alongside my code in the notebooks.

### Video 2: Prompt Hub

We often hardcode our prompts directly into the code. This doesn't allow us to version our prompts as we iterate on the application. This video explores how we can use langsmith's Prompt Hub for storing our prompts. We can store a system message, human message, etc thats passed to the chatbot. We can provide templated variables that allow the user to input the variables at runtime. Information is stored including model providor, model, hyperparameters like temperature, along with things like an output schema. This makes overall a versioning scheme.We learn how to save prompts, fork them for new versions, and pull them into our code using the SDK. 

In this module in the file you can see changes I've made into making a custom output schema and calling it in my code and various ways I've played around with the prompt Hub.

### Video 3: Prompt Engineering Lifecycle 

In previous modules, we used a utils.py file. In this video, we are taught a practical example on how to use the plaground and prompt hub together in the RAG application instead of connecting to a utils.py file. I used the same app.py file I used before, where instead of embedding the langsmith documentation I just use a custom dataset that I have used previously about fantasy books. 

In this video, we start off by opening a trace of a user question like we did in previous modules, however, now we open it in playground and learn how to optimize the prompt by using variables and changing the prompt itself to gain a better output. We then implement this in the dataset as well. By refining our prompts and testing them with datasets, we can ensure our code always uses the latest version of the prompt. This workflow makes it easier for us to improve our applications without constantly updating the code.

## Video 4: Prompt Canvas 

There are no files for this video as it is on the Langsmith Dashboard.

This video covers the feature of the Prompt canvas that allows us to work with LLMs to improve upon our prompt templates according to our specific requirements. This results in better outputs. We can ask followup questions and make followup edits to improve upon our prompts further and rewrite them. We can edit the prompts ourselves in the prompt canvas as well, and we can also see the difference/similarity with older versions. 

<img width="1793" height="903" alt="image" src="https://github.com/user-attachments/assets/8e2b72d2-e769-4172-834a-91a87179588d" />



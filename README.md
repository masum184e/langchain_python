# Language Model

Language Model is a computer program that analyze a given sequence of words and provide a basis for their word prediction. Language model is used in AI, NLP, NLU, NLG system, particularly ones that perform text generation, machine translation and question answering.

__LLM - Large Language Model__ are are designed to understand and generate human language at scale. **GPT**, **BERT**.

__MLM - Masked Language Model__ are a specific type of language model that predicts masked or hidden or blank words in a sentence.

__CLM - Casual Language Model__ generate text sequentially, one token at a time, based only on the tokens that came before it in the input sequence. It basically predict next word based on previous word

Here's how a typical language model works:

1. *Input:* The process starts with the user providing input in the form of text. This input can be a question, a prompt for generating text, or any other form of communication.

2. *Tokenization:* The input text is split into smaller units called tokens. These tokens could be words, subwords, or even characters, depending on the model architecture and tokenization strategy used.

3. *Embedding:* Each token is then converted into a numerical representation called word embeddings or token embeddings. These embeddings capture the semantic meaning of the tokens and their relationships with other tokens.

4. *Processing:* The embeddings of the tokens are fed into the model's neural network architecture. This network consists of multiple layers of processing units (neurons) that transform the input embeddings through various mathematical operations.

5. *Contextual Understanding:* As the input propagate through the network, the model learns to understand the contextual relationships between the tokens. It allow the model to focus on relevant parts of the input.

6. *Prediction:* Based on its understanding of the input text and the context provided, the model generates a response. 

7. *Output:* The model outputs the predicted tokens, which can be used to generate text or to perform other tasks such as text classification, translation, or summarization.

# Large Language Model
Large language model is a machine learning model designed to understand, generate, and manipulate human language on a vast scale. These models are typically built using deep learning techniques, especially variants of the transformer architecture, and are trained on massive datasets of text from the internet and other sources.

# Generative AI
Generative AI refers to deep-learning models that can generate high-quality text, images, and other content based on the data they were trained on.

## Quick Information
- GPT(Generative Pre-trained Transformer) is a series of llm developed by OpenAI
- ChatGPT is a generative AI specifically fine-tuned for conversational interactions.
- OpenAI's work best with JSON while Anthropic's models work best with XML.

# Langchain
LangChain is an open source framework for building applications based on large language models (LLMs). It provides tools and abstractions to improve the customization, accuracy, and relevancy of the information the models generate. Basically it integrate ai(LLm model) with web/mobile applications. By abstracting complexities, it simplifies the process compared to direct integration, making it more accessible and manageable. The core element of any language model application is...the model. LangChain gives you the building blocks to interface with any language model.

## Installation
```
pip install langchain
```

# Model I/O - OpenAI
Language models in LangChain come in two flavors:

__ChatModels:__ The ChatModel objects take a list of messages as input and output a message. Chat models are often backed by LLMs but tuned specifically for having conversations. 

__LLM:__ LLMs in LangChain refer to pure text completion models. The LLM objects take string as input and output string. OpenAI's GPT-3 is implemented as an LLM.

The LLM returns a string, while the ChatModel returns a message. The main difference between them is their input and output schemas.  

## Installation
```
pip install langchain-openai
```

## Initialize The Model
We can see the difference between an LLM and a ChatModel when we invoke it.

```
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",api_key="...")
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125",api_key="...")

text = "What would be a good company name for a company that makes colorful socks?"
print("LLM Response: "+llm.invoke(text))

messages = [HumanMessage(content=text)]
print("Chat Model: "+chat_model.invoke(messages))
```

__Reference:__ [OpenAI Model List](https://platform.openai.com/docs/models), [OpenAI](https://api.python.langchain.com/en/latest/llms/langchain_openai.llms.base.OpenAI.html), [ChatOpenAI](https://api.python.langchain.com/en/latest/llms/langchain_openai.llms.base.OpenAI.html), [HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html)

## Prompt Templates
Most LLM applications do not pass user input directly into an LLM. Usually they will add the user input to a larger piece of text, that provides additional context on the specific task at hand so that llm can understand user input more efficiently.

Typically, language models expect the prompt to either be a string or else a list of chat messages. Use `PromptTemplate` to create a template for a string prompt and `ChatPromptTemplate` to create a list of messages

If the user only had to provide the description of a specific topic but not the instruction that model needs, it would be great!! PromptTemplates help with exactly this! It bundle up all the logic & instruction going from user input into a fully fromatted prompt that llm model required.

```
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",api_key="...")
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125",api_key="...")

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt.format(product="colorful socks")

# PROMPT FROM PROMPT TEMPLATE
print("Prompt: "+prompt)

# LLM RESPONSE
print("LLM Response: "+llm.invoke(text))

# CHAT MODEL RESPONSE
messages = [HumanMessage(content=text)]
print("Chat Model: "+chat_model.invoke(messages))
```

__Reference:__ [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)

## ChatPromptTemplate
Each chat message is associated with content, and an additional parameter called `role`. For example, in the OpenAI Chat Completions API, a chat message can be associated with an AI assistant, a human or a system role.

```
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",api_key="...")
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125",api_key="...")

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI bot. Your name is {name}."),
        ("human", "Hello, how are you doing?"),
        ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

prompt = chat_template.format_messages(name="Bob", user_input="What is your name?")

# PROMPT FROM PROMPT TEMPLATE
print("Prompt: "+prompt)

# LLM RESPONSE
print("LLM Response: "+llm.invoke(text))

# CHAT MODEL RESPONSE
messages = [HumanMessage(content=text)]
print("Chat Model: "+chat_model.invoke(messages))
```

__Reference:__ [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)

## Message Prompts
LangChain provides different types of MessagePromptTemplate. The most commonly used are `AIMessagePromptTemplate`, `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate`, which create an AI message, system message and human message respectively.

All messages have a role and a content property. The role describes WHO is saying the message. The content property describes the content of the message. This can be a few different things:

__Reference:__ [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)

## Output parsers
OutputParsers convert the raw output of a language model into a format that can be used downstream.
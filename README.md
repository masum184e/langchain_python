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

# Langchain
LangChain is an open source framework for building applications based on large language models (LLMs). It provides tools and abstractions to improve the customization, accuracy, and relevancy of the information the models generate. Basically it integrate ai(LLm model) with web/mobile applications. By abstracting complexities, it simplifies the process compared to direct integration, making it more accessible and manageable.
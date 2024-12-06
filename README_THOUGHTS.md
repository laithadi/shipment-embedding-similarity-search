### Note: Understanding the Two README Files

This project contains two README files to provide clear and detailed context:

1. **`README_RUN_CODE.md`**: This file contains all the instructions for setting up, running the code, and interpreting the results. If you’re looking to execute the code and analyze its outputs, this is the file you should start with.

2. **`README_THOUGHTS.md`**: This file explains my thought process, design decisions, challenges faced, and alternative approaches considered while solving the coding challenge. It offers deeper insight into the logic and reasoning behind the implementation. 

Feel free to refer to each README based on your purpose: whether you want to run the code or understand the rationale behind it!

## README: Thoughts and Approach

This document provides a high-level overview of my thought process, design choices, and technical decisions for the coding challenge. It explains the rationale behind the solution and highlights key considerations made during implementation.

## High-Level Thinking: Interpreting the Question

At its core, my understanding of this challenge is straightforward: given a user query in textual form, identify the most relevant or similar piece of information from the provided dataset and return it. However, as simple as the task sounds, it comes with two significant challenges:

1. Handling different data types (numerical, categorical, and date) in a unified manner while embedding them for similarity comparison.
2. Maintaining or, at the very least, considering ordinality in the search, especially for numerical and date fields.

### The Initial Approach

My instinct, like many engineers, was to dive straight into building the embedding model. Questions like "What’s the best embedding model for categorical, numerical, and date data types?" or "How do I encode ordinality effectively?" immediately came to mind. I even considered using separate embedding models for each data type, thinking it would lead to better results.

But from my experience, rushing into the deep end of a task rarely yields an optimal solution. Engineering is not just about coding; it’s about stepping back, asking the right questions, and understanding the problem from all angles. So, I paused and decided to start with the basics.

### Grounding the Problem: Understanding the Data

I began by exploring the dataset. Instead of immediately tackling embedding, I chose to analyze the data—performing basic statistical summaries and examining its structure. This exploration helped me get a clearer picture of the challenges and opportunities within the dataset.

Then I asked myself a question that is often overlooked: *“Do I really need separate embedding models for each data type?”* It’s easy to fall into the trap of assuming that more models—or the biggest and most complex models—will inherently lead to better results. But that’s not always true. I shifted my perspective to, *“How can I preprocess the data to simplify the task? How can I solve the problem without overengineering?”*

From my experience, the quality of a solution almost always starts with the data. It’s about how you process and prepare it, not just the sophistication of the AI models. With that in mind, I committed to simplifying the embedding model by focusing on preprocessing. The goal: use a single embedding model for all data types—categorical, numerical, and date.

### The Balancing Act

While I acknowledge that there are more complex and potentially better-performing approaches, I chose to focus on balancing simplicity, functionality, and clarity. This wasn’t about showcasing flashy techniques or adding unnecessary complexity. Instead, my solution prioritizes ease of use, interpretability, and scalability while meeting the requirements of the challenge.

In the following sections, I’ll discuss my decisions in greater detail, including the trade-offs I considered and the other approaches I explored. This process was about walking the fine line between engineering precision and practicality, with the end-user experience always in mind.

## Software Engineering Practices

As an engineer, I prioritize writing clean, reusable, and maintainable code. This philosophy is evident throughout the implementation of this challenge.

### Documentation and Testing
The majority of the classes and methods in the project include detailed documentation in the form of Google-style docstrings. Each function signature specifies the type of every argument and the return value, ensuring clarity for future users and contributors. Unit tests are also provided for most modules, enabling quick validation of functionality and significantly reducing debugging time. Even though this project is relatively small in scale, the modular testing approach saved me considerable time and effort by catching issues early during development.

### Object-Oriented Programming
Object-oriented programming (OOP) played a role in my design approach. I enjoy building objects that are both generalized and reusable, enabling flexibility and simplifying future development.

### Organized Code Structure
I maintain clean and organized code structures by leveraging `__init__.py` files and adhering to clear folder organization. This helps keep the codebase readable and modular, making it straightforward for others to navigate and understand the project.

### Logging and Debugging
I used Python’s `logging` package throughout the project to provide real-time insights into the execution flow. Logging serves as an invaluable tool for understanding how data flows through the code and for debugging, especially when scaling or adapting a solution to more complex problems. This is a regular practice in my development workflow, as it provides a clear and systematic way to monitor and troubleshoot issues.

<!-- ## My Approach: Balancing Engineering and Business Needs

In this section, I will explain:
- How I approached the solution as both an engineer and a business-minded thinker.
- The importance of simplicity for users and maintainability for engineers.
- Why I chose to use a single embedding model for all data types (text, numerical, and date) to avoid over-engineering. -->

## Approach

### Choice of Embedding Model

For this challenge, I chose to use **DistilBERT**, a smaller and faster version of BERT, to embed textual, numerical, and date data. The decision to use DistilBERT was driven by its ability to provide high-quality embeddings while being computationally efficient. Below, I will dive into the reasoning, configuration, and suitability of DistilBERT for this task.

#### Why DistilBERT?

1. **Compact and Efficient**: DistilBERT compresses the original BERT model while retaining most of its performance. This makes it an excellent choice for scenarios where efficiency matters without significantly sacrificing accuracy.
2. **Pre-training Data**: DistilBERT was pre-trained on massive datasets such as:
   - **BooksCorpus**: 800M words from book texts.
   - **English Wikipedia**: 2,500M words from textual passages, ignoring lists and tables.
   These datasets provide DistilBERT with exposure to numerical data, dates, and contextual language usage, making it well-suited for embedding structured and unstructured data.
   - For example, numbers like "1" and dates like "2023-07-01" can be effectively embedded as they frequently appear in pre-training datasets.
3. **GLUE Benchmark**: DistilBERT has shown strong performance on the **GLUE benchmark**, a suite of tasks measuring natural language understanding. Tasks like textual entailment, sentence similarity, and question answering align closely with the challenge's objective of matching user queries to dataset values.

*[BERT paper](https://arxiv.org/pdf/1810.04805).*

*[HugginFace: glue-benchmark](https://huggingface.co/blog/bert-101#43-glue-benchmark).*

#### Tokenization and WordPiece Embedding

DistilBERT uses **WordPiece embedding**, a tokenization technique that splits text into subword units. This ensures efficient handling of unknown or rare words and provides better coverage for numerical and date data. For example:
- The word "embedding" might be tokenized into ["em", "##bed", "##ding"].
- Dates like "2023-07-01" processed to "july 1 2023" are tokenized into meaningful subwords such as ["july", "1", "2023"].

Key properties of WordPiece embedding in DistilBERT:
1. **30,000 Token Vocabulary**: A balance between vocabulary size and computational efficiency.
2. **Special Tokens**: The [CLS] token represents the entire input sequence and is used for classification or similarity tasks.
3. **Input Representation**: Each token's input representation is constructed by summing token embedding, segment embedding, and position embedding.

#### Model and Tokenizer Configuration

To optimize DistilBERT for this challenge, the tokenizer was configured as follows:
- **`max_length=100`**: Limits tokenized sequences to 100 tokens, sufficient to handle typical user queries and dataset values without truncating important information.
- **`truncation=True`**: Ensures inputs exceeding the maximum length are truncated safely.
- **`padding=True`**: Pads shorter sequences to the same length, enabling batch processing.
- **`return_tensors="pt"`**: Returns PyTorch tensors for compatibility with the DistilBERT model.

#### Implementation Details: Parent and Child Classes

To handle text embedding systematically, I implemented a parent class called **`HuggingFaceEmbedding`** and a child class **`DistilBertTextEmbedding`**. 

- **Parent Class (`HuggingFaceEmbedding`)**:
  - Provides a reusable framework for any HuggingFace model.
  - Initializes the tokenizer and model and provides a method (`encode`) to embed text.
  - Includes configurable parameters for tokenization, such as truncation, padding, and maximum length, offering flexibility for different models and tasks.

- **Child Class (`DistilBertTextEmbedding`)**:
  - Inherits from `HuggingFaceEmbedding`.
  - Specifically initializes the DistilBERT tokenizer and model.
  - Defines an `embed_text` method that leverages the parent class’s `encode` method to process text input.

This design ensures modularity and reusability. If a different embedding model (e.g., RoBERTa or GPT) is needed in the future, a new child class can be implemented with minimal changes.

#### Why This Approach Works for Text, Numbers, and Dates

1. **Textual Data**: DistilBERT was trained on diverse corpora, making it well-suited for embedding natural language text, including categorical columns like "Product_Category."
2. **Numerical Data**: Numbers can be converted to their string representations (e.g., "1" or "2023"). BERT’s pre-training on datasets with numerical contexts ensures meaningful embeddings for such data.
3. **Date Data**: Dates are preprocessed into human-readable formats like "July 1 2023." This approach aligns with BERT’s training data, which frequently includes similar date representations.

More on data processing in the next section. 

### Data Processing
- Why I focused on data quality and preprocessing as the foundation of the solution.
- The rationale behind processing only unique values in a column for efficiency.
- Detailed explanation of how numerical and date columns were preprocessed.

### Similarity Calculation
- The decision to use the dot product for similarity calculations and its advantages.
- Why I chose to embed one value at a time from a column instead of the entire column, ensuring accuracy and precision in similarity scoring.

## Other Approaches

### text to sql model 

### Table Question Answering (Table QA)

### embedding model for each data type (numerical, date, categorical)

### other similarity calcualtion methods 

## things i could add on to current solution 
- threshold so if no col is relevant to the query 

## Error in the solution 
- 1 is the min it found, when there is 0


## Final Thoughts

This section will include:
- Reflections on the solution and areas for potential improvement.
- How this approach aligns with practical business needs and technical scalability.
- Closing remarks on the experience of tackling the challenge.

# side notes 
- create a data class (kyle) mentioned if that helps instead of having stuff in the __init__.py 
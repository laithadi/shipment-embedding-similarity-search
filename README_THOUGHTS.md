## Note: Understanding the Two README Files

This project contains two README files to provide clear and detailed context:

1. **`README_RUN_CODE.md`**: This file contains all the instructions for setting up, running the code, and interpreting the results. If you’re looking to execute the code and analyze its outputs, this is the file you should start with.

2. **`README_THOUGHTS.md`**: This file explains my thought process, design decisions, challenges faced, and alternative approaches considered while solving the coding challenge. It offers deeper insight into the logic and reasoning behind the implementation. 

Feel free to refer to each README based on your purpose: whether you want to run the code or understand the rationale behind it!

# README: Thoughts and Approach

This document provides a high-level overview of my thought process, design choices, and technical decisions for the coding challenge. It explains the rationale behind the solution and highlights key considerations made during implementation.

# High-Level Thinking: Interpreting the Question

At its core, my understanding of this challenge is straightforward: given a user query in textual form, identify the most relevant or similar piece of information from the provided dataset and return it. However, as simple as the task sounds, it comes with two significant challenges:

1. Handling different data types (numerical, categorical, and date) in a unified manner while embedding them for similarity comparison.
2. Maintaining or, at the very least, considering ordinality in the search, especially for numerical and date fields.

## The Initial Approach

My instinct, like many engineers, was to dive straight into building the embedding model. Questions like "What’s the best embedding model for categorical, numerical, and date data types?" or "How do I encode ordinality effectively?" immediately came to mind. I even considered using separate embedding models for each data type, thinking it would lead to better results.

But from my experience, rushing into the deep end of a task rarely yields an optimal solution. Engineering is not just about coding; it’s about stepping back, asking the right questions, and understanding the problem from all angles. So, I paused and decided to start with the basics.

## Grounding the Problem: Understanding the Data

I began by exploring the dataset. Instead of immediately tackling embedding, I chose to analyze the data—performing basic statistical summaries and examining its structure. This exploration helped me get a clearer picture of the challenges and opportunities within the dataset.

Then I asked myself a question that is often overlooked: *“Do I really need separate embedding models for each data type?”* It’s easy to fall into the trap of assuming that more models—or the biggest and most complex models—will inherently lead to better results. But that’s not always true. I shifted my perspective to, *“How can I preprocess the data to simplify the task? How can I solve the problem without overengineering?”*

From my experience, the quality of a solution almost always starts with the data. It’s about how you process and prepare it, not just the sophistication of the AI models. With that in mind, I committed to simplifying the embedding model by focusing on preprocessing. The goal: use a single embedding model for all data types—categorical, numerical, and date.

## The Balancing Act

While I acknowledge that there are more complex and potentially better-performing approaches, I chose to focus on balancing simplicity, functionality, and clarity. This wasn’t about showcasing flashy techniques or adding unnecessary complexity. Instead, my solution prioritizes ease of use, interpretability, and scalability while meeting the requirements of the challenge.

In the following sections, I’ll discuss my decisions in greater detail, including the trade-offs I considered and the other approaches I explored. This process was about walking the fine line between engineering precision and practicality, with the end-user experience always in mind.

# Software Engineering Practices

As an engineer, I prioritize writing clean, reusable, and maintainable code. This philosophy is evident throughout the implementation of this challenge.

## Documentation and Testing
The majority of the classes and methods in the project include detailed documentation in the form of Google-style docstrings. Each function signature specifies the type of every argument and the return value, ensuring clarity for future users and contributors. Unit tests are also provided for most modules, enabling quick validation of functionality and significantly reducing debugging time. Even though this project is relatively small in scale, the modular testing approach saved me considerable time and effort by catching issues early during development.

## Object-Oriented Programming
Object-oriented programming (OOP) played a role in my design approach. I enjoy building objects that are both generalized and reusable, enabling flexibility and simplifying future development.

## Organized Code Structure
I maintain clean and organized code structures by leveraging `__init__.py` files and adhering to clear folder organization. This helps keep the codebase readable and modular, making it straightforward for others to navigate and understand the project.

## Logging and Debugging
I used Python’s `logging` package throughout the project to provide real-time insights into the execution flow. Logging serves as an invaluable tool for understanding how data flows through the code and for debugging, especially when scaling or adapting a solution to more complex problems. This is a regular practice in my development workflow, as it provides a clear and systematic way to monitor and troubleshoot issues.

<!-- ## My Approach: Balancing Engineering and Business Needs

In this section, I will explain:
- How I approached the solution as both an engineer and a business-minded thinker.
- The importance of simplicity for users and maintainability for engineers.
- Why I chose to use a single embedding model for all data types (text, numerical, and date) to avoid over-engineering. -->

# Approach

## Choice of Embedding Model

For this challenge, I chose to use **DistilBERT**, a smaller and faster version of BERT, to embed textual, numerical, and date data. The decision to use DistilBERT was driven by its ability to provide high-quality embeddings while being computationally efficient. Below, I will dive into the reasoning, configuration, and suitability of DistilBERT for this task.

### Why DistilBERT?

1. **Compact and Efficient**: DistilBERT compresses the original BERT model while retaining most of its performance. This makes it an excellent choice for scenarios where efficiency matters without significantly sacrificing accuracy.
2. **Pre-training Data**: DistilBERT was pre-trained on massive datasets such as:
   - **BooksCorpus**: 800M words from book texts.
   - **English Wikipedia**: 2,500M words from textual passages, ignoring lists and tables.
   These datasets provide DistilBERT with exposure to numerical data, dates, and contextual language usage, making it well-suited for embedding structured and unstructured data.
   - For example, numbers like "1" and dates like "2023-07-01" can be effectively embedded as they frequently appear in pre-training datasets.
3. **GLUE Benchmark**: DistilBERT has shown strong performance on the **GLUE benchmark**, a suite of tasks measuring natural language understanding. Tasks like textual entailment, sentence similarity, and question answering align closely with the challenge's objective of matching user queries to dataset values.

*[BERT paper](https://arxiv.org/pdf/1810.04805).*

*[HugginFace: glue-benchmark](https://huggingface.co/blog/bert-101#43-glue-benchmark).*

### Tokenization and WordPiece Embedding

DistilBERT uses **WordPiece embedding**, a tokenization technique that splits text into subword units. This ensures efficient handling of unknown or rare words and provides better coverage for numerical and date data. For example:
- The word "embedding" might be tokenized into ["em", "##bed", "##ding"].
- Dates like "2023-07-01" processed to "july 1 2023" are tokenized into meaningful subwords such as ["july", "1", "2023"].

Key properties of WordPiece embedding in DistilBERT:
1. **30,000 Token Vocabulary**: A balance between vocabulary size and computational efficiency.
2. **Special Tokens**: The [CLS] token represents the entire input sequence and is used for classification or similarity tasks.
3. **Input Representation**: Each token's input representation is constructed by summing token embedding, segment embedding, and position embedding.

### Model and Tokenizer Configuration

To optimize DistilBERT for this challenge, the tokenizer was configured as follows:
- **`max_length=100`**: Limits tokenized sequences to 100 tokens, sufficient to handle typical user queries and dataset values without truncating important information.
- **`truncation=True`**: Ensures inputs exceeding the maximum length are truncated safely.
- **`padding=True`**: Pads shorter sequences to the same length, enabling batch processing.
- **`return_tensors="pt"`**: Returns PyTorch tensors for compatibility with the DistilBERT model.

### Implementation Details: Parent and Child Classes

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

### Why This Approach Works for Text, Numbers, and Dates

1. **Textual Data**: DistilBERT was trained on diverse corpora, making it well-suited for embedding natural language text, including categorical columns like "Product_Category."
2. **Numerical Data**: Numbers can be converted to their string representations (e.g., "1" or "2023"). BERT’s pre-training on datasets with numerical contexts ensures meaningful embeddings for such data.
3. **Date Data**: Dates are preprocessed into human-readable formats like "July 1 2023." This approach aligns with BERT’s training data, which frequently includes similar date representations.

More on data processing in the next section. 

## Data Processing

At a high level, three main steps were taken to process the data and prepare it for embedding and similarity calculations. These steps ensured that the data was formatted in a way that aligned with the strengths of the embedding model and addressed the challenge of working with multiple data types (numerical, date, and categorical).

### Filtering Relevant Columns
The first step involved filtering the dataset to include only the columns specified by the user. The user provides a list of column names in the `user_input/df_cols.txt` file, and the program reads this file to determine which columns to keep. This ensures the analysis focuses only on the relevant data, reducing computational overhead and improving clarity.

### Converting Date Columns
Next, the date columns, often represented as objects or strings in the dataset, were converted to a proper `datetime` data type using `pandas.to_datetime()`. This step is critical for distinguishing between date and string data types, which directly impacts how the data is processed and embedded later. By converting these columns to the correct data type, we ensure that they are handled appropriately during subsequent processing.

### String Conversion and Column Name Concatenation
This is the most significant step in the data processing pipeline. For every numerical or date column, the values were converted into a string and concatenated with the column name. The column name was stripped of underscores or other separators to produce a clean, human-readable string.

For example:
- A value of `3` in the `Days_from_shipment_to_delivery` column would be processed into `"days from shipment to delivery 3"`.
- A date value like `2023-02-20` in the `Estimated_Arrival_Date` column would be processed into `"estimated arrival date on February 20 2023"`.

The inclusion of "on" for date columns ensures the final string reads naturally in English, providing better context for the embedding model.

Processed values were written to new columns prefixed with `s_`. For example, `Days_from_shipment_to_delivery` would produce a new column named `s_Days_from_shipment_to_delivery`. This allowed the program to retain the original values and easily retrieve them later when generating outputs.

### Why This Approach Works
This data processing step aligns with the strengths of NLP models like BERT, which excel at handling textual data. BERT was pre-trained on large text corpora, such as Wikipedia and BooksCorpus, where the input consisted of natural language text. By converting numerical and date values into descriptive strings, the input provides additional context, helping the model generate better embeddings.

Consider the difference between embedding a standalone number like `3` versus `"days from shipment to delivery 3"`. The latter provides the model with crucial information about what the number represents, making the resulting embedding far more meaningful for similarity calculations. 

Similarly, converting `2023-02-20` to `"estimated arrival date on February 20 2023"` leverages BERT's ability to interpret the semantic meaning of natural language. The processed string reads like a natural language phrase, aligning better with the model's training data and ensuring that the embedding captures not just the value but also its contextual significance.

### Benefits for Similarity Calculations
By providing this level of detail in the input strings:
1. **Improved Contextual Understanding:** The model can distinguish between columns and their significance based on the concatenated column name.
2. **Enhanced Semantic Representation:** The processed strings ensure that numerical and date values are represented in a way that is semantically rich and aligned with natural language.
3. **Consistency Across Data Types:** By converting all data types into strings, the embedding process becomes uniform, eliminating the need for multiple specialized models for different data types.

This method emphasizes simplicity and effectiveness, leveraging the strengths of the chosen embedding model while ensuring the data is prepared in a way that supports robust similarity calculations.

## Similarity Calculation

The calculation of similarity between the user query and each value in the dataset relies on **cosine similarity**. This metric is widely used in NLP and machine learning for comparing the similarity of vector representations, such as embeddings.

### Cosine Similarity: Overview
Cosine similarity measures the cosine of the angle between two non-zero vectors in a multi-dimensional space. The result ranges from:
- **1**: When the vectors are identical (angle = 0°).
- **0**: When the vectors are orthogonal (angle = 90°), indicating no similarity.
- **-1**: When the vectors are opposite in direction (angle = 180°), indicating negative correlation.

The formula for cosine similarity is:

cosine similarity = (A · B) / (||A|| * ||B||)

Where:
- A · B: The dot product of the two vectors.
- ||A||: The magnitude (Euclidean norm) of vector A.
- ||B||: The magnitude (Euclidean norm) of vector B.

### Code Explanation
The following snippet implements the cosine similarity calculation in Python:

```python
similarity_score = np.dot(embedded_query, embedded_value) / (
    np.linalg.norm(embedded_query) * np.linalg.norm(embedded_value)
)
```
[View the code here](src/utils/_find_best_match.py)


### Explanation

#### Numerator: `np.dot(embedded_query, embedded_value)`
- Computes the **dot product** of the two vectors, `embedded_query` and `embedded_value`.
- The dot product measures the degree of alignment between the vectors.

#### Denominator: `np.linalg.norm(embedded_query) * np.linalg.norm(embedded_value)`
- `np.linalg.norm(embedded_query)`: Calculates the **magnitude** (Euclidean norm) of the `embedded_query` vector.
- `np.linalg.norm(embedded_value)`: Calculates the **magnitude** of the `embedded_value` vector.
- The product of these magnitudes normalizes the similarity score, ensuring it depends only on the vectors' directions.

#### Division
- Dividing the dot product by the product of magnitudes yields the **cosine similarity**, capturing the cosine of the angle between the vectors.

### Intuition for Cosine Similarity
- **Highly similar vectors**: Cosine similarity approaches **1**.
- **Dissimilar or orthogonal vectors**: Cosine similarity approaches **0**.
- This metric is particularly useful for embeddings because it focuses on their semantic alignment rather than their magnitudes.

### Why Cosine Similarity Was Chosen
In this challenge:
- **Query Embedding (`embedded_query`)**: Represents the embedding of the user's textual query.
- **Value Embedding (`embedded_value`)**: Represents the embedding of a specific value from a column in the dataset.

Cosine similarity is an excellent choice because:
1. It evaluates how semantically aligned the query is with each dataset value.
2. It ignores magnitude differences between embeddings, focusing solely on their **directional similarity**.
3. It allows ranking of values in the dataset based on their relevance to the query, providing a clear measure for selecting the best match.

### Calculation Strategy

The similarity scores were calculated on the **unique values** in each column, one at a time. Instead of embedding all unique values in a column into a matrix (e.g., shape (number of unique values, 768)), and then calculating the dot product by multiplying the user query embedding (1, 768) with this matrix, we opted for a simpler approach:

- **Each unique value in the column is embedded individually.**
- **The dot product is computed for the query embedding (1, 768) and the value embedding (1, 768) one at a time.**

This approach offers the following advantages:

- **Simplicity:** By processing values one at a time, the implementation is straightforward and avoids additional matrix manipulation.
- **Memory Efficiency:** Instead of holding a large embedding matrix in memory, this approach processes smaller data objects sequentially, reducing memory overhead.
- **Flexibility:** It allows for fine-grained debugging and logging of individual similarity scores for each value in the column.

This strategy balances computational efficiency with clarity, ensuring the solution is both scalable and easy to understand.

# Other Approaches

While implementing the solution for this challenge, I explored several alternative approaches that could have been viable or even improved the results. Below, I discuss these approaches, their potential benefits, and why I ultimately decided not to implement them.

## Text-to-SQL Models

Text-to-SQL models are designed to convert natural language queries into SQL statements that can directly operate on structured data. This approach is particularly appealing when working with tabular datasets like the one provided in this challenge.

### Why Consider It?
- **Direct Query Execution:** Text-to-SQL models enable the generation of SQL queries that can fetch precise results from the dataset without requiring complex embedding or similarity calculations.
- **Scalability:** These models are powerful when dealing with large-scale relational databases where embeddings might be computationally expensive.
- **Existing Pretrained Models:** Open-source Text-to-SQL models (e.g., T5-based models) are readily available and can save significant development time.

### Why It Wasn't Used
- **Accuracy Issues:** My experiments with a Text-to-SQL model yielded suboptimal queries that didn’t align well with the user inputs or the dataset schema. This highlighted a challenge in adapting general-purpose models to the specific nuances of this dataset.
- **Ambiguity in Queries:** User queries in this challenge are often ambiguous and rely heavily on semantic understanding, which Text-to-SQL models may struggle with.
- **Complexity of Fine-Tuning:** Fine-tuning a Text-to-SQL model to perform well on this dataset would have been resource-intensive and less straightforward than the embedding-based approach.

## Table Question Answering (Table QA)

Table QA models are designed for answering natural language questions directly over tabular data without the need for SQL or manual query formulations.

### Why Consider It?
- **End-to-End Simplicity:** These models abstract away the need for manually embedding data and user queries.
- **Semantic Understanding:** Table QA models often combine text and numerical reasoning, making them potentially ideal for this challenge.

### Why It Wasn't Used
- **Model Generalization:** Pretrained Table QA models might not generalize well to the challenge’s dataset, which includes unique numerical and date processing requirements.
- **Limited Control:** Using a Table QA model would reduce the ability to finely control and debug the similarity calculation process, which was critical for this challenge.
- **Overhead:** Training or fine-tuning a Table QA model would have required significant effort, overshadowing the simplicity of embedding and cosine similarity.

## Embedding Model for Each Data Type (Numerical, Date, Categorical)

Another approach would have been to use specialized embedding models for each data type:
- **Numerical Data:** Models that can better understand ordinality, such as embedding the numerical values along a learned scale.
- **Date Data:** Time-specific embeddings that could capture relationships between dates (e.g., temporal distance).
- **Categorical Data:** NLP-based models like BERT or Word2Vec.

### Why Consider It?
- **Data-Type Specific Optimization:** Each embedding model would specialize in handling the intricacies of its respective data type, potentially leading to higher-quality embeddings.
- **Fine-Grained Insights:** Such embeddings could capture nuanced relationships within each data type, leading to better similarity calculations.

### Why It Wasn't Used
- **Overengineering Risk:** Introducing multiple embedding models would add complexity without guaranteeing significant improvement for the relatively simple user queries in this challenge.
- **Uniformity:** Using a single embedding model (DistilBERT) across all data types ensures uniformity and reduces computational and implementation overhead.
- **Processing Overlap:** By preprocessing the numerical and date data into textual representations, the single embedding model performed adequately for all data types.

## Other Similarity Calculation Methods

Beyond cosine similarity, several other similarity metrics could have been considered:
- **Euclidean Distance:** Measures the straight-line distance between two embeddings.
- **Manhattan Distance:** Captures differences along individual dimensions of the embeddings.
- **Jaccard Similarity:** Useful for comparing sets or categorical data.
- **Dot Product Without Normalization:** Measures raw alignment without considering magnitude.

### Why Consider It?
- **Task-Specific Metrics:** Some metrics may better capture relationships in specific data types.
- **Experimentation Potential:** Testing multiple metrics could provide insights into the strengths and weaknesses of cosine similarity.

### Why It Wasn't Used
- **Proven Effectiveness of Cosine Similarity:** Cosine similarity has been widely adopted for comparing embeddings due to its simplicity and robustness in high-dimensional spaces.
- **Alignment With Task Requirements:** The normalized comparison offered by cosine similarity aligns well with the challenge’s goal of semantic matching.
- **Implementation Complexity:** Introducing and testing multiple similarity metrics would have diverted focus from refining the core solution.

## LLMs for Query Matching

Large Language Models (LLMs) could be used to match user queries with dataset values by leveraging their contextual understanding. This could involve constructing system and user prompts with context added from the dataset.

### Why Consider It?
- **Natural Interaction:** LLMs can handle natural language queries with minimal preprocessing, offering a user-friendly experience.
- **Contextual Understanding:** By embedding parts of the table into the prompt, the LLM can process complex queries without requiring structured embeddings.
- **Ease of Prototyping:** A prompt-based approach requires minimal code changes and relies heavily on the LLM's pretrained capabilities.

### Why It Wasn't Used
- **Cost and Latency:** Querying LLMs for every user query could be computationally expensive and slow, especially for large datasets.
- **Lack of Determinism:** LLM responses might vary between runs, making debugging and consistency a challenge.
- **Limited Dataset Size:** Embedding large amounts of context from the dataset into the prompt may exceed token limits.

## Fine-Tuning

Fine-tuning a model on the specific dataset could have optimized the embeddings for the task at hand.

### Why Consider It?
- **Task-Specific Adaptation:** Fine-tuning allows the model to better understand the nuances of the dataset, improving similarity calculations.
- **Improved Accuracy:** Customizing the model for the dataset could yield embeddings that align more closely with the desired results.

### Why It Wasn't Used
- **Resource Constraints:** Fine-tuning requires significant computational power and labeled data, neither of which were available for this challenge.
- **Complexity vs. Benefit:** Given the simplicity of the queries and the dataset, fine-tuning might not have provided a significant advantage over using a pretrained model.

# Hiccups in Solution

During the implementation, one issue stood out: 

- **Minimum Value Misalignment:** Sometimes the minimum value returned is **1** instead of **0**, even when **0** exists in the data. For example, if the user query is _"What is the minimum number of days xyz?"_, and **0** is the correct answer but **1** is also part of the data, the system returns **1**.

### Potential Solution
A potential solution is to add a **post-processing step** to validate numerical results. Specifically, for queries involving "minimum," check the dataset for the actual minimum value in the relevant column after the similarity scores are calculated. If **0** exists and aligns semantically, prioritize it over other close matches. 

This adjustment ensures correctness without overcomplicating the implementation and keeps the solution straightforward yet effective.

## Final Remarks

This challenge offered a unique opportunity to showcase a blend of technical expertise, thoughtful design, and practical problem-solving. The solution reflects a deliberate balance between simplicity and functionality, demonstrating the ability to deliver results while maintaining clarity and scalability. By leveraging proven techniques like cosine similarity and versatile tools like DistilBERT, the implementation provides a robust foundation for query matching tasks across varied data types.

At its heart, the project exemplifies the importance of understanding both the data and the user’s needs. Every decision—from preprocessing steps to embedding strategy—was guided by the goal of creating a system that is intuitive, efficient, and extensible. While there is always room for innovation and exploration, such as with LLMs or fine-tuning approaches, this solution ensures that the core functionality is solid and dependable.

Thank you for the opportunity to take on this challenge. It was both engaging and rewarding, and I hope the solution reflects my enthusiasm for the task and my commitment to engineering excellence. I look forward to discussing this further and hearing your feedback!

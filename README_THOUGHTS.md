### Note: Understanding the Two README Files

This project contains two README files to provide clear and detailed context:

1. **`README_RUN_CODE.md`**: This file contains all the instructions for setting up, running the code, and interpreting the results. If you’re looking to execute the code and analyze its outputs, this is the file you should start with.

2. **`README_THOUGHTS.md`**: This file explains my thought process, design decisions, challenges faced, and alternative approaches considered while solving the coding challenge. It offers deeper insight into the logic and reasoning behind the implementation. 

Feel free to refer to each README based on your purpose: whether you want to run the code or understand the rationale behind it!

## README: Thoughts and Approach

This document provides a high-level overview of my thought process, design choices, and technical decisions for the coding challenge. It explains the rationale behind the solution and highlights key considerations made during implementation.

## High-Level Thinking: Interpreting the Question

In this section, I will discuss:
- My interpretation of the challenge requirements.
- The core problem statement and what I aimed to achieve.
- Initial thoughts and potential approaches I considered.

## My Approach: Balancing Engineering and Business Needs

In this section, I will explain:
- How I approached the solution as both an engineer and a business-minded thinker.
- The importance of simplicity for users and maintainability for engineers.
- Why I chose to use a single embedding model for all data types (text, numerical, and date) to avoid over-engineering.

## Software Engineering Practices

This section will cover:
- The structure of the codebase and the use of `__init__.py` files for organization.
- The importance of writing tests to ensure code correctness and reliability.
- The use of logging for real-time insights during execution.
- The role of the `utils` folder in modularizing reusable components.

## Diving Deeper Into the Solution

### Data Processing
- Why I focused on data quality and preprocessing as the foundation of the solution.
- The rationale behind processing only unique values in a column for efficiency.
- Detailed explanation of how numerical and date columns were preprocessed.

### Choice of Embedding Model
- Other embedding models I considered and why I ultimately chose DistilBERT.
- The trade-offs between model complexity, accuracy, and execution speed.

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

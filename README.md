# KB Evaluation Tools

## Introduction

This repo contains useful tools to evaluate your Knowledge Base (KB), used within LLM-powered applications.

## Local Installation and Running

To install the toolset, you need to have the following dependencies installed:

* Python 3.11 or higher
* Poetry

1. Set up the environment variables. Navigate to `infra/` and fill in all the required values in `tools.env`
2. Prepare environment ```poetry install```

The only available tool for now is the KB evaluation tool. It takes in a Q&A dataset of questions and replies, and checks whether your RAG-powering KB can re-create the answer. Run this with:

```poetry run tools evaluation kb-evaluation [OPTIONS] QA_FILEPATH SAVE_TO_FILEPATH```

Where `QA_FILEPATH` is a JSON formatted file with the format:

```
{
  "name": "real-cases-qa-cases",
  "description": "Q&A from customer interactions",
  "metadata": {},
  "items": [
    {
      "input": "...",
      "expected_output": "...",
      "ticket_id": ...,
    },
    ...
  ]
}
```

The only currently supported KB is OpenSearch. We use AWS-hosted Foundational Models to perform this analysis, so do create an account with appropriate permissions to access these models before using these tools.

## KB eval tool - Methodology

The `evaluate_w_question` function is the core of the KB evaluation process. It follows these steps:

1. **Retrieve Relevant Documents**: 
   - Uses a vector store retriever to fetch documents from the KB that are relevant to the provided question.
   - Optionally merges results from multiple retrievers if reranking is enabled.

2. **Re-rank Documents**:
   - Re-ranks the retrieved documents based on their relevance to the agent's answer using either embedding-based or cross-encoder-based methods.

3. **Iterative Evaluation**:
   - Iteratively processes batches of documents using a language model to extract mappings between the question, answer, and document excerpts.
   - Tracks the completeness score using Jaccard similarity between the agent's answer and the extracted excerpts.

4. **Early Stopping**:
   - Stops the evaluation if the completeness score exceeds a predefined threshold or if no significant improvement is observed over a specified number of iterations.

5. **Output**:
   - Returns a dictionary containing the mappings of answer excerpts to document excerpts, along with their similarity scores.

This methodology ensures a systematic and thorough evaluation of the KB's ability to re-construct answers in the provided Q&A pairs.

---

In the future, see all available commands by typing `poetry run tools --help`

# KB Evaluation Tools

## Introduction

This repo contains useful tools to evaluate your knowledge base, used within LLM-powered applications.

## Local Installation and Running

To install the toolset, you need to have the following dependencies installed:

* Python 3.11 or higher
* Poetry

1. Set up the environment variables. Navigate to `infra/` and fill in all the required values in `tools.env`
2. Prepare environment ```poetry install```

The only available tool for now is the knowledge base evaluation tool. It takes in a Q&A dataset of questions and replies, and checks whether your RAG-powering Knowledge Base (KB) can re-create the answer. Run this with:

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

In the future, see all available commands by typing `poetry run tools --help`

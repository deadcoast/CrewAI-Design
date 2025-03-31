---
title: "CSV RAG Search - CrewAI"
source: "https://docs.crewai.com/tools/csvsearchtool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `CSVSearchTool` is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within a CSV file's content."
tags:
  - "clippings"
---
Tools

# CSV RAG Search

The `CSVSearchTool` is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within a CSV file’s content.

# 

[​

](https://docs.crewai.com/tools/#csvsearchtool)

`CSVSearchTool`

**Experimental**: We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## 

[​

](https://docs.crewai.com/tools/#description)

Description

This tool is used to perform a RAG (Retrieval-Augmented Generation) search within a CSV file’s content. It allows users to semantically search for queries in the content of a specified CSV file. This feature is particularly useful for extracting information from large CSV datasets where traditional search methods might be inefficient. All tools with “Search” in their name, including CSVSearchTool, are RAG tools designed for searching different sources of data.

## 

[​

](https://docs.crewai.com/tools/#installation)

Installation

Install the crewai\_tools package

Copy

```shell
pip install 'crewai[tools]'
```

## 

[​

](https://docs.crewai.com/tools/#example)

Example

Code

Copy

```python
from crewai_tools import CSVSearchTool

# Initialize the tool with a specific CSV file. 
# This setup allows the agent to only search the given CSV file.
tool = CSVSearchTool(csv='path/to/your/csvfile.csv')

# OR

# Initialize the tool without a specific CSV file. 
# Agent will need to provide the CSV path at runtime.
tool = CSVSearchTool()
```

## 

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

The following parameters can be used to customize the `CSVSearchTool`’s behavior:

| Argument | Type | Description |
| --- | --- | --- |
| **csv** | `string` | *Optional*. The path to the CSV file you want to search. This is a mandatory argument if the tool was initialized without a specific CSV file; otherwise, it is optional. |

## 

[​

](https://docs.crewai.com/tools/#custom-model-and-embeddings)

Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

Code

Copy

```python
tool = CSVSearchTool(
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
```

Was this page helpful?
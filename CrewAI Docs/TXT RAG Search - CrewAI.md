---
title: "TXT RAG Search - CrewAI"
source: "https://docs.crewai.com/tools/txtsearchtool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `TXTSearchTool` is designed to perform a RAG (Retrieval-Augmented Generation) search within the content of a text file."
tags:
  - "clippings"
---

Tools

# TXT RAG Search

The `TXTSearchTool` is designed to perform a RAG (Retrieval-Augmented Generation) search within the content of a text file.

#

[​

](https://docs.crewai.com/tools/#txtsearchtool)

`TXTSearchTool`

We are still working on improving tools, so there might be unexpected behavior or changes in the future.

##

[​

](https://docs.crewai.com/tools/#description)

Description

This tool is used to perform a RAG (Retrieval-Augmented Generation) search within the content of a text file. It allows for semantic searching of a query within a specified text file’s content, making it an invaluable resource for quickly extracting information or finding specific sections of text based on the query provided.

##

[​

](https://docs.crewai.com/tools/#installation)

Installation

To use the `TXTSearchTool`, you first need to install the `crewai_tools` package. This can be done using pip, a package manager for Python. Open your terminal or command prompt and enter the following command:

Copy

```shell
pip install 'crewai[tools]'
```

This command will download and install the TXTSearchTool along with any necessary dependencies.

##

[​

](https://docs.crewai.com/tools/#example)

Example

The following example demonstrates how to use the TXTSearchTool to search within a text file. This example shows both the initialization of the tool with a specific text file and the subsequent search within that file’s content.

Code

Copy

```python
from crewai_tools import TXTSearchTool

# Initialize the tool to search within any text file's content
# the agent learns about during its execution
tool = TXTSearchTool()

# OR

# Initialize the tool with a specific text file,
# so the agent can search within the given text file's content
tool = TXTSearchTool(txt='path/to/text/file.txt')
```

##

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

- `txt` (str): **Optional**. The path to the text file you want to search. This argument is only required if the tool was not initialized with a specific text file; otherwise, the search will be conducted within the initially provided text file.

##

[​

](https://docs.crewai.com/tools/#custom-model-and-embeddings)

Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

Code

Copy

```python
tool = TXTSearchTool(
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

Was this page helpf

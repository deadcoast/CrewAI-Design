---
title: "PDF RAG Search - CrewAI"
source: "https://docs.crewai.com/tools/pdfsearchtool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `PDFSearchTool` is designed to search PDF files and return the most relevant results."
tags:
  - "clippings"
---
Tools

# PDF RAG Search

The `PDFSearchTool` is designed to search PDF files and return the most relevant results.

# 

[​

](https://docs.crewai.com/tools/#pdfsearchtool)

`PDFSearchTool`

We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## 

[​

](https://docs.crewai.com/tools/#description)

Description

The PDFSearchTool is a RAG tool designed for semantic searches within PDF content. It allows for inputting a search query and a PDF document, leveraging advanced search techniques to find relevant content efficiently. This capability makes it especially useful for extracting specific information from large PDF files quickly.

## 

[​

](https://docs.crewai.com/tools/#installation)

Installation

To get started with the PDFSearchTool, first, ensure the crewai\_tools package is installed with the following command:

Copy

```shell
pip install 'crewai[tools]'
```

## 

[​

](https://docs.crewai.com/tools/#example)

Example

Here’s how to use the PDFSearchTool to search within a PDF document:

Code

Copy

```python
from crewai_tools import PDFSearchTool

# Initialize the tool allowing for any PDF content search if the path is provided during execution
tool = PDFSearchTool()

# OR

# Initialize the tool with a specific PDF path for exclusive search within that document
tool = PDFSearchTool(pdf='path/to/your/document.pdf')
```

## 

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

- `pdf`: **Optional** The PDF path for the search. Can be provided at initialization or within the `run` method’s arguments. If provided at initialization, the tool confines its search to the specified document.

## 

[​

](https://docs.crewai.com/tools/#custom-model-and-embeddings)

Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

Code

Copy

```python
tool = PDFSearchTool(
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
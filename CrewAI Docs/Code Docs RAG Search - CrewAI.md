---
title: "Code Docs RAG Search - CrewAI"
source: "https://docs.crewai.com/tools/codedocssearchtool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `CodeDocsSearchTool` is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within code documentation."
tags:
  - "clippings"
---
Tools

# Code Docs RAG Search

The `CodeDocsSearchTool` is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within code documentation.

# 

[​

](https://docs.crewai.com/tools/#codedocssearchtool)

`CodeDocsSearchTool`

**Experimental**: We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## 

[​

](https://docs.crewai.com/tools/#description)

Description

The CodeDocsSearchTool is a powerful RAG (Retrieval-Augmented Generation) tool designed for semantic searches within code documentation. It enables users to efficiently find specific information or topics within code documentation. By providing a `docs_url` during initialization, the tool narrows down the search to that particular documentation site. Alternatively, without a specific `docs_url`, it searches across a wide array of code documentation known or discovered throughout its execution, making it versatile for various documentation search needs.

## 

[​

](https://docs.crewai.com/tools/#installation)

Installation

To start using the CodeDocsSearchTool, first, install the crewai\_tools package via pip:

Copy

```shell
pip install 'crewai[tools]'
```

## 

[​

](https://docs.crewai.com/tools/#example)

Example

Utilize the CodeDocsSearchTool as follows to conduct searches within code documentation:

Code

Copy

```python
from crewai_tools import CodeDocsSearchTool

# To search any code documentation content 
# if the URL is known or discovered during its execution:
tool = CodeDocsSearchTool()

# OR

# To specifically focus your search on a given documentation site 
# by providing its URL:
tool = CodeDocsSearchTool(docs_url='https://docs.example.com/reference')
```

Substitute ‘[https://docs.example.com/reference](https://docs.example.com/reference)’ with your target documentation URL and ‘How to use search tool’ with the search query relevant to your needs.

## 

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

The following parameters can be used to customize the `CodeDocsSearchTool`’s behavior:

| Argument | Type | Description |
| --- | --- | --- |
| **docs\_url** | `string` | *Optional*. Specifies the URL of the code documentation to be searched. |

## 

[​

](https://docs.crewai.com/tools/#custom-model-and-embeddings)

Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

Code

Copy

```python
tool = CodeDocsSearchTool(
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
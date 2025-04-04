---
title: "Github Search - CrewAI"
source: "https://docs.crewai.com/tools/githubsearchtool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `GithubSearchTool` is designed to search websites and convert them into clean markdown or structured data."
tags:
  - "clippings"
---

Tools

# Github Search

The `GithubSearchTool` is designed to search websites and convert them into clean markdown or structured data.

#

[​

](https://docs.crewai.com/tools/#githubsearchtool)

`GithubSearchTool`

We are still working on improving tools, so there might be unexpected behavior or changes in the future.

##

[​

](https://docs.crewai.com/tools/#description)

Description

The GithubSearchTool is a Retrieval-Augmented Generation (RAG) tool specifically designed for conducting semantic searches within GitHub repositories. Utilizing advanced semantic search capabilities, it sifts through code, pull requests, issues, and repositories, making it an essential tool for developers, researchers, or anyone in need of precise information from GitHub.

##

[​

](https://docs.crewai.com/tools/#installation)

Installation

To use the GithubSearchTool, first ensure the crewai_tools package is installed in your Python environment:

Copy

```shell
pip install 'crewai[tools]'
```

This command installs the necessary package to run the GithubSearchTool along with any other tools included in the crewai_tools package.

##

[​

](https://docs.crewai.com/tools/#example)

Example

Here’s how you can use the GithubSearchTool to perform semantic searches within a GitHub repository:

Code

Copy

```python
from crewai_tools import GithubSearchTool

# Initialize the tool for semantic searches within a specific GitHub repository
tool = GithubSearchTool(
	github_repo='https://github.com/example/repo',
	gh_token='your_github_personal_access_token',
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)

# OR

# Initialize the tool for semantic searches within a specific GitHub repository, so the agent can search any repository if it learns about during its execution
tool = GithubSearchTool(
	gh_token='your_github_personal_access_token',
	content_types=['code', 'issue'] # Options: code, repo, pr, issue
)
```

##

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

- `github_repo` : The URL of the GitHub repository where the search will be conducted. This is a mandatory field and specifies the target repository for your search.
- `gh_token` : Your GitHub Personal Access Token (PAT) required for authentication. You can create one in your GitHub account settings under Developer Settings > Personal Access Tokens.
- `content_types` : Specifies the types of content to include in your search. You must provide a list of content types from the following options: `code` for searching within the code, `repo` for searching within the repository’s general information, `pr` for searching within pull requests, and `issue` for searching within issues. This field is mandatory and allows tailoring the search to specific content types within the GitHub repository.

##

[​

](https://docs.crewai.com/tools/#custom-model-and-embeddings)

Custom model and embeddings

By default, the tool uses OpenAI for both embeddings and summarization. To customize the model, you can use a config dictionary as follows:

Code

Copy

```python
tool = GithubSearchTool(
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

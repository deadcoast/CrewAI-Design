---
title: "Firecrawl Search - CrewAI"
source: "https://docs.crewai.com/tools/firecrawlsearchtool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `FirecrawlSearchTool` is designed to search websites and convert them into clean markdown or structured data."
tags:
  - "clippings"
---

Tools

# Firecrawl Search

The `FirecrawlSearchTool` is designed to search websites and convert them into clean markdown or structured data.

#

[​

](https://docs.crewai.com/tools/#firecrawlsearchtool)

`FirecrawlSearchTool`

##

[​

](https://docs.crewai.com/tools/#description)

Description

[Firecrawl](https://firecrawl.dev/) is a platform for crawling and convert any website into clean markdown or structured data.

##

[​

](https://docs.crewai.com/tools/#installation)

Installation

- Get an API key from [firecrawl.dev](https://firecrawl.dev/) and set it in environment variables (`FIRECRAWL_API_KEY`).
- Install the [Firecrawl SDK](https://github.com/mendableai/firecrawl) along with `crewai[tools]` package:

Copy

```shell
pip install firecrawl-py 'crewai[tools]'
```

##

[​

](https://docs.crewai.com/tools/#example)

Example

Utilize the FirecrawlSearchTool as follows to allow your agent to load websites:

Code

Copy

```python
from crewai_tools import FirecrawlSearchTool

tool = FirecrawlSearchTool(query='what is firecrawl?')
```

##

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

- `api_key`: Optional. Specifies Firecrawl API key. Defaults is the `FIRECRAWL_API_KEY` environment variable.
- `query`: The search query string to be used for searching.
- `page_options`: Optional. Options for result formatting.
- `onlyMainContent`: Optional. Only return the main content of the page excluding headers, navs, footers, etc.
- `includeHtml`: Optional. Include the raw HTML content of the page. Will output a html key in the response.
- `fetchPageContent`: Optional. Fetch the full content of the page.
- `search_options`: Optional. Options for controlling the crawling behavior.
- `limit`: Optional. Maximum number of pages to crawl.

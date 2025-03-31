---
title: "Firecrawl Scrape Website - CrewAI"
source: "https://docs.crewai.com/tools/firecrawlscrapewebsitetool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `FirecrawlScrapeWebsiteTool` is designed to scrape websites and convert them into clean markdown or structured data."
tags:
  - "clippings"
---

Tools

# Firecrawl Scrape Website

The `FirecrawlScrapeWebsiteTool` is designed to scrape websites and convert them into clean markdown or structured data.

#

[​

](https://docs.crewai.com/tools/#firecrawlscrapewebsitetool)

`FirecrawlScrapeWebsiteTool`

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

Utilize the FirecrawlScrapeWebsiteTool as follows to allow your agent to load websites:

Code

Copy

```python
from crewai_tools import FirecrawlScrapeWebsiteTool

tool = FirecrawlScrapeWebsiteTool(url='firecrawl.dev')
```

##

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

- `api_key`: Optional. Specifies Firecrawl API key. Defaults is the `FIRECRAWL_API_KEY` environment variable.
- `url`: The URL to scrape.
- `page_options`: Optional.
- `onlyMainContent`: Optional. Only return the main content of the page excluding headers, navs, footers, etc.
- `includeHtml`: Optional. Include the raw HTML content of the page. Will output a html key in the response.
- `extractor_options`: Optional. Options for LLM-based extraction of structured information from the page content
- `mode`: The extraction mode to use, currently supports ‘llm-extraction’
- `extractionPrompt`: Optional. A prompt describing what information to extract from the page
- `extractionSchema`: Optional. The schema for the data to be extracted
- `timeout`: Optional. Timeout in milliseconds for the request

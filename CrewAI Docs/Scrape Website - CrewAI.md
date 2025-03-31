---
title: "Scrape Website - CrewAI"
source: "https://docs.crewai.com/tools/scrapewebsitetool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `ScrapeWebsiteTool` is designed to extract and read the content of a specified website."
tags:
  - "clippings"
---
Tools

# Scrape Website

The `ScrapeWebsiteTool` is designed to extract and read the content of a specified website.

# 

[​

](https://docs.crewai.com/tools/#scrapewebsitetool)

`ScrapeWebsiteTool`

We are still working on improving tools, so there might be unexpected behavior or changes in the future.

## 

[​

](https://docs.crewai.com/tools/#description)

Description

A tool designed to extract and read the content of a specified website. It is capable of handling various types of web pages by making HTTP requests and parsing the received HTML content. This tool can be particularly useful for web scraping tasks, data collection, or extracting specific information from websites.

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

Copy

```python
from crewai_tools import ScrapeWebsiteTool

# To enable scrapping any website it finds during it's execution
tool = ScrapeWebsiteTool()

# Initialize the tool with the website URL, 
# so the agent can only scrap the content of the specified website
tool = ScrapeWebsiteTool(website_url='https://www.example.com')

# Extract the text from the site
text = tool.run()
print(text)
```

## 

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

| Argument | Type | Description |
| --- | --- | --- |
| **website\_url** | `string` | **Mandatory** website URL to read the file. This is the primary input for the tool, specifying which website’s content should be scraped and read. |

Was this page helpf
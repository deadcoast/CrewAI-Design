---
title: "Selenium Scraper - CrewAI"
source: "https://docs.crewai.com/tools/seleniumscrapingtool"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "The `SeleniumScrapingTool` is designed to extract and read the content of a specified website using Selenium."
tags:
  - "clippings"
---

Tools

# Selenium Scraper

The `SeleniumScrapingTool` is designed to extract and read the content of a specified website using Selenium.

#

[​

](https://docs.crewai.com/tools/#seleniumscrapingtool)

`SeleniumScrapingTool`

This tool is currently in development. As we refine its capabilities, users may encounter unexpected behavior. Your feedback is invaluable to us for making improvements.

##

[​

](https://docs.crewai.com/tools/#description)

Description

The SeleniumScrapingTool is crafted for high-efficiency web scraping tasks. It allows for precise extraction of content from web pages by using CSS selectors to target specific elements. Its design caters to a wide range of scraping needs, offering flexibility to work with any provided website URL.

##

[​

](https://docs.crewai.com/tools/#installation)

Installation

To get started with the SeleniumScrapingTool, install the crewai_tools package using pip:

Copy

```shell
pip install 'crewai[tools]'
```

##

[​

](https://docs.crewai.com/tools/#usage-examples)

Usage Examples

Below are some scenarios where the SeleniumScrapingTool can be utilized:

Code

Copy

```python
from crewai_tools import SeleniumScrapingTool

# Example 1:
# Initialize the tool without any parameters to scrape
# the current page it navigates to
tool = SeleniumScrapingTool()

# Example 2:
# Scrape the entire webpage of a given URL
tool = SeleniumScrapingTool(website_url='https://example.com')

# Example 3:
# Target and scrape a specific CSS element from a webpage
tool = SeleniumScrapingTool(
    website_url='https://example.com',
    css_element='.main-content'
)

# Example 4:
# Perform scraping with additional parameters for a customized experience
tool = SeleniumScrapingTool(
    website_url='https://example.com',
    css_element='.main-content',
    cookie={'name': 'user', 'value': 'John Doe'},
    wait_time=10
)
```

##

[​

](https://docs.crewai.com/tools/#arguments)

Arguments

The following parameters can be used to customize the SeleniumScrapingTool’s scraping process:

| Argument        | Type     | Description                                                                                                                                   |
| --------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **website_url** | `string` | **Mandatory**. Specifies the URL of the website from which content is to be scraped.                                                          |
| **css_element** | `string` | **Mandatory**. The CSS selector for a specific element to target on the website, enabling focused scraping of a particular part of a webpage. |
| **cookie**      | `object` | **Optional**. A dictionary containing cookie information, useful for simulating a logged-in session to access restricted content.             |
| **wait_time**   | `int`    | **Optional**. Specifies the delay (in seconds) before scraping, allowing the website and any dynamic content to fully load.                   |

Since the `SeleniumScrapingTool` is under active development, the parameters and functionality may evolve over time. Users are encouraged to keep the tool updated and report any issues or suggestions for enhancements.

Was this page helpful?

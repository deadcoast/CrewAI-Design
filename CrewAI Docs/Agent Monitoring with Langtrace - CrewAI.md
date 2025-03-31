---
title: "Agent Monitoring with Langtrace - CrewAI"
source: "https://docs.crewai.com/how-to/langtrace-observability"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "How to monitor cost, latency, and performance of CrewAI Agents using Langtrace, an external observability tool."
tags:
  - "clippings"
---

How to Guides

# Agent Monitoring with Langtrace

How to monitor cost, latency, and performance of CrewAI Agents using Langtrace, an external observability tool.

#

[​

](https://docs.crewai.com/how-to/#langtrace-overview)

Langtrace Overview

Langtrace is an open-source, external tool that helps you set up observability and evaluations for Large Language Models (LLMs), LLM frameworks, and Vector Databases. While not built directly into CrewAI, Langtrace can be used alongside CrewAI to gain deep visibility into the cost, latency, and performance of your CrewAI Agents. This integration allows you to log hyperparameters, monitor performance regressions, and establish a process for continuous improvement of your Agents.

![Overview of a select series of agent session runs](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/langtrace1.png) ![Overview of agent traces](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/langtrace2.png) ![Overview of llm traces in details](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/langtrace3.png)

##

[​

](https://docs.crewai.com/how-to/#setup-instructions)

Setup Instructions

1

Sign up for Langtrace

Sign up by visiting [https://langtrace.ai/signup](https://langtrace.ai/signup).

2

Create a project

Set the project type to `CrewAI` and generate an API key.

3

Install Langtrace in your CrewAI project

Use the following command:

Copy

```bash
pip install langtrace-python-sdk
```

4

Import Langtrace

Import and initialize Langtrace at the beginning of your script, before any CrewAI imports:

Copy

```python
from langtrace_python_sdk import langtrace
langtrace.init(api_key='<LANGTRACE_API_KEY>')

# Now import CrewAI modules
from crewai import Agent, Task, Crew
```

###

[​

](https://docs.crewai.com/how-to/#features-and-their-application-to-crewai)

Features and Their Application to CrewAI

1. **LLM Token and Cost Tracking**

- Monitor the token usage and associated costs for each CrewAI agent interaction.

2. **Trace Graph for Execution Steps**

- Visualize the execution flow of your CrewAI tasks, including latency and logs.
- Useful for identifying bottlenecks in your agent workflows.

3. **Dataset Curation with Manual Annotation**

- Create datasets from your CrewAI task outputs for future training or evaluation.

4. **Prompt Versioning and Management**

- Keep track of different versions of prompts used in your CrewAI agents.
- Useful for A/B testing and optimizing agent performance.

5. **Prompt Playground with Model Comparisons**

- Test and compare different prompts and models for your CrewAI agents before deployment.

6. **Testing and Evaluations**

- Set up automated tests for your CrewAI agents and tasks.

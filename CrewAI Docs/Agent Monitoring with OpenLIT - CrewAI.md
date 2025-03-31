---
title: "Agent Monitoring with OpenLIT - CrewAI"
source: "https://docs.crewai.com/how-to/openlit-observability"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "Quickly start monitoring your Agents in just a single line of code with OpenTelemetry."
tags:
  - "clippings"
---
How to Guides

# Agent Monitoring with OpenLIT

Quickly start monitoring your Agents in just a single line of code with OpenTelemetry.

# 

[​

](https://docs.crewai.com/how-to/#openlit-overview)

OpenLIT Overview

[OpenLIT](https://github.com/openlit/openlit?src=crewai-docs) is an open-source tool that makes it simple to monitor the performance of AI agents, LLMs, VectorDBs, and GPUs with just **one** line of code.

It provides OpenTelemetry-native tracing and metrics to track important parameters like cost, latency, interactions and task sequences. This setup enables you to track hyperparameters and monitor for performance issues, helping you find ways to enhance and fine-tune your agents over time.

![Overview Agent usage including cost and tokens](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/openlit1.png)![Overview of agent otel traces and metrics](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/openlit2.png)![Overview of agent traces in details](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/openlit3.png)

OpenLIT Dashboard

### 

[​

](https://docs.crewai.com/how-to/#features)

Features

- **Analytics Dashboard**: Monitor your Agents health and performance with detailed dashboards that track metrics, costs, and user interactions.
- **OpenTelemetry-native Observability SDK**: Vendor-neutral SDKs to send traces and metrics to your existing observability tools like Grafana, DataDog and more.
- **Cost Tracking for Custom and Fine-Tuned Models**: Tailor cost estimations for specific models using custom pricing files for precise budgeting.
- **Exceptions Monitoring Dashboard**: Quickly spot and resolve issues by tracking common exceptions and errors with a monitoring dashboard.
- **Compliance and Security**: Detect potential threats such as profanity and PII leaks.
- **Prompt Injection Detection**: Identify potential code injection and secret leaks.
- **API Keys and Secrets Management**: Securely handle your LLM API keys and secrets centrally, avoiding insecure practices.
- **Prompt Management**: Manage and version Agent prompts using PromptHub for consistent and easy access across Agents.
- **Model Playground** Test and compare different models for your CrewAI agents before deployment.

## 

[​

](https://docs.crewai.com/how-to/#setup-instructions)

Setup Instructions

1

Deploy OpenLIT

1

Git Clone OpenLIT Repository

Copy

```shell
git clone git@github.com:openlit/openlit.git
```

2

Start Docker Compose

From the root directory of the [OpenLIT Repo](https://github.com/openlit/openlit), Run the below command:

Copy

```shell
docker compose up -d
```

2

Install OpenLIT SDK

Copy

```shell
pip install openlit
```

3

Initialize OpenLIT in Your Application

Add the following two lines to your application code:

- 
- 

Copy

```python
import openlit
openlit.init(otlp_endpoint="http://127.0.0.1:4318")
```

Example Usage for monitoring a CrewAI Agent:

Copy

```python
from crewai import Agent, Task, Crew, Process
import openlit

openlit.init(disable_metrics=True)
# Define your agents
researcher = Agent(
    role="Researcher",
    goal="Conduct thorough research and analysis on AI and AI agents",
    backstory="You're an expert researcher, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently researching for a new client.",
    allow_delegation=False,
    llm='command-r'
)

# Define your task
task = Task(
    description="Generate a list of 5 interesting ideas for an article, then write one captivating paragraph for each idea that showcases the potential of a full article on this topic. Return the list of ideas with their paragraphs and your notes.",
    expected_output="5 bullet points, each with a paragraph and accompanying notes.",
)

# Define the manager agent
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    allow_delegation=True,
    llm='command-r'
)

# Instantiate your crew with a custom manager
crew = Crew(
    agents=[researcher],
    tasks=[task],
    manager_agent=manager,
    process=Process.hierarchical,
)

# Start the crew's work
result = crew.kickoff()

print(result)
```

Refer to OpenLIT [Python SDK repository](https://github.com/openlit/openlit/tree/main/sdk/python) for more advanced configurations and use cases.

4

Visualize and Analyze

With the Agent Observability data now being collected and sent to OpenLIT, the next step is to visualize and analyze this data to get insights into your Agent’s performance, behavior, and identify areas of improvement.

Just head over to OpenLIT at `127.0.0.1:3000` on your browser to start exploring. You can login using the default credentials

- **Email**: `user@openlit.io`
- **Password**: `openlituser`

![Overview Agent usage including cost and tokens](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/openlit1.png)![Overview of agent otel traces and metrics](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/openlit2.png)

OpenLIT Dashboard

Was this page helpful?
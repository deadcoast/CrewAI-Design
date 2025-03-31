---
title: "Portkey Observability and Guardrails - CrewAI"
source: "https://docs.crewai.com/how-to/portkey-observability"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "How to use Portkey with CrewAI"
tags:
  - "clippings"
---
How to Guides

# Portkey Observability and Guardrails

How to use Portkey with CrewAI

![Portkey CrewAI Header Image](https://raw.githubusercontent.com/siddharthsambharia-portkey/Portkey-Product-Images/main/Portkey-CrewAI.png)

[Portkey](https://portkey.ai/?utm_source=crewai&utm_medium=crewai&utm_campaign=crewai) is a 2-line upgrade to make your CrewAI agents reliable, cost-efficient, and fast.

Portkey adds 4 core production capabilities to any CrewAI agent:

1. Routing to **200+ LLMs**
2. Making each LLM call more robust
3. Full-stack tracing & cost, performance analytics
4. Real-time guardrails to enforce behavior

## 

[​

](https://docs.crewai.com/how-to/#getting-started)

Getting Started

1

Install CrewAI and Portkey

Copy

```bash
pip install -qU crewai portkey-ai
```

2

Configure the LLM Client

To build CrewAI Agents with Portkey, you’ll need two keys:

- **Portkey API Key**: Sign up on the [Portkey app](https://app.portkey.ai/?utm_source=crewai&utm_medium=crewai&utm_campaign=crewai) and copy your API key
- **Virtual Key**: Virtual Keys securely manage your LLM API keys in one place. Store your LLM provider API keys securely in Portkey’s vault

Copy

```python
from crewai import LLM
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

gpt_llm = LLM(
    model="gpt-4",
    base_url=PORTKEY_GATEWAY_URL,
    api_key="dummy", # We are using Virtual key
    extra_headers=createHeaders(
        api_key="YOUR_PORTKEY_API_KEY",
        virtual_key="YOUR_VIRTUAL_KEY", # Enter your Virtual key from Portkey
    )
)
```

3

Create and Run Your First Agent

Copy

```python
from crewai import Agent, Task, Crew

# Define your agents with roles and goals
coder = Agent(
    role='Software developer',
    goal='Write clear, concise code on demand',
    backstory='An expert coder with a keen eye for software trends.',
    llm=gpt_llm
)

# Create tasks for your agents
task1 = Task(
    description="Define the HTML for making a simple website with heading- Hello World! Portkey is working!",
    expected_output="A clear and concise HTML code",
    agent=coder
)

# Instantiate your crew
crew = Crew(
    agents=[coder],
    tasks=[task1],
)

result = crew.kickoff()
print(result)
```

## 

[​

](https://docs.crewai.com/how-to/#key-features)

Key Features

| Feature | Description |
| --- | --- |
| 🌐 Multi-LLM Support | Access OpenAI, Anthropic, Gemini, Azure, and 250+ providers through a unified interface |
| 🛡️ Production Reliability | Implement retries, timeouts, load balancing, and fallbacks |
| 📊 Advanced Observability | Track 40+ metrics including costs, tokens, latency, and custom metadata |
| 🔍 Comprehensive Logging | Debug with detailed execution traces and function call logs |
| 🚧 Security Controls | Set budget limits and implement role-based access control |
| 🔄 Performance Analytics | Capture and analyze feedback for continuous improvement |
| 💾 Intelligent Caching | Reduce costs and latency with semantic or simple caching |

## 

[​

](https://docs.crewai.com/how-to/#production-features-with-portkey-configs)

Production Features with Portkey Configs

All features mentioned below are through Portkey’s Config system. Portkey’s Config system allows you to define routing strategies using simple JSON objects in your LLM API calls. You can create and manage Configs directly in your code or through the Portkey Dashboard. Each Config has a unique ID for easy reference.

![](https://raw.githubusercontent.com/Portkey-AI/docs-core/refs/heads/main/images/libraries/libraries-3.avif)

### 

[​

](https://docs.crewai.com/how-to/#1-use-250-llms)

1\. Use 250+ LLMs

Access various LLMs like Anthropic, Gemini, Mistral, Azure OpenAI, and more with minimal code changes. Switch between providers or use them together seamlessly. [Learn more about Universal API](https://portkey.ai/docs/product/ai-gateway/universal-api)

Easily switch between different LLM providers:

Copy

```python
# Anthropic Configuration
anthropic_llm = LLM(
    model="claude-3-5-sonnet-latest",
    base_url=PORTKEY_GATEWAY_URL,
    api_key="dummy",
    extra_headers=createHeaders(
        api_key="YOUR_PORTKEY_API_KEY",
        virtual_key="YOUR_ANTHROPIC_VIRTUAL_KEY", #You don't need provider when using Virtual keys
        trace_id="anthropic_agent"
    )
)

# Azure OpenAI Configuration
azure_llm = LLM(
    model="gpt-4",
    base_url=PORTKEY_GATEWAY_URL,
    api_key="dummy",
    extra_headers=createHeaders(
        api_key="YOUR_PORTKEY_API_KEY",
        virtual_key="YOUR_AZURE_VIRTUAL_KEY", #You don't need provider when using Virtual keys
        trace_id="azure_agent"
    )
)
```

### 

[​

](https://docs.crewai.com/how-to/#2-caching)

2\. Caching

Improve response times and reduce costs with two powerful caching modes:

- **Simple Cache**: Perfect for exact matches
- **Semantic Cache**: Matches responses for requests that are semantically similar [Learn more about Caching](https://portkey.ai/docs/product/ai-gateway/cache-simple-and-semantic)

Copy

```py
config = {
    "cache": {
        "mode": "semantic",  # or "simple" for exact matching
    }
}
```

### 

[​

](https://docs.crewai.com/how-to/#3-production-reliability)

3\. Production Reliability

Portkey provides comprehensive reliability features:

- **Automatic Retries**: Handle temporary failures gracefully
- **Request Timeouts**: Prevent hanging operations
- **Conditional Routing**: Route requests based on specific conditions
- **Fallbacks**: Set up automatic provider failovers
- **Load Balancing**: Distribute requests efficiently

[Learn more about Reliability Features](https://portkey.ai/docs/product/ai-gateway/)

### 

[​

](https://docs.crewai.com/how-to/#4-metrics)

4\. Metrics

Agent runs are complex. Portkey automatically logs **40+ comprehensive metrics** for your AI agents, including cost, tokens used, latency, etc. Whether you need a broad overview or granular insights into your agent runs, Portkey’s customizable filters provide the metrics you need.

- Cost per agent interaction
- Response times and latency
- Token usage and efficiency
- Success/failure rates
- Cache hit rates

![Portkey Dashboard](https://github.com/siddharthsambharia-portkey/Portkey-Product-Images/blob/main/Portkey-Dashboard.png?raw=true)

### 

[​

](https://docs.crewai.com/how-to/#5-detailed-logging)

5\. Detailed Logging

Logs are essential for understanding agent behavior, diagnosing issues, and improving performance. They provide a detailed record of agent activities and tool use, which is crucial for debugging and optimizing processes.

Access a dedicated section to view records of agent executions, including parameters, outcomes, function calls, and errors. Filter logs based on multiple parameters such as trace ID, model, tokens used, and metadata.

### 

[​

](https://docs.crewai.com/how-to/#6-enterprise-security-features)

6\. Enterprise Security Features

- Set budget limit and rate limts per Virtual Key (disposable API keys)
- Implement role-based access control
- Track system changes with audit logs
- Configure data retention policies

For detailed information on creating and managing Configs, visit the [Portkey documentation](https://docs.portkey.ai/product/ai-gateway/configs).

## 

[​

](https://docs.crewai.com/how-to/#resources)

Resources

- [📘 Portkey Documentation](https://docs.portkey.ai/)
- [📊 Portkey Dashboard](https://app.portkey.ai/?utm_source=crewai&utm_medium=crewai&utm_campaign=crewai)
- [🐦 Twitter](https://twitter.com/portkeyai)
- [💬 Discord Community](https://discord.gg/DD7vgKK299)
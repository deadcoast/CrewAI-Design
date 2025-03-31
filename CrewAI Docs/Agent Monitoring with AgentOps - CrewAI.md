---
title: "Agent Monitoring with AgentOps - CrewAI"
source: "https://docs.crewai.com/how-to/agentops-observability"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "Understanding and logging your agent performance with AgentOps."
tags:
  - "clippings"
---

How to Guides

# Agent Monitoring with AgentOps

Understanding and logging your agent performance with AgentOps.

#

[​

](https://docs.crewai.com/how-to/#introduction)

Introduction

Observability is a key aspect of developing and deploying conversational AI agents. It allows developers to understand how their agents are performing, how their agents are interacting with users, and how their agents use external tools and APIs. AgentOps is a product independent of CrewAI that provides a comprehensive observability solution for agents.

##

[​

](https://docs.crewai.com/how-to/#agentops)

AgentOps

[AgentOps](https://agentops.ai/?=crew) provides session replays, metrics, and monitoring for agents.

At a high level, AgentOps gives you the ability to monitor cost, token usage, latency, agent failures, session-wide statistics, and more. For more info, check out the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).

###

[​

](https://docs.crewai.com/how-to/#overview)

Overview

AgentOps provides monitoring for agents in development and production. It provides a dashboard for tracking agent performance, session replays, and custom reporting.

Additionally, AgentOps provides session drilldowns for viewing Crew agent interactions, LLM calls, and tool usage in real-time. This feature is useful for debugging and understanding how agents interact with users as well as other agents.

![Overview of a select series of agent session runs](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/agentops-overview.png) ![Overview of session drilldowns for examining agent runs](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/agentops-session.png) ![Viewing a step-by-step agent replay execution graph](https://mintlify.s3.us-west-1.amazonaws.com/crewai/images/agentops-replay.png)

###

[​

](https://docs.crewai.com/how-to/#features)

Features

- **LLM Cost Management and Tracking**: Track spend with foundation model providers.
- **Replay Analytics**: Watch step-by-step agent execution graphs.
- **Recursive Thought Detection**: Identify when agents fall into infinite loops.
- **Custom Reporting**: Create custom analytics on agent performance.
- **Analytics Dashboard**: Monitor high-level statistics about agents in development and production.
- **Public Model Testing**: Test your agents against benchmarks and leaderboards.
- **Custom Tests**: Run your agents against domain-specific tests.
- **Time Travel Debugging**: Restart your sessions from checkpoints.
- **Compliance and Security**: Create audit logs and detect potential threats such as profanity and PII leaks.
- **Prompt Injection Detection**: Identify potential code injection and secret leaks.

###

[​

](https://docs.crewai.com/how-to/#using-agentops)

Using AgentOps

1

Create an API Key

Create a user API key here: [Create API Key](https://app.agentops.ai/account)

2

Configure Your Environment

Add your API key to your environment variables:

Copy

```bash
AGENTOPS_API_KEY=<YOUR_AGENTOPS_API_KEY>
```

3

Install AgentOps

Install AgentOps with:

Copy

```bash
pip install 'crewai[agentops]'
```

or

Copy

```bash
pip install agentops
```

4

Initialize AgentOps

Before using `Crew` in your script, include these lines:

Copy

```python
import agentops
agentops.init()
```

This will initiate an AgentOps session as well as automatically track Crew agents. For further info on how to outfit more complex agentic systems, check out the [AgentOps documentation](https://docs.agentops.ai/) or join the [Discord](https://discord.gg/j4f3KbeH).

###

[​

](https://docs.crewai.com/how-to/#crew-agentops-examples)

Crew + AgentOps Examples

[

## Job Posting

Example of a Crew agent that generates job posts.

](https://github.com/joaomdmoura/crewAI-examples/tree/main/job-posting)[

## Markdown Validator

Example of a Crew agent that validates Markdown files.

](https://github.com/joaomdmoura/crewAI-examples/tree/main/markdown_validator)[

## Instagram Post

Example of a Crew agent that generates Instagram posts.

](https://github.com/joaomdmoura/crewAI-examples/tree/main/instagram_post)

###

[​

](https://docs.crewai.com/how-to/#further-information)

Further Information

To get started, create an [AgentOps account](https://agentops.ai/?=crew).

For feature requests or bug reports, please reach out to the AgentOps team on the [AgentOps Repo](https://github.com/AgentOps-AI/agentops).

####

[​

](https://docs.crewai.com/how-to/#extra-links)

Extra links

[🐦 Twitter](https://twitter.com/agentopsai/)   •   [📢 Discord](https://discord.gg/JHPt4C7r)   •   [🖇️ AgentOps Dashboard](https://app.agentops.ai/?=crew)   •   [📙 Documentation](https://docs.agentops.ai/introduction)

---
title: "Force Tool Output as Result - CrewAI"
source: "https://docs.crewai.com/how-to/force-tool-output-as-result"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "Learn how to force tool output as the result in an Agent's task in CrewAI."
tags:
  - "clippings"
---
How to Guides

# Force Tool Output as Result

Learn how to force tool output as the result in an Agent’s task in CrewAI.

## 

[​

](https://docs.crewai.com/how-to/#introduction)

Introduction

In CrewAI, you can force the output of a tool as the result of an agent’s task. This feature is useful when you want to ensure that the tool output is captured and returned as the task result, avoiding any agent modification during the task execution.

## 

[​

](https://docs.crewai.com/how-to/#forcing-tool-output-as-result)

Forcing Tool Output as Result

To force the tool output as the result of an agent’s task, you need to set the `result_as_answer` parameter to `True` when adding a tool to the agent. This parameter ensures that the tool output is captured and returned as the task result, without any modifications by the agent.

Here’s an example of how to force the tool output as the result of an agent’s task:

Code

Copy

```python
from crewai.agent import Agent
from my_tool import MyCustomTool

# Create a coding agent with the custom tool
coding_agent = Agent(
        role="Data Scientist",
        goal="Produce amazing reports on AI",
        backstory="You work with data and AI",
        tools=[MyCustomTool(result_as_answer=True)],
    )

# Assuming the tool's execution and result population occurs within the system
task_result = coding_agent.execute_task(task)
```

## 

[​

](https://docs.crewai.com/how-to/#workflow-in-action)

Workflow in Action

1

Task Execution

The agent executes the task using the tool provided.

2

Tool Output

The tool generates the output, which is captured as the task result.

3

Agent Interaction

The agent may reflect and take learnings from the tool but the output is not modified.

4

Result Return

The tool output is returned as the task result without any modifications.
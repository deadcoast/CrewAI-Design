---
title: "Replay Tasks from Latest Crew Kickoff - CrewAI"
source: "https://docs.crewai.com/how-to/replay-tasks-from-latest-crew-kickoff"
author:
  - "[[CrewAI]]"
published:
created: 2025-01-17
description: "Replay tasks from the latest crew.kickoff(...)"
tags:
  - "clippings"
---
How to Guides

# Replay Tasks from Latest Crew Kickoff

Replay tasks from the latest crew.kickoff(…)

## 

[​

](https://docs.crewai.com/how-to/#introduction)

Introduction

CrewAI provides the ability to replay from a task specified from the latest crew kickoff. This feature is particularly useful when you’ve finished a kickoff and may want to retry certain tasks or don’t need to refetch data over and your agents already have the context saved from the kickoff execution so you just need to replay the tasks you want to.

You must run `crew.kickoff()` before you can replay a task. Currently, only the latest kickoff is supported, so if you use `kickoff_for_each`, it will only allow you to replay from the most recent crew run.

Here’s an example of how to replay from a task:

### 

[​

](https://docs.crewai.com/how-to/#replaying-from-specific-task-using-the-cli)

Replaying from Specific Task Using the CLI

To use the replay feature, follow these steps:

1

Open your terminal or command prompt.

2

Navigate to the directory where your CrewAI project is located.

3

Run the following commands:

To view the latest kickoff task\_ids use:

Copy

```shell
crewai log-tasks-outputs
```

Once you have your `task_id` to replay, use:

Copy

```shell
crewai replay -t <task_id>
```

Ensure `crewai` is installed and configured correctly in your development environment.

### 

[​

](https://docs.crewai.com/how-to/#replaying-from-a-task-programmatically)

Replaying from a Task Programmatically

To replay from a task programmatically, use the following steps:

1

Specify the \`task\_id\` and input parameters for the replay process.

Specify the `task_id` and input parameters for the replay process.

2

Execute the replay command within a try-except block to handle potential errors.

Execute the replay command within a try-except block to handle potential errors.

Copy

```python
  def replay():
  """
  Replay the crew execution from a specific task.
  """
  task_id = '<task_id>'
  inputs = {"topic": "CrewAI Training"}  # This is optional; you can pass in the inputs you want to replay; otherwise, it uses the previous kickoff's inputs.
  try:
      YourCrewName_Crew().crew().replay(task_id=task_id, inputs=inputs)

  except subprocess.CalledProcessError as e:
      raise Exception(f"An error occurred while replaying the crew: {e}")

  except Exception as e:
      raise Exception(f"An unexpected error occurred: {e}")
```

## 

[​

](https://docs.crewai.com/how-to/#conclusion)

Conclusion

With the above enhancements and detailed functionality, replaying specific tasks in CrewAI has been made more efficient and robust. Ensure you follow the commands and steps precisely to make the most of these features.
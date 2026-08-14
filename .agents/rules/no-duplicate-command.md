---
trigger: always_on
---

# Prevent duplicate terminal commands

- Never execute an identical terminal command more than once in the same task unless the user explicitly approves the rerun.
- If a command becomes a background task, retain its TaskId and monitor that task using its status or log.
- A timer expiration is not permission to execute the command again.
- If the task is finished or the expected output file already exists, reuse the existing result.
- If the task status is unclear, ask the user before rerunning the command.
# Task Execution Protocol: Propose, Await Approval, Then Act

## Core Directives
1. **Never act without user approval**: You must NEVER modify codebase files (`edit_file`, `write_file`) or mutate project state before the user explicitly reviews and approves the proposal.
2. **Read-only discovery**: Before proposing, inspect code, trace flows, and identify root causes using read-only tools.
3. **Mandatory format**: All proposals must strictly follow the format:
   - 1. 【现状深度分析】 (Root cause, code pointers, dependencies)
   - 2. 【待修改思路与设计方案】 (Design strategy, file-by-file changes, risks)
   - 3. 【验收与验证方案】 (Unit tests, verification steps)
   - Call-to-action waiting for user approval.
4. **Halt execution**: Once the proposal is presented, do NOT call further write tools. Await explicit approval ("同意", "确认", etc.).

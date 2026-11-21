---
name: propose-before-act
description: >-
  Standard Operating Procedure for any development, refactoring, or bug-fix task.
  Enforces thorough current-status analysis and proposed solution submission before
  touching code, strictly requiring explicit user approval prior to modifying files.
---

# Propose Before Act (先分析方案，获批后再修改)

## When to use this skill
Triggered automatically on ANY task that requires codebase modification, feature implementation, refactoring, or UI adjustment.

## Procedure
1. **Analyze (Read-Only)**
   - Locate files, line numbers, and trace dataflow.
   - Clarify the root cause or existing mechanism.
2. **Draft Proposal**
   - Detail the modification strategy and exact files to change.
   - Identify edge cases, regression risks, and fallback mechanisms.
3. **Present Proposal & Stop**
   - Output using the standard 3-section template:
     1. 【现状深度分析】
     2. 【待修改思路与设计方案】
     3. 【验收与验证方案】
   - End turn immediately without calling write tools.
4. **Execute ONLY Upon Explicit Approval**
   - When user responds with "同意", "确认", or equivalent confirmation, proceed with the planned edits and run validation tests.

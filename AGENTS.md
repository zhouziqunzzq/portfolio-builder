# Repository coding style

- Prefer readable code over terse code. Separate logical steps with whitespace and explain non-obvious decisions in comments or docstrings.
- Keep functions focused. Extract reusable helpers when a function grows difficult to follow; functions over roughly 150 lines deserve particular scrutiny.
- Use named, typed dataclasses for meaningful data structures instead of passing loosely shaped dictionaries. Explain fields whose purpose or constraints are not obvious.
- Preserve the behavior and safety assumptions established in the relevant feature plan. Record new assumptions near the code they govern.
- Add tests for meaningful behavior and failure paths, then run the relevant checks before reporting completion.

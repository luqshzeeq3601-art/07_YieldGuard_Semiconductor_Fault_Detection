# YieldGuard implementation instructions

## 1. Communication

- Explain directly, briefly and in numbered steps with useful bullets.
- State the result, verification evidence and next ready task.

## 2. Before working

1. Read [START_HERE](docs/00_START_HERE.md), [progress](docs/09_PROGRESS_LOG.md), [decisions](docs/08_DECISIONS_LOG.md) and [tasks](tasks/todo.md).
2. Read the PRD and technical documents relevant to the selected task.
3. Inspect existing files and working changes. Preserve unrelated work and sibling projects.
4. Select the first unfinished task whose dependencies are satisfied.

## 3. Execution rules

- Treat the Markdown documents as the implementation contract. Do not depend on earlier chat history.
- Requirements belong in the PRD; data rules in the data specification; model rules in the experiment plan; task status only in `tasks/todo.md`.
- If documents conflict, resolve and record the decision before implementing the affected behaviour. Human instructions take precedence.
- Complete one bounded task, run its checks, record evidence, then proceed to the next ready task within the authorized session.
- Put reusable logic in `src/yieldguard/`.
- Use Python type hints, `snake_case` functions and pytest tests for meaningful behaviour.
- Keep data processing and serving on the same feature and cleaning implementation.
- Fit all cleaning, sensor selection and thresholds on training blocks only. Never use random row splits, future runs or labels as predictors.
- Keep holdout labels (`data/processed/labels_holdout.parquet`) unavailable to training, tuning, sensor selection and threshold setting. Only the holdout evaluation command may open `labels_holdout.parquet`.
- Required tests must pass on synthetic data without the SECOM download.
- Record missed targets and negative results honestly as measured. Do not change targets, thresholds or protocol after observing results.
- Check official documentation for the versions actually installed; pin compatible dependencies during T01.
- Model artifact loading accepts only project-generated trusted bundles. Never deserialize a user-uploaded model.
- Do not install dependencies globally, touch sibling projects, fabricate results or publish files without authorization.

## 4. Scope control

- Proceed with documented local setup, data acquisition, training, testing and report generation as authorized.
- Stop at declared checkpoints when required by user instructions (specifically C3 before T12 holdout evaluation).
- Ask for direction when changing the primary dataset, split, targets or evaluation protocol.
- External publication, paid resources and public cloud deployment require explicit user authorization.

## 5. End each task / session

1. Mark a task complete in `tasks/todo.md` only when its acceptance criteria pass with evidence.
2. Append a dated entry to `docs/09_PROGRESS_LOG.md` with commands, results, changed files and unresolved issues.
3. Record material design changes in `docs/08_DECISIONS_LOG.md` and update the owning specification.
4. Leave a concrete next task ID and any prerequisite.

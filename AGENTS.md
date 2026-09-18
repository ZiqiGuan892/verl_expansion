# verl-multi-task development rules

This independent repository contains the `multi_task_scheduler` Python package.
It is not the documentation directory of the outer repository.

- Keep the existing Git history/configuration. Do not commit or push without a user request.
- Current authorized work: D0 baseline and acceptance infrastructure only. Do not start D01 or any later capability until the user explicitly confirms that D0 passed.
- Replace the ABC-only scaffold with actual native subclasses/composition and preserve native training behavior.
- Necessary native entry/profile wiring is allowed; do not modify original verl class implementations or restore Impl/SPI patches.
- Remove unnecessary earlier P1 feature protocols, dummy services and tests. The user explicitly requests deletion without backups.
- Do not remove unrelated documents, Git metadata, environments or historical P0 backups under this cleanup authorization.
- Keep package imports dependency-light and free of Ray initialization, actor discovery, global registration and monkey patches.
- An unimplemented runtime adapter must fail explicitly, never fall back to a fake/native class and pretend integration succeeded.
- New business features remain empty; no scheduling, GPU donation, bootstrap, extra membership or concurrency machinery.
- GS discovery and entity creation are real, not empty. Retain only simple state/handles and interfaces needed by this startup chain.
- Reports to GroupScheduler contain metadata only. Donor runtime handles must never enter borrower placement contracts.
- The manager owns local replica runtime references. Checkpoint Engine and load balancing hold separate projections.
- Preserve the single `experimental_fully_async_standalone` profile and the native verl training entry.
- Do not add a companion training entry, mirrored Hydra primary, class-FQN configuration, or old Impl/SPI dependency.
- Use `apply_patch` for source/document edits. Do not overwrite user changes.
- Use a project virtual environment and `uv` for environment management; install only the dependencies needed by selected tests.
- Tests target native entry selection, actual GS discovery, subclass construction/wiring and native delegation. Label mocked coverage explicitly.
- Selected real dependency tests may use a source copy inside verl. Never claim GPU/native-runtime success from AST or mocks.
- Source deployment must preserve unknown files, record provenance and never copy Git, environments or model data. Do not build a copy framework for this stage.
- Document each component's owner, creation point, inherited behavior and the boundary of unimplemented features.
- Report test results by layer. Do not disguise a skipped or mocked integration test as a successful runtime check.

## Current handoff gate

- Read [`Agent.md`](Agent.md) and [`docs/develop_step.md`](docs/develop_step.md) before changing files.
- The current implementation stage is D0. D0 added only GPU acceptance fixtures, the `gpu_integration` marker, a GPU configuration template, and the D0 development log.
- D01 and later work is blocked. Do not implement replica fields, placement validation, PG lookup, lease handling, sleep/wake, shutdown, borrowed runtime creation, CE membership, LB lifecycle, or TaskRunner lifecycle commands until the user confirms D0.
- The D0 real-environment checks are not complete in this workspace: Python, the project virtual environment, GPU, Ray, and native verl/vLLM runtime are unavailable here. Never report D0 as passed based on static checks.
- Do not reset, discard, or overwrite the existing uncommitted documents and tests. Inspect `git status --short` before editing.
- The requested GitHub private repository upload was not completed. Do not assume that a `verl_test` remote exists or that any local files have been pushed.

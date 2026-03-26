# Phase 5/6 Cleanup Manifest

## Purpose

This manifest freezes the pre-cleanup baseline, file ownership, deletion batches, and minimal regression commands for the Phase 5/6 cleanup track.

## Baseline

### Compile baseline

Command:

```bash
./.venv/bin/python -m compileall aletheia
```

Observed result:

- pass

### Minimal regression baseline

Command:

```bash
./.venv/bin/python -m unittest \
  aletheia.tests.test_training_loop_integration.TestTrainingLoopRolloutReplayIntegration.test_build_imagined_batch_emits_controller_signal_metrics \
  aletheia.tests.test_run_train_contracts.TestRunTrainContracts.test_run_train_uses_agent_collector_and_writes_resume_artifacts \
  aletheia.tests.test_controller_signals \
  aletheia.tests.test_controller_relation_audit \
  aletheia.tests.test_controller_signal_audit \
  aletheia.tests.test_post_entry_audit \
  aletheia.tests.test_post_entry_template_compare \
  aletheia.tests.test_post_entry_precursor_audit
```

Observed result:

- pass
- `Ran 19 tests`

### Notes

- historical command names had drifted
- manifest uses the actual test names present in the workspace

## Deletion ownership

### Batch 1: audit-only

Delete or remove references from:

- `aletheia/post_entry_audit.py`
- `aletheia/post_entry_template_compare.py`
- `aletheia/post_entry_precursor_audit.py`
- `aletheia/controller/relation_audit.py`
- `aletheia/controller/signal_audit.py`
- `aletheia/aletheia_api.py`
- `aletheia/tests/test_post_entry_audit.py`
- `aletheia/tests/test_post_entry_template_compare.py`
- `aletheia/tests/test_post_entry_precursor_audit.py`
- `aletheia/tests/test_controller_relation_audit.py`
- `aletheia/tests/test_controller_signal_audit.py`
- `aletheia/tests/test_run_train_contracts.py`

### Batch 2: signal/schema layer

Delete or remove references from:

- `aletheia/controller/signals.py`
- `aletheia/controller/schema.py`
- `aletheia/controller/invariants.py`
- `aletheia/controller/__init__.py`
- `aletheia/aletheia_train.py`
- `aletheia/tests/test_controller_signals.py`
- `aletheia/tests/test_training_loop_integration.py`

### Batch 3: live controller fold

Primary ownership:

- `aletheia/aletheia_train.py`

Secondary follow-up:

- `aletheia/aletheia_config.py`
- `aletheia/tests/test_training_loop_integration.py`

### Batch 4: config / metrics / test cleanup

Primary ownership:

- `aletheia/aletheia_config.py`
- `aletheia/tests/test_training_loop_integration.py`
- any remaining references in `aletheia/aletheia_api.py`

## Minimal surviving behavior after cleanup

The post-cleanup system must keep only:

- `contract`
- `arbiter`
- `minimal compensation`

The following behavior must still exist after Batch 3:

- bounded compensation for actor scale
- bounded compensation for return-cap behavior
- bounded late-tail compensation

The following must no longer exist as first-class architecture:

- controller audit/report subsystem
- controller signal subsystem
- stage-heavy controller schema/invariant subsystem
- large controller-only config families

## Rollback anchors

If Batch 1 fails:

- restore audit imports, summary artifact output, and deleted tests

If Batch 2 fails:

- restore signal snapshot generation in training and controller package exports

If Batch 3 fails:

- revert to the last passing Batch 2 state before folding live controller

## Review gates

### Gate A

After Batch 1:

- compile passes
- no import of deleted audit modules remains
- run-train summary no longer expects post-entry audit artifacts

### Gate B

After Batch 2:

- compile passes
- no `imag/signal_*` emission remains
- no controller schema/invariant imports remain

### Gate C

After Batch 3 and Batch 4:

- compile passes
- training integration passes on new minimal-compensation semantics
- config surface is smaller and free of dead controller-only knobs

### Gate D

After Batch 5:

- final report documents what was removed
- surviving boundary is clearly `contract / arbiter / minimal compensation`

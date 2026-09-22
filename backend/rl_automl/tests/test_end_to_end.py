from __future__ import annotations

import pytest

from rl_automl.core.errors import NotApprovedError
from rl_automl.core.types import RunStatus
from rl_automl.dataset.loaders import load_source
from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest


@pytest.mark.slow
def test_small_end_to_end_plan_approve_execute_and_reload(config):
    frame, _source = load_source("iris")
    config.rl.enabled = False
    orchestrator = AutoMLOrchestrator(config)
    run = orchestrator.plan(
        PlanRequest(
            problem_statement="Classify iris target using flower measurements",
            frame=frame,
            dataset_name="iris.csv",
            dataset_sha256="test",
            target="species",
            task_type="classification",
            metric="f1",
            use_policy=False,
        )
    )
    assert run.status is RunStatus.AWAITING_APPROVAL
    assert run.rl_recommendations
    with pytest.raises(NotApprovedError):
        orchestrator.execute(run, frame=frame)
    run = orchestrator.approve(run, selection=[1])
    run = orchestrator.execute(run, frame=frame)
    assert run.status is RunStatus.COMPLETE, run.error
    assert run.best_model is not None
    assert run.artifacts.verified
    assert orchestrator.load_run(run.run_id).status is RunStatus.COMPLETE

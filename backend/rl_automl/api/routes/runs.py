"""Run endpoints: create, inspect, approve, cancel, and stream progress.

The approval gate is visible in the shape of this API: a run is created, it plans, it stops
in ``awaiting_approval``, and nothing in this module can move it past that point except an
explicit ``POST /runs/{id}/approve``.
"""

from __future__ import annotations

import json
import queue
from collections.abc import Iterator
from typing import Any

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from rl_automl.api.dependencies import get_manager
from rl_automl.api.run_manager import RunManager
from rl_automl.api.schemas import (
    ApproveRequest,
    CancelRequest,
    RecommendationResponse,
    RunCreateRequest,
    RunSummary,
)
from rl_automl.core.errors import AutoMLError, RunStateError
from rl_automl.core.logging import get_logger
from rl_automl.core.types import RunObject, RunStatus

logger = get_logger("api.routes.runs")

router = APIRouter(prefix="/runs", tags=["runs"])

TERMINAL_STAGES = {RunStatus.COMPLETE.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
KEEP_ALIVE_S = 15.0


def _summarise(run: RunObject) -> RunSummary:
    best = run.best_model
    return RunSummary(
        run_id=run.run_id,
        status=run.status.value,
        created_at=run.created_at,
        updated_at=run.updated_at,
        problem_statement=run.problem_statement,
        dataset_name=run.dataset_name,
        task_type=run.task.task_type.value if run.task else None,
        metric=run.task.metric if run.task else None,
        best_model=best.model if best else None,
        validation_score=best.validation_score if best else None,
        test_score=best.test_score if best else None,
        n_recommended=len(run.rl_recommendations),
        error=run.error,
    )


def _not_found(exc: RunStateError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.to_dict())


@router.post("", response_model=RunObject, status_code=status.HTTP_202_ACCEPTED)
def create_run(
    payload: RunCreateRequest,
    manager: RunManager = Depends(get_manager),
) -> RunObject:
    """Create a run and begin planning. Planning trains nothing; approval comes next."""
    dataset_id = payload.dataset_id
    if dataset_id is None:
        if payload.source is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="provide either 'dataset_id' or 'source'",
            )
        try:
            dataset_id = manager.register_source(payload.source).dataset_id
        except AutoMLError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=exc.to_dict()
            ) from exc

    try:
        return manager.submit_plan(
            problem_statement=payload.problem_statement,
            dataset_id=dataset_id,
            target=payload.target,
            task_type=payload.task_type,
            metric=payload.metric,
            use_policy=payload.use_policy,
        )
    except RunStateError as exc:
        raise _not_found(exc) from exc


@router.get("", response_model=list[RunSummary])
def list_runs(manager: RunManager = Depends(get_manager)) -> list[RunSummary]:
    return [_summarise(run) for run in manager.list_runs()]


@router.get("/{run_id}", response_model=RunObject)
def get_run(run_id: str, manager: RunManager = Depends(get_manager)) -> RunObject:
    try:
        return manager.get_run(run_id)
    except RunStateError as exc:
        raise _not_found(exc) from exc


@router.get("/{run_id}/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    run_id: str, manager: RunManager = Depends(get_manager)
) -> RecommendationResponse:
    try:
        run = manager.get_run(run_id)
    except RunStateError as exc:
        raise _not_found(exc) from exc

    recommendation = run.recommendation_set
    if recommendation is None:
        return RecommendationResponse(
            run_id=run_id,
            status=run.status.value,
            notes=["the run has not finished planning yet", *run.messages],
        )

    return RecommendationResponse(
        run_id=run_id,
        status=run.status.value,
        planner=recommendation.planner,
        task_type=recommendation.task.value,
        metric=recommendation.metric,
        metric_direction=recommendation.metric_direction.value,
        requires_user_approval=recommendation.requires_user_approval,
        estimated_runtime_s=recommendation.estimated_runtime_s,
        estimated_compute=recommendation.estimated_compute,
        baseline_expectation=recommendation.baseline_expectation,
        notes=list(recommendation.notes),
        recommendations=[
            item.model_dump(mode="json") for item in recommendation.recommended_experiments
        ],
    )


@router.post("/{run_id}/approve", response_model=RunObject, status_code=status.HTTP_202_ACCEPTED)
def approve_run(
    run_id: str,
    payload: ApproveRequest,
    manager: RunManager = Depends(get_manager),
) -> RunObject:
    """Authorise pipelines. This is the only path that can start training."""
    try:
        return manager.approve(run_id, selection=payload.selection, approved_by=payload.approved_by)
    except RunStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.to_dict()) from exc


@router.post("/{run_id}/cancel", response_model=RunObject)
def cancel_run(
    run_id: str,
    payload: CancelRequest | None = None,
    manager: RunManager = Depends(get_manager),
) -> RunObject:
    reason = payload.reason if payload is not None else "cancelled by user"
    try:
        return manager.cancel(run_id, reason)
    except RunStateError as exc:
        raise _not_found(exc) from exc


def _render_event(payload: dict[str, Any]) -> str:
    stage = str(payload.get("stage", "message"))
    return f"event: {stage}\ndata: {json.dumps(payload, default=str)}\n\n"


def _next_event(channel: queue.Queue, timeout: float) -> Any:
    try:
        return channel.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError from None


@router.get("/{run_id}/events")
async def stream_events(
    run_id: str,
    request: Request,
    manager: RunManager = Depends(get_manager),
) -> StreamingResponse:
    """Server-sent events for one run.

    The recorded history is replayed first, then live events are streamed. The subscription
    is created *before* the replay and events already present in the history are dropped by
    sequence number, so an event cannot fall into the gap between the two.
    """
    try:
        run = manager.get_run(run_id)
    except RunStateError as exc:
        raise _not_found(exc) from exc

    async def publisher() -> Iterator[str]:
        channel = manager.subscribe(run_id)
        try:
            history = list(manager.replay(run_id))
            sequence = history[-1].get("sequence", 0) if history else 0
            for event in history:
                yield _render_event(event)

            if run.status.is_terminal:
                yield _render_event(
                    {
                        "run_id": run_id,
                        "stage": run.status.value,
                        "message": "run is already finished",
                        "sequence": sequence + 1,
                        "progress": 1.0,
                    }
                )
                return

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await anyio.to_thread.run_sync(_next_event, channel, KEEP_ALIVE_S)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                payload = event.model_dump(mode="json")
                if payload.get("sequence", 0) <= sequence:
                    continue
                sequence = payload.get("sequence", sequence)
                yield _render_event(payload)
                if payload.get("stage") in TERMINAL_STAGES:
                    break
        finally:
            manager.unsubscribe(run_id, channel)

    return StreamingResponse(
        publisher(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["router"]

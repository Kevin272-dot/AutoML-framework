"""Asynchronous run ownership: threads, the run registry, and the event fan-out.

The orchestrator is **synchronous on purpose** -- a run is a sequence of steps whose order
is the correctness argument, and hiding that behind async machinery is how approval gates
get bypassed by a second code path. So asynchrony lives here instead, in one place, as a
small thread pool:

* ``plan`` and ``execute`` each run on a worker thread, never on the event loop
* progress events from the orchestrator are pushed to every subscriber of that run
* the run objects stay in memory while they are live, and are reloaded from disk on
  startup so a restarted API still serves its history

The approval gate is untouched by all of this: nothing here calls ``execute`` except the
job submitted by an explicit ``approve``.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from rl_automl.core.config import AutoMLConfig
from rl_automl.core.errors import AutoMLError, RunStateError
from rl_automl.core.logging import get_logger
from rl_automl.core.types import (
    ExecutionEvent,
    RunObject,
    RunStatus,
    new_id,
    utc_now_iso,
)
from rl_automl.dataset.loaders import load_source, save_frame
from rl_automl.dataset.validation import store_upload
from rl_automl.orchestrator import AutoMLOrchestrator, PlanRequest

logger = get_logger("api.run_manager")

ACTIVE_STATUSES = frozenset(
    {
        RunStatus.CREATED,
        RunStatus.PROFILING,
        RunStatus.UNDERSTANDING,
        RunStatus.PLANNING,
        RunStatus.EXECUTING,
        RunStatus.EVALUATING,
        RunStatus.COMPARING,
        RunStatus.SELECTING,
        RunStatus.PACKAGING,
    }
)

SUBSCRIBER_QUEUE_SIZE = 512


@dataclass
class StoredDataset:
    """A dataset that has passed the upload gate and lives under the artifact tree."""

    dataset_id: str
    name: str
    path: str
    n_rows: int
    n_cols: int
    sha256: str
    columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "path": self.path,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "sha256": self.sha256,
            "columns": list(self.columns),
            "warnings": list(self.warnings),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StoredDataset:
        return cls(**payload)


class RunManager:
    def __init__(self, config: AutoMLConfig | None = None) -> None:
        self.config = config or AutoMLConfig.load()
        self.config.ensure_directories()

        self.orchestrator = AutoMLOrchestrator(self.config, listener=self._on_event)
        self._lock = threading.RLock()
        self._runs: dict[str, RunObject] = {}
        self._datasets: dict[str, StoredDataset] = {}
        self._subscribers: dict[str, set[queue.Queue]] = {}
        self._cancel_requested: set[str] = set()
        self._execution_submitted: set[str] = set()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.config.api.max_concurrent_runs)),
            thread_name_prefix="automl-run",
        )

        self._load_state()

    # ==================================================================================
    # Datasets
    # ==================================================================================

    def datasets_dir(self) -> Path:
        return self.config.artifacts_dir() / "datasets"

    def register_source(self, source_name: str, name: str | None = None) -> StoredDataset:
        """Materialise a bundled/synthetic dataset to disk and register it.

        Writing it to disk rather than holding it in memory is what lets a run be executed
        asynchronously, and re-executed after a restart, without the dataset being passed
        around: the run carries a ``dataset_path`` like any other dataset.
        """
        frame, source = load_source(source_name, cache_dir=self.config.remote_dataset_dir())
        return self.register_frame(frame, name=name or source.name, source_label=source_name)

    def register_frame(
        self, frame: pd.DataFrame, *, name: str, source_label: str = ""
    ) -> StoredDataset:
        dataset_id = new_id("data")
        directory = self.datasets_dir() / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        path = save_frame(frame, directory / "data.csv")

        from rl_automl.dataset.validation import sha256_file

        record = StoredDataset(
            dataset_id=dataset_id,
            name=name,
            path=str(path),
            n_rows=int(frame.shape[0]),
            n_cols=int(frame.shape[1]),
            sha256=sha256_file(path),
            columns=[str(column) for column in frame.columns],
            warnings=[f"materialised from '{source_label}'"] if source_label else [],
        )
        self._save_dataset(record)
        with self._lock:
            self._datasets[dataset_id] = record
        logger.info(
            "dataset registered",
            extra={"context": {"dataset_id": dataset_id, "name": name, "rows": record.n_rows}},
        )
        return record

    def register_upload(self, source_path: str | Path, filename: str) -> StoredDataset:
        """Send an uploaded file through the security gate and register the result."""
        dataset_id = new_id("data")
        directory = self.datasets_dir() / dataset_id
        cleaned, report = store_upload(
            source_path, directory, filename=filename, config=self.config.security
        )
        record = StoredDataset(
            dataset_id=dataset_id,
            name=Path(report.filename).name,
            path=str(cleaned),
            n_rows=report.n_rows,
            n_cols=report.n_cols,
            sha256=report.sha256,
            columns=[str(column) for column in pd.read_csv(cleaned, nrows=0).columns],
            warnings=list(report.warnings),
        )
        self._save_dataset(record)
        with self._lock:
            self._datasets[dataset_id] = record
        logger.info(
            "uploaded dataset registered",
            extra={"context": {"dataset_id": dataset_id, "rows": record.n_rows}},
        )
        return record

    def get_dataset(self, dataset_id: str) -> StoredDataset:
        with self._lock:
            record = self._datasets.get(dataset_id)
        if record is None:
            raise RunStateError(f"dataset '{dataset_id}' was not found", dataset_id=dataset_id)
        return record

    def list_datasets(self) -> list[StoredDataset]:
        with self._lock:
            return sorted(self._datasets.values(), key=lambda item: item.created_at, reverse=True)

    def dataset_preview(self, record: StoredDataset, rows: int = 5) -> list[dict[str, Any]]:
        try:
            frame = pd.read_csv(record.path, nrows=rows)
        except (OSError, ValueError):  # pragma: no cover - file removed out of band
            return []
        return json.loads(frame.to_json(orient="records", date_format="iso"))

    # ==================================================================================
    # Runs
    # ==================================================================================

    def submit_plan(
        self,
        *,
        problem_statement: str,
        dataset_id: str,
        target: str | None = None,
        task_type: str | None = None,
        metric: str | None = None,
        use_policy: bool = True,
    ) -> RunObject:
        """Create a run and start planning on a worker thread."""
        record = self.get_dataset(dataset_id)
        run_id = new_id("run")

        placeholder = RunObject(
            run_id=run_id,
            status=RunStatus.CREATED,
            problem_statement=problem_statement,
            dataset_name=record.name,
            dataset_path=record.path,
            dataset_sha256=record.sha256,
            config_snapshot=self.config.to_snapshot(),
        )
        with self._lock:
            self._runs[run_id] = placeholder
        self.orchestrator.save_run(placeholder)

        request = PlanRequest(
            problem_statement=problem_statement,
            frame=pd.DataFrame(),  # never used: the worker reloads from dataset_path
            dataset_name=record.name,
            dataset_sha256=record.sha256,
            dataset_path=record.path,
            target=target,
            task_type=task_type,
            metric=metric,
            run_id=run_id,
            use_policy=use_policy,
        )
        self._pool.submit(self._plan_job, request)
        return placeholder

    def submit_execution(self, run_id: str, run: RunObject) -> None:
        """Queue the only call path that trains anything. Reached via ``approve`` only."""
        self._pool.submit(self._execute_job, run_id, run)

    def _plan_job(self, request: PlanRequest) -> None:
        try:
            frame = self._load_frame(request.dataset_path)
            request.frame = frame
            run = self.orchestrator.plan(request)
            self._store(run)
        except AutoMLError as exc:
            self._fail(request.run_id or "", exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("planning failed", extra={"context": {"run_id": request.run_id}})
            self._fail(request.run_id or "", exc)

    def _execute_job(self, run_id: str, run: RunObject) -> None:
        try:
            frame = self._load_frame(run.dataset_path)
            result = self.orchestrator.execute(run, frame=frame)
            if run_id in self._cancel_requested:
                # Cancellation during execution is honoured at the step boundary: the run
                # is not claimed complete when the user asked for it to stop.
                result = self.orchestrator.cancel(result, "cancelled by user during execution")
                self._cancel_requested.discard(run_id)
            self._store(result)
        except AutoMLError as exc:
            self._fail(run_id, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("execution failed", extra={"context": {"run_id": run_id}})
            self._fail(run_id, exc)

    def approve(
        self,
        run_id: str,
        *,
        selection: str | list[int] | list[str] = "all",
        approved_by: str = "user",
    ) -> RunObject:
        """Authorise pipelines and start execution, exactly once per run.

        The claim below is what makes this safe. The orchestrator deliberately leaves a run
        in ``awaiting_approval`` after approval -- approval records intent, it does not
        start work -- so status alone cannot distinguish "approved, waiting to start" from
        "not yet approved". Two requests arriving in that window would both pass a
        status check and both train the same pipelines into the same directory.
        """
        run = self.get_run(run_id)
        with self._lock:
            if run_id in self._execution_submitted:
                raise RunStateError(
                    f"run '{run_id}' has already been approved and submitted for execution",
                    run_id=run_id,
                )
            if run.status is not RunStatus.AWAITING_APPROVAL:
                raise RunStateError(
                    f"run '{run_id}' is in state '{run.status.value}' and cannot be approved",
                    run_id=run_id,
                    status=run.status.value,
                )
            self._execution_submitted.add(run_id)

        try:
            approved = self.orchestrator.approve(run, selection=selection, approved_by=approved_by)
        except BaseException:
            # A rejected selection must stay retryable.
            with self._lock:
                self._execution_submitted.discard(run_id)
            raise

        self._store(approved)
        self.submit_execution(run_id, approved)
        return approved

    def cancel(self, run_id: str, reason: str = "cancelled by user") -> RunObject:
        run = self.get_run(run_id)
        if run.status.is_terminal:
            return run

        if run.status in ACTIVE_STATUSES and run.status is not RunStatus.CREATED:
            # Already executing: record the request and resolve it at the step boundary.
            with self._lock:
                self._cancel_requested.add(run_id)
            run.messages.append(f"cancellation requested: {reason}")
            self._store(run)
            return run

        cancelled = self.orchestrator.cancel(run, reason)
        self._store(cancelled)
        return cancelled

    def get_run(self, run_id: str) -> RunObject:
        with self._lock:
            run = self._runs.get(run_id)
        if run is not None:
            return run
        try:
            run = self.orchestrator.load_run(run_id)
        except RunStateError:
            raise
        with self._lock:
            self._runs[run_id] = run
        return run

    def list_runs(self) -> list[RunObject]:
        with self._lock:
            in_memory = dict(self._runs)
        for summary in self.orchestrator.list_runs():
            run_id = summary.get("run_id")
            if run_id and run_id not in in_memory:
                try:
                    in_memory[run_id] = self.orchestrator.load_run(run_id)
                except RunStateError:  # pragma: no cover - raced with a delete
                    continue
        return sorted(in_memory.values(), key=lambda run: run.created_at, reverse=True)

    def artifact_path(self, run_id: str, kind: str) -> Path:
        """Resolve an artifact for download, refusing anything outside the artifact tree."""
        run = self.get_run(run_id)
        if kind == "model":
            recorded = run.artifacts.model_zip
        elif kind == "results":
            recorded = run.artifacts.results_zip
        else:
            raise RunStateError(f"unknown artifact kind '{kind}'", kind=kind)

        if not recorded:
            raise RunStateError(
                f"run '{run_id}' has no {kind} artifact; it may still be running or may "
                "have failed",
                run_id=run_id,
                kind=kind,
            )

        path = Path(recorded).resolve()
        root = self.config.artifacts_dir().resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RunStateError(
                f"the {kind} artifact for run '{run_id}' is not available",
                run_id=run_id,
                kind=kind,
            )
        return path

    # ==================================================================================
    # Events
    # ==================================================================================

    def subscribe(self, run_id: str) -> queue.Queue:
        channel: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            self._subscribers.setdefault(run_id, set()).add(channel)
        return channel

    def unsubscribe(self, run_id: str, channel: queue.Queue) -> None:
        with self._lock:
            channels = self._subscribers.get(run_id)
            if not channels:
                return
            channels.discard(channel)
            if not channels:
                self._subscribers.pop(run_id, None)

    def _on_event(self, event: ExecutionEvent) -> None:
        """Orchestrator listener. Runs on whichever worker thread produced the event."""
        with self._lock:
            channels = list(self._subscribers.get(event.run_id, ()))
        for channel in channels:
            try:
                channel.put_nowait(event)
            except queue.Full:
                # A stalled client must not stall a run; the replay on reconnect covers it.
                logger.warning(
                    "event subscriber is not draining; dropping an event",
                    extra={"context": {"run_id": event.run_id}},
                )

    def replay(self, run_id: str) -> Iterator[dict[str, Any]]:
        run = self.get_run(run_id)
        yield from run.progress

    # ==================================================================================
    # Internals
    # ==================================================================================

    def _store(self, run: RunObject) -> None:
        with self._lock:
            self._runs[run.run_id] = run
        self.orchestrator.save_run(run)

    def _fail(self, run_id: str, exc: Exception) -> None:
        from rl_automl.orchestrator import _error_message

        message = _error_message(exc)
        try:
            run = self.get_run(run_id)
        except RunStateError:  # pragma: no cover - the run never got persisted
            return
        run.status = RunStatus.FAILED
        run.error = message
        run.messages.append(message)
        self._store(run)

    def _load_frame(self, path: str) -> pd.DataFrame:
        if not path:
            raise RunStateError("the run has no dataset path")
        resolved = Path(path)
        if resolved.suffix.lower() == ".parquet":
            return pd.read_parquet(resolved)
        return pd.read_csv(resolved)

    def _save_dataset(self, record: StoredDataset) -> None:
        directory = self.datasets_dir() / record.dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "dataset.json").write_text(
            json.dumps(record.to_dict(), indent=2), encoding="utf-8"
        )

    def _load_state(self) -> None:
        """Reload datasets and runs written by a previous process."""
        root = self.datasets_dir()
        if root.is_dir():
            for directory in root.iterdir():
                record_path = directory / "dataset.json"
                if not record_path.is_file():
                    continue
                try:
                    payload = json.loads(record_path.read_text(encoding="utf-8"))
                    record = StoredDataset.from_dict(payload)
                except (json.JSONDecodeError, TypeError, ValueError):  # pragma: no cover
                    logger.warning("skipping an unreadable dataset record: %s", record_path)
                    continue
                if Path(record.path).is_file():
                    self._datasets[record.dataset_id] = record

        logger.info(
            "run manager ready",
            extra={"context": {"datasets": len(self._datasets)}},
        )

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=True)

    @property
    def dataset_count(self) -> int:
        with self._lock:
            return len(self._datasets)

    @property
    def run_count(self) -> int:
        with self._lock:
            return len(self._runs)


__all__ = ["ACTIVE_STATUSES", "RunManager", "StoredDataset"]

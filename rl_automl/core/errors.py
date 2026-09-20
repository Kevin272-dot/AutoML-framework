"""Typed error taxonomy.

Every failure the system can produce has a stable, catchable type. The executor
relies on this to isolate a failing pipeline without aborting a whole run.
"""

from __future__ import annotations

from typing import Any


class AutoMLError(Exception):
    """Base class for all errors raised by this package."""

    #: Stable machine-readable code, surfaced in run metadata and API responses.
    code = "automl_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigError(AutoMLError):
    code = "config_error"


class DatasetValidationError(AutoMLError):
    """The uploaded dataset failed a validation rule."""

    code = "dataset_validation_error"


class SecurityError(AutoMLError):
    """An input violated a security policy (untrusted type, unsafe name, oversize)."""

    code = "security_error"


class TaskUnderstandingError(AutoMLError):
    code = "task_understanding_error"


class ModelFitError(AutoMLError):
    """A model failed to fit or predict. Isolated per pipeline by the executor."""

    code = "model_fit_error"


class RegistryError(AutoMLError):
    code = "registry_error"


class BudgetExceededError(AutoMLError):
    """An enforced cap (experiments, runtime, memory) was hit."""

    code = "budget_exceeded"


class RunStateError(AutoMLError):
    """An operation was attempted in an invalid run state (e.g. execute before approve)."""

    code = "run_state_error"


class NotApprovedError(RunStateError):
    """Execution was attempted without explicit user approval."""

    code = "not_approved"


class PolicyLoadError(AutoMLError):
    """A saved policy is incompatible with the current action space or encoder schema."""

    code = "policy_load_error"


class PackagingError(AutoMLError):
    code = "packaging_error"


class ArtifactIntegrityError(PackagingError):
    """A generated archive failed verification before being offered for download."""

    code = "artifact_integrity_error"


__all__ = [
    "ArtifactIntegrityError",
    "AutoMLError",
    "BudgetExceededError",
    "ConfigError",
    "DatasetValidationError",
    "ModelFitError",
    "NotApprovedError",
    "PackagingError",
    "PolicyLoadError",
    "RegistryError",
    "RunStateError",
    "SecurityError",
    "TaskUnderstandingError",
]

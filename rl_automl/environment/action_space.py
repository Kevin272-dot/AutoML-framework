"""Hierarchical action space (spec §7).

The action is factored into four heads rather than enumerated as a flat product, which
keeps the space small enough for PPO to explore while still expressing
model + preprocessing + hyperparameters + feature selection:

===========================  ==========================================================
head 0 ``model``             index 0 is ``STOP``, then one slot per registry entry
head 1 ``preprocessing``     multi-binary over :data:`PREPROCESSING_OPTIONS`
head 2 ``preset``            hyperparameter preset index, **masked given head 0**
head 3 ``feature_selection`` feature-selection strategy
===========================  ==========================================================

Heads 1-3 are conditioned on the sampled model slot, so this is a proper autoregressive
factorisation, not four independent guesses. Masks are *derived from the registry*, which
is the mechanism that keeps the RL core independent of the model library: registering a
new algorithm changes the action space declaratively and this file does not change at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_automl.core.errors import RegistryError
from rl_automl.core.types import ExperimentSource, ExperimentSpec, TaskType
from rl_automl.core.vocabulary import (
    FEATURE_SELECTION_OPTIONS,
    PREPROCESSING_OPTIONS,
    SUPERVISED_ONLY_FEATURE_SELECTION,
    FeatureSelectionOption,
    PreprocessingOption,
)
from rl_automl.search.model_registry import ModelSpec

ACTION_SPACE_VERSION = "1.0.0"

STOP_INDEX = 0

#: When several mutually-exclusive toggles are selected, this order decides which wins.
#: The multi-binary head samples each toggle independently, so conflicts are resolved
#: deterministically here rather than by conditioning the head on itself.
_SCALING_PRIORITY = (
    PreprocessingOption.SCALE_ROBUST.value,
    PreprocessingOption.SCALE_STANDARD.value,
)
_MUTUALLY_EXCLUSIVE_GROUPS = (_SCALING_PRIORITY,)


@dataclass(frozen=True)
class ActionSpaceLayout:
    """Immutable description of the space. Serialised into policy checkpoints."""

    version: str
    model_keys: tuple[str, ...]
    n_preprocessing: int
    max_presets: int
    feature_selection: tuple[str, ...]
    task_type: str

    @property
    def n_heads(self) -> int:
        return 4

    @property
    def n_model_slots(self) -> int:
        """Model slots including the leading STOP action."""
        return len(self.model_keys) + 1

    @property
    def head_sizes(self) -> tuple[int, ...]:
        return (
            self.n_model_slots,
            self.n_preprocessing,
            self.max_presets,
            len(self.feature_selection),
        )

    def model_key(self, slot: int) -> str | None:
        """Registry key for a slot, or ``None`` for STOP."""
        if slot == STOP_INDEX:
            return None
        index = slot - 1
        if not 0 <= index < len(self.model_keys):
            raise RegistryError(f"model slot {slot} is out of range", n_slots=self.n_model_slots)
        return self.model_keys[index]

    def slot_for(self, model_key: str) -> int:
        try:
            return self.model_keys.index(model_key) + 1
        except ValueError as exc:
            raise RegistryError(
                f"model '{model_key}' is not part of this action space", model=model_key
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_keys": list(self.model_keys),
            "n_preprocessing": self.n_preprocessing,
            "max_presets": self.max_presets,
            "feature_selection": list(self.feature_selection),
            "task_type": self.task_type,
            "head_sizes": list(self.head_sizes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ActionSpaceLayout:
        return cls(
            version=payload["version"],
            model_keys=tuple(payload["model_keys"]),
            n_preprocessing=int(payload["n_preprocessing"]),
            max_presets=int(payload["max_presets"]),
            feature_selection=tuple(payload["feature_selection"]),
            task_type=payload["task_type"],
        )


@dataclass
class ActionMaskTable:
    """Masks for every head, indexed by the sampled model slot.

    ``model`` has one entry per slot. The other arrays have one row per slot so that
    after sampling head 0 the remaining three masks can be looked up directly.
    """

    model: np.ndarray  # (n_model_slots,)
    preprocessing: np.ndarray  # (n_model_slots, n_preprocessing)
    preset: np.ndarray  # (n_model_slots, max_presets)
    feature_selection: np.ndarray  # (n_model_slots, n_feature_selection)

    def for_slot(self, slot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.preprocessing[slot],
            self.preset[slot],
            self.feature_selection[slot],
        )

    def is_legal_model(self, slot: int) -> bool:
        return bool(self.model[slot])

    def legal_model_slots(self) -> list[int]:
        return [slot for slot in range(len(self.model)) if self.model[slot]]

    def experiment_slots(self) -> list[int]:
        """Legal slots that denote an actual experiment.

        ``STOP`` is deliberately excluded: stopping is the RL agent's decision, whereas the
        random/greedy/grid baselines consume a fixed experiment budget (spec §30).
        """
        return [slot for slot in self.legal_model_slots() if slot != STOP_INDEX]

    def summary(self) -> dict[str, Any]:
        return {
            "n_legal_models": int(self.model.sum()),
            "stop_legal": bool(self.model[STOP_INDEX]),
            "n_legal_preprocessing": int(self.preprocessing.any(axis=1).sum()),
        }


class ActionSpace:
    """Builds the layout and masks, and converts between actions and experiments."""

    def __init__(
        self,
        task_type: TaskType,
        specs: list[ModelSpec],
        *,
        min_experiments_before_stop: int = 1,
        allow_stop: bool = True,
    ) -> None:
        self.task_type = task_type
        self.specs = specs
        self.min_experiments_before_stop = max(1, min_experiments_before_stop)
        self.allow_stop = allow_stop

        self._by_slot: dict[int, ModelSpec] = {index + 1: spec for index, spec in enumerate(specs)}
        self.layout = ActionSpaceLayout(
            version=ACTION_SPACE_VERSION,
            model_keys=tuple(spec.key for spec in specs),
            n_preprocessing=len(PREPROCESSING_OPTIONS),
            max_presets=max((spec.n_presets(task_type) for spec in specs), default=1),
            feature_selection=tuple(FEATURE_SELECTION_OPTIONS),
            task_type=task_type.value,
        )

    # -- layout ------------------------------------------------------------------

    @property
    def n_model_slots(self) -> int:
        return self.layout.n_model_slots

    def spec_for_slot(self, slot: int) -> ModelSpec | None:
        return self._by_slot.get(slot)

    # -- masks -------------------------------------------------------------------

    def mask_table(
        self,
        *,
        n_experiments: int = 0,
        elapsed_fraction: float = 0.0,
    ) -> ActionMaskTable:
        """Derive every mask from the registry plus the current search progress."""
        n_slots = self.n_model_slots
        n_preproc = self.layout.n_preprocessing
        max_presets = self.layout.max_presets
        n_fs = len(self.layout.feature_selection)

        model_mask = np.zeros(n_slots, dtype=bool)
        preprocessing_mask = np.zeros((n_slots, n_preproc), dtype=bool)
        preset_mask = np.zeros((n_slots, max_presets), dtype=bool)
        feature_mask = np.zeros((n_slots, n_fs), dtype=bool)

        for slot, spec in self._by_slot.items():
            available, _reason = spec.is_available()
            model_mask[slot] = available and spec.supports(self.task_type)

            forbidden = set(spec.forbidden_preprocessing)
            for index, option in enumerate(PREPROCESSING_OPTIONS):
                # A forbidden toggle is masked out rather than silently ignored, so the
                # policy learns which options are meaningful for which algorithm.
                preprocessing_mask[slot, index] = option not in forbidden

            n_presets = spec.n_presets(self.task_type)
            preset_mask[slot, :n_presets] = True

            for index, strategy in enumerate(FEATURE_SELECTION_OPTIONS):
                feature_mask[slot, index] = self._feature_selection_legal(spec, strategy)

        # STOP: always legal in principle, but held back until the search has produced
        # something, otherwise a fresh episode could end after zero experiments and the
        # planner would have nothing to recommend.
        any_model_available = bool(model_mask[1:].any())
        model_mask[STOP_INDEX] = bool(
            self.allow_stop
            and n_experiments >= self.min_experiments_before_stop
            and any_model_available
        )
        # Degenerate case: if no model can be run at all, STOP is the only sane action.
        if not any_model_available:
            model_mask[STOP_INDEX] = True

        return ActionMaskTable(
            model=model_mask,
            preprocessing=preprocessing_mask,
            preset=preset_mask,
            feature_selection=feature_mask,
        )

    def _feature_selection_legal(self, spec: ModelSpec, strategy: str) -> bool:
        if strategy in SUPERVISED_ONLY_FEATURE_SELECTION and not self.task_type.is_supervised:
            return False
        # PCA is the model for this task; using it again as a stage is redundant.
        return not (
            strategy == FeatureSelectionOption.PCA.value
            and self.task_type is TaskType.DIMENSIONALITY_REDUCTION
        )

    # -- decoding ----------------------------------------------------------------

    def decode(
        self,
        action: np.ndarray | tuple[int, ...] | list[int],
        *,
        source: ExperimentSource = ExperimentSource.RL,
        rationale: str = "",
    ) -> ExperimentSpec | None:
        """Turn an action into an experiment, or ``None`` for ``STOP``."""
        values = _normalise_action(action)
        n_preprocessing = self.layout.n_preprocessing
        model_slot = values[0]
        preprocessing_bits = values[1 : 1 + n_preprocessing]
        preset_index = values[1 + n_preprocessing]
        feature_index = values[2 + n_preprocessing]

        if model_slot == STOP_INDEX:
            return None

        spec = self.spec_for_slot(model_slot)
        if spec is None:
            raise RegistryError(f"model slot {model_slot} has no registered model")

        if not spec.supports(self.task_type):
            raise RegistryError(
                f"{spec.display_name} does not support {self.task_type.value}",
                model=spec.key,
            )

        n_presets = spec.n_presets(self.task_type)
        if not 0 <= preset_index < n_presets:
            raise RegistryError(
                f"preset {preset_index} is out of range for {spec.key}",
                model=spec.key,
                n_presets=n_presets,
            )

        toggles = self.resolve_preprocessing(preprocessing_bits, spec)
        feature_selection = self.layout.feature_selection[feature_index]
        preset = spec.preset(preset_index, self.task_type)

        return ExperimentSpec(
            model=spec.key,
            preprocessing=toggles,
            feature_selection=feature_selection,
            preset_index=preset_index,
            # Hyperparameters are materialised onto the spec so that the recommendation
            # and the executed pipeline are self-describing and reproducible even if the
            # registry's presets change later.
            hyperparameters=dict(preset.params),
            expected_cost=preset.cost,
            source=source,
            rationale=rationale or f"{spec.display_name} · {preset.name} preset",
        )

    def resolve_preprocessing(self, bits: np.ndarray | list[int], spec: ModelSpec) -> list[str]:
        """Convert a multi-binary head into a validated, conflict-free toggle list."""
        forbidden = set(spec.forbidden_preprocessing)
        selected = [
            option
            for option, bit in zip(PREPROCESSING_OPTIONS, bits, strict=False)
            if bit and option not in forbidden
        ]

        for group in _MUTUALLY_EXCLUSIVE_GROUPS:
            present = [option for option in group if option in selected]
            if len(present) > 1:
                winner = next(option for option in group if option in present)
                selected = [
                    option for option in selected if option not in group or option == winner
                ]

        # Preserve the canonical order so that signatures and de-duplication are stable.
        return [option for option in PREPROCESSING_OPTIONS if option in selected]

    def encode(self, spec: ExperimentSpec) -> tuple[int, ...]:
        """Inverse of :meth:`decode`, used for logging and duplicate detection."""
        slot = self.layout.slot_for(spec.model)
        bits = tuple(
            1 if option in set(spec.preprocessing) else 0 for option in PREPROCESSING_OPTIONS
        )
        feature_index = (
            self.layout.feature_selection.index(spec.feature_selection)
            if spec.feature_selection in self.layout.feature_selection
            else 0
        )
        return (slot, *bits, spec.preset_index, feature_index)

    # -- helpers -----------------------------------------------------------------

    def empty_preprocessing_mask(self) -> np.ndarray:
        return np.zeros(self.layout.n_preprocessing, dtype=bool)

    def sample_random_action(self, masks: ActionMaskTable, rng: np.random.Generator) -> np.ndarray:
        """Uniform sample over legal experiments. Used by the random-search baseline."""
        legal_slots = masks.experiment_slots()
        if not legal_slots:
            raise RegistryError(
                "no legal experiment slots; every registered model is unavailable "
                "or does not support this task"
            )
        slot = int(rng.choice(legal_slots))
        preprocessing = masks.preprocessing[slot] & (rng.random(masks.preprocessing.shape[1]) < 0.5)
        preset = int(rng.choice(np.flatnonzero(masks.preset[slot])))
        feature = int(rng.choice(np.flatnonzero(masks.feature_selection[slot])))
        return np.concatenate(
            [np.asarray([slot]), preprocessing.astype(int), np.asarray([preset, feature])]
        )

    def all_legal_specs(self, masks: ActionMaskTable) -> list[ExperimentSpec]:
        """Enumerate the grid the grid-search baseline walks.

        Covers model x preset x feature-selection with the *full legal preprocessing set*
        switched on. Toggle subsets are not enumerated exhaustively because 2^9
        preprocessing combinations per model would make "grid search" meaningless as a
        baseline.
        """
        specs: list[ExperimentSpec] = []
        for slot in masks.legal_model_slots():
            if slot == STOP_INDEX:
                continue
            for preset in np.flatnonzero(masks.preset[slot]):
                for feature in np.flatnonzero(masks.feature_selection[slot]):
                    bits = masks.preprocessing[slot].astype(int)
                    action = np.concatenate(
                        [
                            np.asarray([slot]),
                            bits,
                            np.asarray([preset, feature]),
                        ]
                    )
                    decoded = self.decode(action, source=ExperimentSource.BASELINE)
                    if decoded is not None:
                        specs.append(decoded)
        return specs


def _normalise_action(action: Any) -> list[int]:
    """Flatten and validate the ``(model, *preprocessing_bits, preset, feature)`` layout."""
    values = [int(value) for value in np.asarray(action).reshape(-1)]
    expected = len(PREPROCESSING_OPTIONS) + 3
    if len(values) != expected:
        raise ValueError(
            f"action has {len(values)} components; expected {expected} "
            f"(model + {len(PREPROCESSING_OPTIONS)} preprocessing bits + preset + "
            f"feature selection)"
        )
    return values


def build_action_space(
    task_type: TaskType,
    enabled_models: list[str] | None = None,
    *,
    min_experiments_before_stop: int = 1,
) -> ActionSpace:
    """Action space over the registry entries that support ``task_type``."""
    from rl_automl.search.model_registry import specs_for_task

    specs = specs_for_task(task_type, enabled_models)
    if not specs:
        raise RegistryError(
            f"no registered model supports {task_type.value}",
            enabled=enabled_models,
        )
    return ActionSpace(
        task_type,
        specs,
        min_experiments_before_stop=min_experiments_before_stop,
    )


__all__ = [
    "ACTION_SPACE_VERSION",
    "STOP_INDEX",
    "ActionMaskTable",
    "ActionSpace",
    "ActionSpaceLayout",
    "build_action_space",
]

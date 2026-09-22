"""Task understanding (spec §3).

Turns a free-text problem statement into a :class:`TaskSpec`. This layer is deliberately
independent of the RL agent, the dataset profiler and the executor: it is pure text
analysis over an optional list of column names.

Design notes
------------
The extractor is **rule-based by default** because the problem statement is untrusted
input and must never be executed (spec §32). It is structured so an LLM extractor can be
dropped in behind ``TaskConfig.llm_extractor`` without changing callers.

The strongest signal available is the column names of the uploaded dataset: matching
"churn" in the sentence against a column literally called ``churn`` is far more reliable
than any single regex, so explicit regex candidates are scored *and* cross-checked
against columns.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from rl_automl.core.logging import get_logger
from rl_automl.core.metrics import (
    METRIC_ALIASES,
    METRICS,
    direction_for,
    normalise_metric_name,
    primary_metric_for,
)
from rl_automl.core.types import (
    ClassDistribution,
    MetricDirection,
    TaskSpec,
    TaskType,
)

logger = get_logger("task.classifier")

#: Weighted phrases per task type. Longer, more specific phrases carry more weight.
TASK_SIGNALS: dict[TaskType, tuple[tuple[str, float], ...]] = {
    TaskType.CLASSIFICATION: (
        ("classif", 1.0),
        ("predict whether", 1.2),
        ("predict if", 1.0),
        ("binary classification", 1.4),
        ("multiclass", 1.2),
        ("multi-class", 1.2),
        ("churn", 1.1),
        ("spam", 1.0),
        ("fraud", 0.8),
        ("will default", 0.9),
        ("default on", 0.8),
        ("whether a customer", 0.9),
        ("whether an", 0.5),
        ("yes/no", 0.9),
        ("yes or no", 0.9),
        ("probability of", 0.5),
        ("categor", 0.6),
        ("label", 0.5),
        ("sentiment", 0.9),
        ("diagnos", 0.9),
        ("detect whether", 0.9),
        ("survive", 0.6),
        ("approve or reject", 0.9),
        ("predict the class", 1.2),
        ("attrition", 0.8),
        ("conversion", 0.6),
        ("will buy", 0.8),
        ("click-through", 0.8),
        ("is fraudulent", 0.9),
    ),
    TaskType.REGRESSION: (
        ("regress", 1.4),
        ("predict the price", 1.3),
        ("predict price", 1.3),
        ("house price", 1.1),
        ("forecast", 1.0),
        ("estimate", 0.9),
        ("how much", 1.0),
        ("how many", 0.8),
        ("continuous value", 1.2),
        ("numeric value", 1.0),
        ("demand", 0.7),
        ("revenue", 0.6),
        ("sales amount", 0.9),
        ("temperature tomorrow", 0.9),
        ("root mean squared", 1.2),
        ("rmse", 1.2),
        ("mean absolute error", 1.2),
        ("r-squared", 1.0),
        ("r2 score", 1.0),
        ("quantity", 0.5),
        ("predicted value", 0.6),
    ),
    TaskType.CLUSTERING: (
        ("cluster", 1.5),
        ("segment", 1.2),
        ("segmentation", 1.3),
        ("group customers", 1.2),
        ("grouping", 1.0),
        ("cohort", 1.0),
        ("k-means", 1.5),
        ("kmeans", 1.5),
        ("partition", 0.9),
        ("unsupervised grouping", 1.3),
        ("customer personas", 0.9),
    ),
    TaskType.ANOMALY_DETECTION: (
        ("anomal", 1.5),
        ("outlier", 1.4),
        ("intrusion", 1.2),
        ("fault detection", 1.2),
        ("novelty detection", 1.4),
        ("abnormal", 1.2),
        ("defect", 0.9),
        ("tamper", 0.9),
        ("suspicious", 0.9),
        ("rare events", 0.8),
        ("fraud detection", 1.1),
    ),
    TaskType.DIMENSIONALITY_REDUCTION: (
        ("dimensionality reduction", 1.6),
        ("reduce dimension", 1.5),
        ("reduce the number of features", 1.4),
        ("feature compression", 1.2),
        ("principal component", 1.4),
        ("pca", 1.3),
        ("latent space", 1.1),
        ("embedding", 0.9),
        ("compress the features", 1.3),
    ),
}

#: Clauses after which tokens describe *inputs*, not the prediction target.
_FEATURE_CLAUSE_MARKERS = (
    "based on",
    "using",
    "given",
    "from the",
    "from historical",
    "with the",
    "according to",
    "as a function of",
    "considering",
)

_TARGET_PATTERNS: tuple[str, ...] = (
    r"predict(?:ing)?\s+whether\s+.{0,60}?\b(?:will|is|are|has|have|can|should|might|would)\s+"
    r"([a-z][a-z0-9_]{2,30})",
    r"predict(?:ing)?\s+(?:the\s+)?(?:value\s+of\s+)?([a-z][a-z0-9_]{2,30})",
    r"predict(?:ing)?\s+(?:the\s+)?(?:amount|number|count|level|rate)\s+of\s+([a-z][a-z0-9_]{2,30})",
    r"estimate\s+(?:the\s+)?([a-z][a-z0-9_]{2,30})",
    r"forecast\s+(?:the\s+)?([a-z][a-z0-9_]{2,30})",
    r"classify\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z0-9_]{2,30})",
    r"target\s*(?:column|variable|feature|label)?\s*(?:is|:|=)\s*([a-z][a-z0-9_]{2,30})",
    r"segment\s+(?:the\s+|our\s+)?([a-z][a-z0-9_]{2,30})",
    r"detect\s+(?:the\s+|any\s+|an?\s+)?([a-z][a-z0-9_]{2,30})",
)

_EXPLICIT_METRIC_PATTERN = re.compile(
    r"(?:maximi[sz]e|optimi[sz]e(?:\s+for)?|minimi[sz]e|prioriti[sz]e|metric(?:\s+is)?|"
    r"measured\s+by|score\s+by)\s+([a-z0-9_\-2]{2,25})"
)

_WORD_RE = re.compile(r"[a-z0-9_]+")


class TaskClassifier:
    """Extracts the ML task from a problem statement."""

    def __init__(self, min_confidence: float = 0.35, default_metric_by_task: dict | None = None):
        self.min_confidence = min_confidence
        self.default_metric_by_task = {
            normalise_metric_name(k): v for k, v in (default_metric_by_task or {}).items()
        }

    # -- public API --------------------------------------------------------------

    def classify(
        self,
        problem_statement: str,
        *,
        columns: Iterable[str] | None = None,
        column_kinds: dict[str, str] | None = None,
        class_distribution: ClassDistribution | None = None,
        target_hint: str | None = None,
        task_hint: TaskType | str | None = None,
        metric_hint: str | None = None,
    ) -> TaskSpec:
        """Produce a :class:`TaskSpec`, with user hints always taking precedence."""
        statement = (problem_statement or "").strip()
        text = _normalise(statement)
        column_list = [str(c) for c in (columns or [])]
        notes: list[str] = []

        task_type, task_confidence, matched = self._infer_task_type(
            text, column_kinds=column_kinds, class_distribution=class_distribution
        )
        if matched:
            notes.append("signals: " + ", ".join(sorted(matched)[:6]))

        target = target_hint or self._infer_target(text, column_list)
        if target_hint:
            notes.append("target provided by user")
        elif target:
            notes.append(f"target inferred from statement/columns: {target}")

        # A target found in the data is decent evidence of supervised learning.
        if target and not task_type.is_supervised:
            task_type = (
                TaskType.CLASSIFICATION
                if self._looks_categorical(target, column_kinds, class_distribution)
                else TaskType.REGRESSION
            )
            notes.append("task type reconciled with an identified target column")

        if task_type.is_supervised and not target:
            notes.append(
                "no target column identified; the caller may need to supply `target` explicitly"
            )

        metric, metric_explicit = self._infer_metric(text, task_type, metric_hint)
        if metric_hint:
            notes.append(f"metric provided by user: {metric}")
        elif metric_explicit:
            notes.append(f"metric read from statement: {metric}")
        else:
            notes.append(f"default metric for {task_type.value}: {metric}")

        direction = direction_for(metric)
        stated_direction = _stated_direction(text, metric)
        if stated_direction is not None and stated_direction is not direction:
            asked = "maximize" if stated_direction is MetricDirection.MAXIMIZE else "minimize"
            notes.append(f"statement asks to {asked} {metric}; honouring that")
            direction = stated_direction

        confidence = self._confidence(task_confidence, target is not None, metric_explicit)
        if confidence < self.min_confidence:
            notes.append(
                f"low confidence ({confidence:.2f}); falling back to {task_type.value} defaults"
            )

        spec = TaskSpec(
            learning_type=task_type.learning_type,
            task_type=task_type,
            objective=self._objective(statement, target, task_type),
            target=target if task_type.is_supervised else None,
            metric=metric,
            metric_direction=direction,
            confidence=round(confidence, 4),
            source="heuristic",
            notes=notes,
            problem_statement=statement,
        )

        # Explicit hints win outright, regardless of heuristic scores.
        if task_hint is not None:
            spec = self.apply_overrides(spec, task_type=task_hint)
        if target_hint is not None:
            spec = self.apply_overrides(spec, target=target_hint)
        if metric_hint is not None:
            spec = self.apply_overrides(spec, metric=metric_hint)

        logger.info(
            "task inferred",
            extra={
                "context": {
                    "task_type": spec.task_type.value,
                    "metric": spec.metric,
                    "target": spec.target,
                    "confidence": spec.confidence,
                }
            },
        )
        return spec

    # -- overrides ---------------------------------------------------------------

    def apply_overrides(
        self,
        spec: TaskSpec,
        *,
        task_type: TaskType | str | None = None,
        target: str | None = None,
        metric: str | None = None,
        objective: str | None = None,
        metric_direction: MetricDirection | str | None = None,
    ) -> TaskSpec:
        """User-supplied values always beat inferred ones (spec §15)."""
        updates: dict[str, Any] = {}

        if task_type is not None:
            resolved = task_type if isinstance(task_type, TaskType) else TaskType(str(task_type))
            updates["task_type"] = resolved
            updates["learning_type"] = resolved.learning_type
            if target is None and not resolved.is_supervised:
                updates["target"] = None

        if target is not None:
            updates["target"] = target

        if objective is not None:
            updates["objective"] = objective

        if metric is not None:
            canonical = normalise_metric_name(metric)
            updates["metric"] = canonical
            updates["metric_direction"] = direction_for(canonical)

        if metric_direction is not None:
            updates["metric_direction"] = (
                metric_direction
                if isinstance(metric_direction, MetricDirection)
                else MetricDirection(str(metric_direction))
            )

        merged = spec.model_copy(update=updates)
        if merged.source == "heuristic" and updates:
            merged.source = "user"
            merged.notes = [*merged.notes, "task specification overridden by the caller"]
        return merged

    # -- task type ---------------------------------------------------------------

    def _infer_task_type(
        self,
        text: str,
        *,
        column_kinds: dict[str, str] | None,
        class_distribution: ClassDistribution | None,
    ) -> tuple[TaskType, float, set[str]]:
        scores: dict[TaskType, float] = {}
        matched: set[str] = set()

        for task, signals in TASK_SIGNALS.items():
            total = 0.0
            for phrase, weight in signals:
                if _contains(text, phrase):
                    total += weight
                    matched.add(phrase)
            scores[task] = total

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_task, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_score <= 0.0:
            fallback = self._infer_from_data(column_kinds, class_distribution)
            return fallback, 0.25, matched

        margin = (best_score - second_score) / best_score if best_score else 0.0
        confidence = min(0.95, 0.45 + 0.5 * margin)
        return best_task, confidence, matched

    @staticmethod
    def _infer_from_data(
        column_kinds: dict[str, str] | None,
        class_distribution: ClassDistribution | None,
    ) -> TaskType:
        """No textual signal: guess from the shape of the data itself."""
        if class_distribution is not None and class_distribution.n_classes:
            # A handful of repeated values is far more likely to be a label.
            if class_distribution.n_classes <= 20:
                return TaskType.CLASSIFICATION
            return TaskType.REGRESSION
        if column_kinds:
            n_categorical = sum(1 for kind in column_kinds.values() if kind == "categorical")
            if n_categorical == 0:
                return TaskType.REGRESSION
        return TaskType.CLASSIFICATION

    # -- target ------------------------------------------------------------------

    def _infer_target(self, text: str, columns: list[str]) -> str | None:
        head = _text_before_feature_clause(text)

        # 1. Columns named in the statement. By far the most reliable signal: matching
        #    "churn" against a column literally called `churn` beats any regex.
        from_columns = self._match_columns(head, columns)
        if from_columns is not None:
            return from_columns

        # 2. Explicit syntactic patterns, accepted only when they name a real column.
        candidates: list[str] = []
        for pattern in _TARGET_PATTERNS:
            match = re.search(pattern, head)
            if not match:
                continue
            candidate = match.group(1).strip("_ ")
            if len(candidate) < 3 or candidate in _STOPWORDS:
                continue
            candidates.append(candidate)

        for candidate in candidates:
            resolved = _resolve_to_column(candidate, columns)
            if resolved is not None:
                return resolved

        # 3. A conventionally-named label column, e.g. `label`, `target`, `y`.
        conventional = _conventional_target(columns)
        if conventional is not None:
            return conventional

        # 4. With no column list there is nothing to validate against, so the syntactic
        #    candidate is the only signal available. With one, an unmatched candidate is
        #    not a target: the caller's frame has no such column, and returning it would
        #    both mislabel an unsupervised problem as supervised and hand the executor a
        #    target it cannot find.
        if not columns:
            return candidates[0] if candidates else None
        return None

    @staticmethod
    def _match_columns(head: str, columns: list[str]) -> str | None:
        if not columns or not head:
            return None
        tokens = set(_WORD_RE.findall(head))
        best: tuple[float, str] | None = None

        for column in columns:
            score = _column_match_score(column, head, tokens)
            if score > 0.0 and (best is None or score > best[0]):
                best = (score, column)

        if best is None or best[0] < 0.8:
            return None
        return best[1]

    @staticmethod
    def _looks_categorical(
        target: str,
        column_kinds: dict[str, str] | None,
        class_distribution: ClassDistribution | None,
    ) -> bool:
        if column_kinds:
            for name, kind in column_kinds.items():
                if _normalise_identifier(name) == _normalise_identifier(target):
                    return kind == "categorical"
        if class_distribution is not None and class_distribution.n_classes <= 20:
            return True
        return True  # default for a named target with no other evidence

    # -- metric ------------------------------------------------------------------

    def _infer_metric(
        self, text: str, task_type: TaskType, metric_hint: str | None
    ) -> tuple[str, bool]:
        if metric_hint:
            canonical = normalise_metric_name(metric_hint)
            if canonical in METRICS:
                return canonical, True

        configured = self.default_metric_by_task.get(task_type.value)
        if configured:
            return configured, False

        # Explicit "maximize f1" style instructions.
        for match in _EXPLICIT_METRIC_PATTERN.finditer(text):
            canonical = normalise_metric_name(match.group(1))
            spec = METRICS.get(canonical)
            if spec is not None and spec.applicable_to(task_type):
                return canonical, True

        # Any metric mentioned anywhere, preferring the longest match.
        candidates: list[tuple[int, str]] = []
        for name in (*METRICS, *METRIC_ALIASES):
            if _contains(text, name.replace("_", " ")) or _contains(text, name):
                canonical = normalise_metric_name(name)
                spec = METRICS.get(canonical)
                if spec is not None and spec.applicable_to(task_type):
                    candidates.append((len(name), canonical))
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1], True

        return primary_metric_for(task_type), False

    # -- misc --------------------------------------------------------------------

    @staticmethod
    def _confidence(task_confidence: float, has_target: bool, metric_explicit: bool) -> float:
        value = task_confidence
        if has_target:
            value += 0.15
        if metric_explicit:
            value += 0.05
        return max(0.05, min(0.98, value))

    @staticmethod
    def _objective(statement: str, target: str | None, task_type: TaskType) -> str:
        if target:
            pretty = target.replace("_", " ").strip()
            suffix = {
                TaskType.CLASSIFICATION: "prediction",
                TaskType.REGRESSION: "estimation",
                TaskType.CLUSTERING: "segmentation",
                TaskType.ANOMALY_DETECTION: "detection",
                TaskType.DIMENSIONALITY_REDUCTION: "reduction",
            }[task_type]
            return f"{pretty} {suffix}"
        if statement:
            first = re.split(r"[.!?\n]", statement, maxsplit=1)[0].strip()
            return first[:200] if first else statement[:200]
        return f"{task_type.value} task"


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "them",
        "data",
        "dataset",
        "model",
        "value",
        "values",
        "target",
        "class",
        "label",
        "output",
        "whether",
        "customer",
        "customers",
        "user",
        "users",
        "patient",
        "patients",
    }
)

#: Column names conventionally used for the prediction target.
CONVENTIONAL_TARGET_NAMES = (
    "target",
    "label",
    "labels",
    "class",
    "y",
    "outcome",
    "response",
    "result",
    "y_true",
    "ground_truth",
    "observation",
)

#: Columns that identify a row rather than describe it. Never a sensible target.
_IDENTIFIER_PATTERN = re.compile(r"(^|_)(id|uuid|guid|key|index|idx|no|num|number)$|_id$")


def _spaced_identifier(name: str) -> str:
    """``median_house_value`` -> ``median house value``."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    spaced = re.sub(r"[^A-Za-z0-9]+", " ", spaced)
    return re.sub(r"\s+", " ", spaced).strip().lower()


def _is_identifier_like(column: str) -> bool:
    return _IDENTIFIER_PATTERN.search(str(column).lower()) is not None


def _word_present(word: str, tokens: set[str]) -> bool:
    """Token-level match that tolerates inflection ('fraud' in 'fraudulent')."""
    if word in tokens:
        return True
    if len(word) < 4:
        return False
    return any(
        len(token) >= 4 and (token.startswith(word) or word.startswith(token)) for token in tokens
    )


def _column_match_score(column: str, head: str, tokens: set[str]) -> float:
    """How strongly a column name is referenced by the statement."""
    normalised = _normalise_identifier(column)
    if not normalised:
        return 0.0

    if normalised in tokens:
        score = 1.0
    elif len(normalised) >= 3 and _contains(head, normalised):
        # Word-boundary match, not a substring test: a one-character column called `a`
        # would otherwise "appear" inside any word containing that letter ("PCA",
        # "data"), and short names are exactly where substring matching goes wrong.
        score = 0.9
    else:
        words = [w for w in _WORD_RE.findall(_spaced_identifier(column)) if len(w) >= 3]
        if not words:
            return 0.0
        present = sum(1 for word in words if _word_present(word, tokens))
        if present == 0:
            return 0.0
        coverage = present / len(words)
        score = 0.95 * coverage if coverage >= 0.99 else 0.8 * coverage

    # An identifier column mentioning the entity ("customer" -> customer_id) is noise.
    return score * 0.2 if _is_identifier_like(column) else score


def _resolve_to_column(candidate: str, columns: list[str]) -> str | None:
    """Strictly resolve a candidate to a real column, else ``None``."""
    if not columns:
        return None
    norm = _normalise_identifier(candidate)
    for column in columns:
        if _normalise_identifier(column) == norm:
            return column
    best: tuple[float, str] | None = None
    for column in columns:
        ratio = SequenceMatcher(None, norm, _normalise_identifier(column)).ratio()
        if ratio >= 0.82 and (best is None or ratio > best[0]):
            best = (ratio, column)
    return best[1] if best is not None else None


def _conventional_target(columns: list[str]) -> str | None:
    if not columns:
        return None
    normalised = {_normalise_identifier(column): column for column in columns}
    for candidate in CONVENTIONAL_TARGET_NAMES:
        key = _normalise_identifier(candidate)
        if key in normalised:
            return normalised[key]
    return None


def _normalise(text: str) -> str:
    lowered = (text or "").lower()
    lowered = lowered.replace("'", "'").replace(""", '"').replace(""", '"')
    lowered = re.sub(r"[^a-z0-9_'\-2\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _normalise_identifier(name: str) -> str:
    """``CamelCase`` / ``snake_case`` / ``Title Case`` all collapse to single tokens."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    spaced = re.sub(r"[^A-Za-z0-9]+", " ", spaced)
    return re.sub(r"\s+", "", spaced).lower()


#: Shortest single-token phrase that is matched as a stem rather than a whole word.
#: Below this, prefix matching is too blunt: a two-letter metric like ``f1`` or ``r2``
#: would start matching unrelated words.
_MIN_STEM_LENGTH = 4


def _contains(text: str, phrase: str) -> bool:
    """Phrase presence, tolerating inflection for longer single tokens.

    Signal tables are written as stems on purpose -- ``classif`` covers
    classify/classification, ``anomal`` covers anomaly/anomalous, ``categor`` covers
    categorical/categorisation. Matching them on a word boundary would silently defeat
    that, because ``anomal`` is not a word in ``anomalous``. A single token of at least
    :data:`_MIN_STEM_LENGTH` characters therefore matches at a word *start*; shorter
    tokens and multi-word phrases keep the strict test.
    """
    if not phrase:
        return False
    if re.fullmatch(r"[a-z0-9_]+", phrase):
        if len(phrase) >= _MIN_STEM_LENGTH:
            return re.search(rf"\b{re.escape(phrase)}", text) is not None
        return re.search(rf"\b{re.escape(phrase)}\b", text) is not None
    return phrase in text


def _text_before_feature_clause(text: str) -> str:
    """Drop "based on X" / "using X" tails, where nouns are inputs rather than targets."""
    cut = len(text)
    for marker in _FEATURE_CLAUSE_MARKERS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


def _stated_direction(text: str, metric: str) -> MetricDirection | None:
    """Detect an explicit "minimize <metric>" instruction that contradicts the default."""
    for match in _EXPLICIT_METRIC_PATTERN.finditer(text):
        if normalise_metric_name(match.group(1)) != normalise_metric_name(metric):
            continue
        prefix = text[max(0, match.start() - 20) : match.start()]
        if "minimi" in prefix:
            return MetricDirection.MINIMIZE
        if "maximi" in prefix:
            return MetricDirection.MAXIMIZE
    return None


__all__ = ["TASK_SIGNALS", "TaskClassifier"]

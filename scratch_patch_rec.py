"""Apply targeted edits to recommendation.py (temporary scratch script)."""

from __future__ import annotations

import pathlib

PATH = pathlib.Path("rl_automl/search/recommendation.py")
text = PATH.read_text(encoding="utf-8")

EDITS: list[tuple[str, str]] = [
    (
        """        ranked = self._rank(list(candidates.values()))
        selected = ranked[: self._recommended_count(ranked)]""",
        """        ranked = self._rank(list(candidates.values()))
        selected = self._select_diverse(ranked, self._recommended_count(ranked))""",
    ),
    (
        """            preprocessing=spec.preprocessing,
            feature_selection=spec.feature_selection,
            n_rows=n_rows,
            n_cols=n_cols,
        )""",
        """            preprocessing=spec.preprocessing,
            feature_selection=spec.feature_selection,
            preset_params=spec.hyperparameters,
            n_rows=n_rows,
            n_cols=n_cols,
        )""",
    ),
    (
        """    def _recommended_count(self, ranked: list[PlannedExperiment]) -> int:""",
        '''    def _select_diverse(
        self, ranked: list[PlannedExperiment], k: int
    ) -> list[PlannedExperiment]:
        """Pick ``k`` candidates, preferring distinct algorithms.

        Recommending the same model three times with different preprocessing gives the user
        nothing to choose between. The first pass takes the best candidate from each
        distinct model, which is what makes a recommendation list look like the spec's
        example (LightGBM, XGBoost, Extra Trees); a second pass fills any remaining slots
        with the next best candidates regardless of model, so a registry with fewer than
        ``k`` usable entries still produces a full plan.
        """
        if k <= 0:
            return []

        selected: list[PlannedExperiment] = []
        chosen_signatures: set[str] = set()
        seen_models: set[str] = set()

        for candidate in ranked:
            if len(selected) >= k:
                break
            if candidate.experiment.model in seen_models:
                continue
            selected.append(candidate)
            chosen_signatures.add(candidate.experiment.signature())
            seen_models.add(candidate.experiment.model)

        for candidate in ranked:
            if len(selected) >= k:
                break
            signature = candidate.experiment.signature()
            if signature in chosen_signatures:
                continue
            selected.append(candidate)
            chosen_signatures.add(signature)

        return selected

    def _recommended_count(self, ranked: list[PlannedExperiment]) -> int:''',
    ),
]

for index, (old, new) in enumerate(EDITS, 1):
    occurrences = text.count(old)
    if occurrences != 1:
        raise SystemExit(f"edit {index}: expected 1 occurrence, found {occurrences}")
    text = text.replace(old, new)
    print(f"edit {index}: applied")

PATH.write_text(text, encoding="utf-8")
print("recommendation.py updated")

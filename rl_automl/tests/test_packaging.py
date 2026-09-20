from __future__ import annotations

import json
import os
import subprocess
import sys

from rl_automl.core.types import ExperimentSpec
from rl_automl.dataset.profiler import profile_dataset
from rl_automl.execution.executor import PipelineExecutor
from rl_automl.packaging.inference_generator import InferenceGenerator
from rl_automl.packaging.model_exporter import ModelExporter


def _build_package(tmp_path, config, frame, task, run_id="package"):
    executor = PipelineExecutor(task, config, run_id=run_id)
    executor.prepare(frame)
    experiment = ExperimentSpec(
        model="logistic_regression",
        preset_index=0,
        preprocessing=["impute_numeric", "scale_standard", "encode_categorical"],
    )
    result = executor.execute(experiment)
    assert result.succeeded
    trained = executor.get_trained(experiment.experiment_id)
    profile = profile_dataset(frame, task.target, config.dataset)
    package = ModelExporter(config).export(
        trained=trained,
        result=result,
        task=task,
        profile=profile,
        destination=tmp_path,
        run_id=run_id,
        dataset_name="tiny.csv",
        dataset_sha256="abc",
        column_kinds=executor.column_kinds,
        label_classes=executor.class_labels,
        label_encoding_required=executor.label_encoding_required,
        split_summary=executor.splits.summary(),
        include_onnx=False,
    )
    InferenceGenerator(config).generate(package, split_summary=executor.splits.summary())
    return package, executor, result


def test_packaging_label_mapping_only_when_encoding_is_required(
    tmp_path, config, binary_frame, classification_task
):
    package, executor, _ = _build_package(
        tmp_path / "strings",
        config,
        binary_frame.drop(columns="when"),
        classification_task,
        "strings",
    )
    metadata = json.loads((package.directory / "metadata/model_metadata.json").read_text())
    assert executor.label_encoding_required
    assert metadata["label_mapping"] == {"0": "no", "1": "yes"}

    integer_frame = binary_frame.drop(columns="when").copy()
    integer_frame["outcome"] = (integer_frame["outcome"] == "yes").astype(int)
    package2, executor2, _ = _build_package(
        tmp_path / "integers", config, integer_frame, classification_task, "integers"
    )
    metadata2 = json.loads((package2.directory / "metadata/model_metadata.json").read_text())
    assert not executor2.label_encoding_required
    assert "label_mapping" not in metadata2


def test_package_clean_environment_round_trip(tmp_path, config, binary_frame, classification_task):
    frame = binary_frame.drop(columns="when")
    package, executor, result = _build_package(tmp_path, config, frame, classification_task)
    probe = frame.drop(columns="outcome").head(6)
    probe.to_csv(package.directory / "probe.csv", index=False)
    expected = (
        executor.decode_labels(
            executor.get_trained(result.experiment.experiment_id).model.predict(
                executor.get_trained(result.experiment.experiment_id).preprocessor.transform(probe)
            )
        )
        .astype(str)
        .tolist()
    )
    script = (
        "import json,sys; "
        "sys.path.insert(0, 'inference'); "
        "from model_loader import load_model; "
        "m=load_model('.'); print(json.dumps(m.predict('probe.csv').astype(str).tolist()))"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package.directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == expected
    assert (package.directory / "README.md").is_file()
    assert (package.directory / "requirements.txt").is_file()

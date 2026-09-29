"""SageMaker pipeline definition for one manual training run.

The rendered steps are Validate, Process, Train, Evaluate, Quality Gate, and
Register. Register sits on the accept branch of the quality gate, so a
rejected candidate skips it. Dataset version, resolved configuration, and git
SHA are pipeline parameters.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

from forecastops_ml.evaluation.folds import MIN_PUBLISHED_FOLDS, PREFERRED_BENCHMARK_FOLDS
from forecastops_ml.pipelines.metadata import (
    artifact_locations,
    require_dataset_version,
    require_git_sha,
)
from forecastops_ml.pipelines.steps import (
    ENTRYPOINTS,
    GATE_PASS_STATUS,
    GATE_PROPERTY,
    REGISTER_APPROVAL,
)
from forecastops_ml.training.deepar_export import (
    DEFAULT_GRAIN,
    DEFAULT_PREDICTION_LENGTH,
    DeepARChannels,
    DeepARRecord,
)
from forecastops_ml.training.deepar_job import (
    CPU_INSTANCE_TYPE,
    DEEPAR_TRAINING_IMAGE,
    TrainingGates,
    build_training_job_request,
    deepar_training_image,
    require_cpu_instance,
)

STEP_NAMES = (
    "Validate",
    "Process",
    "Train",
    "Evaluate",
    "Quality Gate",
    "Register",
)
_SAGEMAKER_NAMES = {"Quality Gate": "QualityGate"}
_DISPLAY_NAMES = {"QualityGate": "Quality Gate"}
_PIPELINE_VERSION = "2020-12-01"
_PARAMETER_LIMIT = 1024
# Public SageMaker scikit-learn processing image for us-east-1.
PROCESSING_IMAGE = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
)
_PROCESSING_INSTANCE = "ml.m5.large"
_EVALUATION_METRICS = "mae,rmse,wape,smape,bias,pinball_loss"


@dataclass(frozen=True)
class TrainingPipeline:
    """Inputs that render one pipeline definition."""

    dataset_version: str
    git_sha: str
    configuration: Mapping[str, object]
    bucket: str
    pipeline_role_arn: str
    training_role_arn: str
    region: str
    gates: TrainingGates


@dataclass(frozen=True)
class RenderedStep:
    """One logical step in the rendered definition."""

    name: str
    sagemaker_name: str
    kind: str
    entrypoint: str


@dataclass(frozen=True)
class RenderedPipeline:
    """A pipeline definition and the parameters that start one execution."""

    steps: tuple[RenderedStep, ...]
    document: dict[str, Any]
    execution_parameters: dict[str, str]
    configuration: dict[str, object]
    dataset_version: str
    git_sha: str

    @property
    def step_names(self) -> tuple[str, ...]:
        """Return the logical step names in execution order."""

        return tuple(step.name for step in self.steps)


def resolve_configuration(
    raw: Mapping[str, object],
    gates: TrainingGates,
) -> dict[str, object]:
    """Return the configuration a run actually uses.

    The instance type, runtime ceiling, and search switch come from the
    training limits. A request for a GPU or a hyperparameter search is rejected.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("Configuration must be a JSON object.")
    if raw.get("hyperparameter_search") is True:
        raise ValueError("The training pipeline does not run a hyperparameter search.")
    if raw.get("allow_gpu_training") is True:
        raise ValueError("GPU training is disabled.")
    requested = raw.get("instance_type", CPU_INSTANCE_TYPE)
    if not isinstance(requested, str):
        raise ValueError("instance_type must be a string.")
    require_cpu_instance(requested)
    if requested != CPU_INSTANCE_TYPE:
        raise ValueError(f"The training pipeline uses {CPU_INSTANCE_TYPE}.")
    grain = raw.get("grain", DEFAULT_GRAIN)
    if grain not in {"day", "week"}:
        raise ValueError("grain must be day or week.")
    frequency = "1W" if grain == "week" else "1D"
    supplied_frequency = raw.get("frequency", frequency)
    if supplied_frequency != frequency:
        raise ValueError(f"frequency for grain {grain} must be {frequency}.")
    resolved: dict[str, object] = dict(raw)
    resolved["allow_gpu_training"] = False
    resolved["category_cardinality"] = _config_int(
        raw.get("category_cardinality"),
        1,
        "category_cardinality",
    )
    resolved["frequency"] = frequency
    resolved["grain"] = grain
    resolved["hyperparameter_search"] = False
    resolved["instance_count"] = 1
    resolved["instance_type"] = CPU_INSTANCE_TYPE
    resolved["max_training_jobs_per_day"] = gates.max_training_jobs_per_day
    resolved["max_training_runtime_minutes"] = gates.max_training_runtime_minutes
    resolved["model_family"] = "deepar"
    resolved["prediction_length"] = _config_int(
        raw.get("prediction_length"),
        DEFAULT_PREDICTION_LENGTH,
        "prediction_length",
    )
    resolved["series_count"] = _config_int(raw.get("series_count"), 32, "series_count")
    resolved["store_cardinality"] = _config_int(
        raw.get("store_cardinality"),
        1,
        "store_cardinality",
    )
    return _json_object(resolved)


def render_pipeline(pipeline: TrainingPipeline) -> RenderedPipeline:
    """Render the pipeline definition and its execution parameters."""

    dataset_version = require_dataset_version(pipeline.dataset_version)
    git_sha = require_git_sha(pipeline.git_sha)
    configuration = resolve_configuration(pipeline.configuration, pipeline.gates)
    bucket = _require_bucket(pipeline.bucket)
    pipeline_role = _require_role(pipeline.pipeline_role_arn, "Pipeline role")
    training_role = _require_role(pipeline.training_role_arn, "Training role")
    deepar_training_image(pipeline.region)
    require_cpu_instance(_PROCESSING_INSTANCE)
    parameters = _execution_parameters(dataset_version, git_sha, configuration)
    document = _document(
        bucket=bucket,
        pipeline_role_arn=pipeline_role,
        training_role_arn=training_role,
        region=pipeline.region,
        gates=pipeline.gates,
        configuration=configuration,
        parameters=parameters,
    )
    if executed_step_names(document, gate_passed=True) != STEP_NAMES:
        raise ValueError(
            "Pipeline steps must be Validate, Process, Train, Evaluate, Quality Gate, Register."
        )
    if executed_step_names(document, gate_passed=False) != STEP_NAMES[:-1]:
        raise ValueError("Register must be skipped when the quality gate rejects the candidate.")
    steps = _logical_steps()
    return RenderedPipeline(
        steps=steps,
        document=document,
        execution_parameters=parameters,
        configuration=configuration,
        dataset_version=dataset_version,
        git_sha=git_sha,
    )


def executed_step_names(document: Mapping[str, Any], *, gate_passed: bool) -> tuple[str, ...]:
    """Return the steps that run when the quality gate accepts or rejects.

    Register is included only when ``gate_passed`` is true.
    """

    raw_steps = document.get("Steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Pipeline definition needs a Steps list.")
    names: list[str] = []
    for step in raw_steps:
        names.append(_display_name(_step_name(step)))
        if _step_name(step) != "QualityGate":
            continue
        arguments = step.get("Arguments")
        if not isinstance(arguments, dict):
            raise ValueError("Quality Gate arguments must be an object.")
        branch = "IfSteps" if gate_passed else "ElseSteps"
        nested = arguments.get(branch)
        if not isinstance(nested, list):
            raise ValueError(f"Quality Gate {branch} must be a list.")
        names.extend(_display_name(_step_name(item)) for item in nested)
    return tuple(names)


def _logical_steps() -> tuple[RenderedStep, ...]:
    kinds = {
        "Validate": "Processing",
        "Process": "Processing",
        "Train": "Training",
        "Evaluate": "Processing",
        "Quality Gate": "Condition",
        "Register": "RegisterModel",
    }
    return tuple(
        RenderedStep(
            name=name,
            sagemaker_name=_SAGEMAKER_NAMES.get(name, name),
            kind=kinds[name],
            entrypoint=ENTRYPOINTS[name],
        )
        for name in STEP_NAMES
    )


def _document(
    *,
    bucket: str,
    pipeline_role_arn: str,
    training_role_arn: str,
    region: str,
    gates: TrainingGates,
    configuration: Mapping[str, object],
    parameters: Mapping[str, str],
) -> dict[str, Any]:
    validate = _processing_step(
        name="Validate",
        slug="validate",
        role_arn=pipeline_role_arn,
        gates=gates,
        inputs=[_input("dataset", _versioned(bucket, "raw"))],
        output_name="quality",
        output_uri=_versioned(bucket, "processed", "quality"),
        environment=_environment(ENTRYPOINTS["Validate"]),
        property_file=False,
    )
    process = _processing_step(
        name="Process",
        slug="process",
        role_arn=pipeline_role_arn,
        gates=gates,
        inputs=[
            _input("dataset", _versioned(bucket, "raw")),
            _input("quality", _output_uri("Validate", "quality")),
        ],
        output_name="channels",
        output_uri=_versioned(bucket, "processed", "channels"),
        environment=_environment(ENTRYPOINTS["Process"]),
        property_file=False,
    )
    evaluate = _processing_step(
        name="Evaluate",
        slug="evaluate",
        role_arn=pipeline_role_arn,
        gates=gates,
        inputs=[
            _input("model", {"Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"}),
            _input("channels", _output_uri("Process", "channels")),
        ],
        output_name="evaluation",
        output_uri=_versioned(bucket, "evaluations"),
        environment=_evaluate_environment(),
        property_file=True,
    )
    train = _training_step(
        training_role_arn=training_role_arn,
        gates=gates,
        configuration=configuration,
        bucket=bucket,
        region=region,
    )
    quality_gate = _quality_gate(train)
    raw = {
        "Version": _PIPELINE_VERSION,
        "Parameters": [
            {"Name": name, "Type": "String", "DefaultValue": value}
            for name, value in parameters.items()
        ],
        "Steps": [validate, process, train, evaluate, quality_gate],
    }
    encoded = json.dumps(raw)
    parsed = json.loads(encoded)
    if not isinstance(parsed, dict):
        raise ValueError("Pipeline definition must be a JSON object.")
    return cast(dict[str, Any], parsed)


def _training_step(
    *,
    training_role_arn: str,
    gates: TrainingGates,
    configuration: Mapping[str, object],
    bucket: str,
    region: str,
) -> dict[str, object]:
    request = build_training_job_request(
        _template_channels(configuration),
        gates=gates,
        bucket=bucket,
        role_arn=training_role_arn,
        region=region,
        job_name="forecastops-training",
    )
    arguments = dict(request)
    arguments.pop("TrainingJobName", None)
    arguments["RoleArn"] = training_role_arn
    arguments["InputDataConfig"] = [
        _train_channel("train"),
        _train_channel("test"),
    ]
    arguments["OutputDataConfig"] = {"S3OutputPath": _versioned(bucket, "models")}
    arguments["AlgorithmSpecification"] = {
        "TrainingImage": DEEPAR_TRAINING_IMAGE,
        "TrainingInputMode": "File",
    }
    return {"Name": "Train", "Type": "Training", "Arguments": arguments}


def _quality_gate(train: Mapping[str, object]) -> dict[str, object]:
    register = {
        "Name": "Register",
        "Type": "RegisterModel",
        "Arguments": {
            "ModelPackageGroupName": "forecastops-deepar",
            "ModelApprovalStatus": REGISTER_APPROVAL,
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": DEEPAR_TRAINING_IMAGE,
                        "ModelDataUrl": {"Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"},
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
            },
        },
    }
    if train.get("Name") != "Train":
        raise ValueError("Register follows the Train step.")
    return {
        "Name": "QualityGate",
        "Type": "Condition",
        "Arguments": {
            "Conditions": [
                {
                    "Type": "Equals",
                    "LeftValue": {
                        "Std:JsonGet": {
                            "PropertyFile": {"Get": "Steps.Evaluate.PropertyFiles.evaluation"},
                            "Path": GATE_PROPERTY,
                        }
                    },
                    "RightValue": GATE_PASS_STATUS,
                }
            ],
            "IfSteps": [register],
            "ElseSteps": [],
        },
    }


def _processing_step(
    *,
    name: str,
    slug: str,
    role_arn: str,
    gates: TrainingGates,
    inputs: list[dict[str, object]],
    output_name: str,
    output_uri: dict[str, object],
    environment: dict[str, object],
    property_file: bool,
) -> dict[str, object]:
    arguments: dict[str, object] = {
        "AppSpecification": {
            "ImageUri": PROCESSING_IMAGE,
            "ContainerEntrypoint": ["python", "-m", "forecastops_ml.pipelines.steps"],
            "ContainerArguments": [slug],
        },
        "Environment": environment,
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {"Outputs": [_output(output_name, output_uri)]},
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": _PROCESSING_INSTANCE,
                "VolumeSizeInGB": 10,
            }
        },
        "RoleArn": role_arn,
        "StoppingCondition": {
            "MaxRuntimeInSeconds": gates.max_training_runtime_minutes * 60,
        },
    }
    if property_file:
        arguments["PropertyFiles"] = [
            {
                "PropertyFileName": "evaluation",
                "OutputName": output_name,
                "FilePath": "metrics.json",
            }
        ]
    return {"Name": name, "Type": "Processing", "Arguments": arguments}


def _evaluate_environment() -> dict[str, object]:
    environment = _environment(ENTRYPOINTS["Evaluate"])
    environment["FORECASTOPS_FOLDS"] = "forecastops_ml.evaluation.folds.build_folds"
    environment["FORECASTOPS_MIN_PUBLISHED_FOLDS"] = str(MIN_PUBLISHED_FOLDS)
    environment["FORECASTOPS_BENCHMARK_FOLDS"] = str(PREFERRED_BENCHMARK_FOLDS)
    environment["FORECASTOPS_METRICS"] = _EVALUATION_METRICS
    environment["FORECASTOPS_GATE"] = ENTRYPOINTS["Quality Gate"]
    return environment


def _environment(entrypoint: str) -> dict[str, object]:
    return {
        "CONFIGURATION": {"Get": "Parameters.Configuration"},
        "DATASET_VERSION": {"Get": "Parameters.DatasetVersion"},
        "FORECASTOPS_ENTRYPOINT": entrypoint,
        "GIT_SHA": {"Get": "Parameters.GitSha"},
    }


def _execution_parameters(
    dataset_version: str,
    git_sha: str,
    configuration: Mapping[str, object],
) -> dict[str, str]:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    if len(encoded) > _PARAMETER_LIMIT:
        raise ValueError(
            f"The resolved configuration must be at most {_PARAMETER_LIMIT} characters."
        )
    return {
        "DatasetVersion": dataset_version,
        "GitSha": git_sha,
        "Configuration": encoded,
    }


def _template_channels(configuration: Mapping[str, object]) -> DeepARChannels:
    grain = configuration["grain"]
    if not isinstance(grain, str) or grain not in {"day", "week"}:
        raise ValueError("grain must be day or week.")
    prediction_length = _required_int(configuration, "prediction_length")
    series_count = min(_required_int(configuration, "series_count"), 32)
    frequency = "1W" if grain == "week" else "1D"
    target = tuple(float(index + 1) for index in range(prediction_length + 2))
    zeros = tuple(0.0 for _ in target)
    record = DeepARRecord(
        series_id="store-01|sku-1",
        start=date(2026, 1, 5),
        target=target,
        cat=(0, 0),
        dynamic_feat=(zeros, zeros),
        target_end=date(2026, 1, 5),
    )
    return DeepARChannels(
        train=tuple(record for _ in range(series_count)),
        test=(record,),
        grain=grain,
        frequency=frequency,
        prediction_length=prediction_length,
        store_cardinality=_required_int(configuration, "store_cardinality"),
        category_cardinality=_required_int(configuration, "category_cardinality"),
    )


def _train_channel(name: str) -> dict[str, object]:
    channels = "Steps.Process.ProcessingOutputConfig.Outputs['channels'].S3Output.S3Uri"
    return {
        "ChannelName": name,
        "ContentType": "application/jsonlines",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": {
                    "Std:Join": {
                        "On": "",
                        "Values": [
                            {"Get": channels},
                            f"{name}/",
                        ],
                    }
                },
                "S3DataDistributionType": "FullyReplicated",
            }
        },
    }


def _versioned(bucket: str, head: str, tail: str = "") -> dict[str, object]:
    suffix = f"/{tail}/" if tail else "/"
    return {
        "Std:Join": {
            "On": "",
            "Values": [
                f"s3://{bucket}/{head}/",
                {"Get": "Parameters.DatasetVersion"},
                suffix,
            ],
        }
    }


def _output_uri(step: str, name: str) -> dict[str, object]:
    return {"Get": f"Steps.{step}.ProcessingOutputConfig.Outputs['{name}'].S3Output.S3Uri"}


def _input(name: str, uri: dict[str, object]) -> dict[str, object]:
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": uri,
            "LocalPath": f"/opt/ml/processing/input/{name}",
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
        },
    }


def _output(name: str, uri: dict[str, object]) -> dict[str, object]:
    return {
        "OutputName": name,
        "S3Output": {
            "S3Uri": uri,
            "LocalPath": f"/opt/ml/processing/output/{name}",
            "S3UploadMode": "EndOfJob",
        },
    }


def _json_object(value: Mapping[str, object]) -> dict[str, object]:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise ValueError("Configuration must be JSON serializable.") from exc
    parsed = json.loads(encoded)
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise ValueError("Configuration must be a JSON object.")
    return cast(dict[str, object], parsed)


def _config_int(value: object, default: int, label: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _required_int(configuration: Mapping[str, object], key: str) -> int:
    value = configuration[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _require_bucket(bucket: str) -> str:
    if not isinstance(bucket, str) or bucket.strip() == "" or "/" in bucket:
        raise ValueError("Artifacts bucket must be a non-empty bucket name.")
    return bucket.strip()


def _require_role(role_arn: str, label: str) -> str:
    if not isinstance(role_arn, str) or not role_arn.startswith("arn:aws:iam::"):
        raise ValueError(f"{label} must be an IAM role ARN.")
    return role_arn


def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


def _step_name(step: object) -> str:
    if not isinstance(step, dict):
        raise ValueError("Each pipeline step must be an object.")
    name = step.get("Name")
    if not isinstance(name, str) or name == "":
        raise ValueError("Each pipeline step needs a name.")
    return name


def planned_locations(bucket: str, dataset_version: str) -> dict[str, str]:
    """Return the artifact locations stored on the training run."""

    return artifact_locations(bucket, dataset_version)

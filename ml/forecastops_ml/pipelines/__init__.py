"""Training pipeline definition and manual execution.

The steps are Validate, Process, Train, Evaluate, Quality Gate, and Register.
Register is skipped when the quality gate rejects the candidate. This package
does not register a schedule. A person starts one execution with the manual
command.
"""

from forecastops_ml.pipelines.definition import (
    STEP_NAMES,
    RenderedPipeline,
    TrainingPipeline,
    executed_step_names,
    render_pipeline,
)
from forecastops_ml.pipelines.execution import (
    PIPELINE_NAME,
    PipelineSubmission,
    pipeline_block_reason,
    start_pipeline,
)
from forecastops_ml.pipelines.metadata import (
    MemoryTrainingRunStore,
    TrainingRunRecord,
    store_finished_execution,
)

__all__ = [
    "PIPELINE_NAME",
    "STEP_NAMES",
    "MemoryTrainingRunStore",
    "PipelineSubmission",
    "RenderedPipeline",
    "TrainingPipeline",
    "TrainingRunRecord",
    "executed_step_names",
    "pipeline_block_reason",
    "render_pipeline",
    "start_pipeline",
    "store_finished_execution",
]

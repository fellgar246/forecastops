"""Request and response models for the local forecast API."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

ModelFamily = Literal["naive", "seasonal_naive", "holt_winters", "gradient_boosting", "deepar"]
DatasetSource = Literal["synthetic", "upload"]
DatasetStatus = Literal["registered", "valid", "invalid"]
Granularity = Literal["day", "week"]
ForecastStatus = Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]
TrainingStatus = Literal[
    "QUEUED",
    "PREPROCESSING",
    "TRAINING",
    "EVALUATING",
    "REGISTERING",
    "COMPLETED",
    "FAILED",
]
ModelStatusName = Literal[
    "TRAINED",
    "EVALUATED",
    "PENDING_APPROVAL",
    "APPROVED",
    "PRODUCTION",
    "REJECTED",
]


class AcceptedJob(BaseModel):
    """Acknowledgement for work the client polls later."""

    job_id: str


class DatasetCreate(BaseModel):
    """Register an existing dataset directory."""

    name: str = Field(min_length=1)
    source: DatasetSource
    uri: str = Field(min_length=1)


class DatasetUploadResponse(BaseModel):
    """Location of a dataset object stored by the API."""

    uri: str
    name: str


class PresignedUploadRequest(BaseModel):
    """File name the client will upload with a pre-signed URL."""

    name: str = Field(min_length=1)


class PresignedUploadResponse(BaseModel):
    """Short-lived URL for uploading one dataset object."""

    url: str
    uri: str
    expires_in: int


class DatasetValidate(BaseModel):
    """Optional as-of date for the staleness check."""

    as_of: date | None = None


class DatasetResponse(BaseModel):
    """One dataset snapshot."""

    id: str
    name: str
    version: str
    source: DatasetSource
    status: DatasetStatus
    uri: str
    schema_version: str
    row_count: int
    date_min: date | None
    date_max: date | None
    quality_report: dict[str, object] | None
    created_at: datetime


class DatasetList(BaseModel):
    """Datasets in registration order."""

    items: list[DatasetResponse]


class TrainingCreate(BaseModel):
    """Start a local training run."""

    dataset_id: str
    model_family: ModelFamily
    configuration: dict[str, object] = Field(default_factory=dict)


class TrainingResponse(BaseModel):
    """One training run."""

    id: str
    dataset_id: str
    dataset_version: str
    model_family: str
    configuration: dict[str, object]
    status: TrainingStatus
    started_at: datetime | None
    finished_at: datetime | None
    artifact_uri: str
    metrics: dict[str, object] | None
    error_message: str | None
    git_sha: str
    pipeline_execution_arn: str
    created_at: datetime


class TrainingList(BaseModel):
    """Training runs in creation order."""

    items: list[TrainingResponse]


class GateChecksBody(BaseModel):
    """Quality-gate clauses stored on a promotion decision."""

    wape_improved: bool
    bias_within_limit: bool
    coverage_within_limit: bool | None
    no_critical_segment_regression: bool


class SegmentRegressionBody(BaseModel):
    """One category whose WAPE worsened past the gate."""

    category_id: str
    candidate_wape: float
    reference_wape: float


class PromotionBody(BaseModel):
    """The quality-gate record for a model version."""

    reference_id: str
    thresholds: dict[str, float]
    checks: GateChecksBody
    reason: str | None
    p90_coverage: float | None
    regressed_categories: list[SegmentRegressionBody]


class ModelResponse(BaseModel):
    """One model version."""

    id: str
    model_family: str
    version: str
    training_run_id: str
    dataset_id: str
    dataset_version: str
    registry_arn: str
    registry_status: str
    status: ModelStatusName
    metrics: dict[str, object] | None
    rejection_reason: str | None
    promotion: PromotionBody | None
    approved_at: datetime | None
    created_at: datetime


class ModelList(BaseModel):
    """Model versions in registration order."""

    items: list[ModelResponse]


class ApproveRequest(BaseModel):
    """Human approval. ``promote`` also moves the model into production."""

    actor_id: str = Field(min_length=1)
    promote: bool = False


class RejectRequest(BaseModel):
    """Human rejection. The reason must be ``human``."""

    actor_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ForecastCreate(BaseModel):
    """Request a batch forecast from an approved model.

    ``idempotency_key`` returns the original run when the client repeats it.
    """

    model_id: str
    horizon: int = Field(gt=0)
    granularity: Granularity = "day"
    dataset_id: str | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)


class ForecastResponse(BaseModel):
    """One forecast run."""

    id: str
    model_id: str
    model_family: str
    model_version: str
    dataset_id: str
    dataset_version: str
    horizon: int
    granularity: Granularity
    status: ForecastStatus
    output_uri: str
    error_message: str | None
    created_at: datetime


class ForecastList(BaseModel):
    """Forecast runs in request order."""

    items: list[ForecastResponse]


class ForecastPointResponse(BaseModel):
    """One forecast point. ``p10`` and ``p90`` are null until the model emits them."""

    series_id: str
    date: date
    p10: float | None
    p50: float
    p90: float | None
    actual: float | None


class SeriesSummary(BaseModel):
    """Totals for one series, summed from stored points."""

    series_id: str
    point_count: int
    p10_total: float | None
    p50_total: float | None
    p90_total: float | None


class HistoryPoint(BaseModel):
    """Observed demand before the forecast cutoff, summed across the filtered series."""

    date: date
    actual: float


class DailyDemand(BaseModel):
    """One forecast date, summed across the filtered series."""

    date: date
    p10: float | None
    p50: float | None
    p90: float | None
    actual: float | None


class ForecastSeries(BaseModel):
    """Points produced by one forecast run, plus server-side totals for the same points."""

    items: list[ForecastPointResponse]
    series: list[SeriesSummary]
    history: list[HistoryPoint]
    daily: list[DailyDemand]
    cutoff: date | None
    p10_total: float | None
    p50_total: float | None
    p90_total: float | None


class CatalogEntry(BaseModel):
    """One store or category the user can filter on."""

    id: str
    name: str


class SkuEntry(BaseModel):
    """One SKU and the category it belongs to."""

    id: str
    category_id: str


class DatasetCatalog(BaseModel):
    """Filter values read from a dataset's dimension tables."""

    stores: list[CatalogEntry]
    categories: list[CatalogEntry]
    skus: list[SkuEntry]


class NumericDrift(BaseModel):
    """PSI and KS for one numeric column."""

    feature: str
    psi: float
    ks: float


class FrequencyDrift(BaseModel):
    """Absolute change in one rate. Null fields mean the column was absent."""

    feature: str
    baseline: float | None
    recent: float | None
    absolute_delta: float | None


class CoverageDrift(BaseModel):
    """Absolute share changes for stores or categories."""

    feature: str
    absolute_deltas: dict[str, float]
    max_absolute_delta: float


class DriftThresholdsBody(BaseModel):
    """Limits stored with a monitoring report."""

    psi_warning: float
    psi_retrain: float
    wape_degradation_limit: float
    max_dataset_age_days: int
    frequency_warning_delta: float
    recent_window_days: int
    psi_bin_count: int


class MonitoringMetrics(BaseModel):
    """Metric values stored on a monitoring report."""

    dataset_age_days: int
    approved_wape: float | None
    recent_wape: float | None
    wape_degradation: float | None
    numeric: list[NumericDrift]
    frequencies: list[FrequencyDrift]
    coverage: list[CoverageDrift]
    reasons: list[str]
    thresholds: DriftThresholdsBody


MonitoringStatus = Literal["HEALTHY", "WARNING", "RETRAIN_RECOMMENDED"]


class MonitoringReportResponse(BaseModel):
    """A persisted comparison of the recent window with the training baseline."""

    id: str
    status: MonitoringStatus
    model_version_id: str
    dataset_id: str
    as_of: date
    date_max: date
    baseline_start: date
    baseline_end: date
    recent_start: date
    recent_end: date
    dataset_age_days: int
    approved_wape: float | None
    recent_wape: float | None
    wape_degradation: float | None
    numeric: list[NumericDrift]
    frequencies: list[FrequencyDrift]
    coverage: list[CoverageDrift]
    reasons: list[str]
    thresholds: DriftThresholdsBody
    retrain_request_id: str | None
    created_at: datetime


class ModelPerformance(BaseModel):
    """Scores stored on registered models, plus the latest monitoring report."""

    items: list[ModelResponse]
    monitoring: MonitoringReportResponse | None = None


class DataQualityItem(BaseModel):
    """Latest quality report stored for a dataset."""

    dataset_id: str
    dataset_version: str
    status: DatasetStatus
    quality_report: dict[str, object] | None


class DataQualityList(BaseModel):
    """Quality reports in dataset registration order."""

    items: list[DataQualityItem]


class ExplanationSignal(BaseModel):
    """One signal named by a validated explanation."""

    name: str
    direction: Literal["positive", "negative"] | None = None
    value: float | None = None
    kind: Literal["driver", "context"] = "context"


class ScheduledForecastResponse(BaseModel):
    """A data refresh marker and the forecast it started."""

    marker_id: str
    dataset_id: str
    refreshed_at: datetime
    forecast_id: str
    forecast_status: ForecastStatus


class ForecastErrorEvaluationResponse(BaseModel):
    """Error between a stored forecast and actuals that have arrived."""

    id: str
    forecast_run_id: str
    compared_points: int
    wape: float | None
    bias: float | None
    created_at: datetime


RetrainRequestStatus = Literal["PENDING", "CONFIRMED"]


class RetrainRequestResponse(BaseModel):
    """A retrain request waiting for a person, or the training run they confirmed."""

    id: str
    status: RetrainRequestStatus
    dataset_id: str
    model_version_id: str
    model_family: str
    training_run_id: str | None
    requested_at: datetime
    confirmed_at: datetime | None
    confirmed_by: str | None


class RetrainRequestList(BaseModel):
    """Retrain requests in creation order."""

    items: list[RetrainRequestResponse]


class ConfirmRetrainRequest(BaseModel):
    """Person who confirms a retrain request. Confirmation does not promote the model."""

    actor_id: str = Field(min_length=1)


class ExplanationResponse(BaseModel):
    """A validated explanation and the package it was written from."""

    id: str
    model_id: str
    prompt_version: str
    summary: str
    signals: list[ExplanationSignal]
    risks: list[str]
    uncertainty: str
    checks: list[str]
    generated_at: datetime
    package: dict[str, object]

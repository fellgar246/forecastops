locals {
  log_retention_days = var.environment == "demo" ? 14 : 7
  api_environment = {
    EXECUTION_MODE                   = "aws"
    AWS_ENABLED                      = "true"
    BEDROCK_ENABLED                  = var.enable_bedrock ? "true" : "false"
    SAGEMAKER_ENABLED                = "false"
    ONLINE_INFERENCE                 = "false"
    AI_ENABLED                       = "true"
    TRAINING_ENABLED                 = "true"
    AWS_ML_ENABLED                   = "false"
    MAX_BEDROCK_CALLS_PER_DAY        = "30"
    MAX_BEDROCK_INPUT_TOKENS         = "5000"
    MAX_BEDROCK_OUTPUT_TOKENS        = "700"
    MAX_TRAINING_JOBS_PER_DAY        = "2"
    MAX_TRAINING_RUNTIME_MINUTES     = "45"
    ALLOW_GPU_TRAINING               = "false"
    MAX_BATCH_INFERENCE_JOBS_PER_DAY = "5"
    MAX_DATASET_ROWS_DEMO            = "2000000"
    MAX_FORECAST_HORIZON_DAYS        = "90"
    RANDOM_SEED                      = "20260921"
    METADATA_TABLE_NAME              = module.dynamodb.table_name
    ARTIFACTS_BUCKET                 = module.s3.artifacts_bucket_name
    AWS_REGION                       = var.aws_region
    WEB_ORIGIN                       = "http://localhost:3000"
    AUTH_ENABLED                     = "true"
    COGNITO_USER_POOL_ID             = module.cognito.user_pool_id
    COGNITO_APP_CLIENT_ID            = module.cognito.app_client_id
  }
}

module "budget" {
  source = "../../modules/budget"

  environment = var.environment
  alert_email = var.budget_alert_email
}

module "cloudwatch" {
  source = "../../modules/cloudwatch"

  environment       = var.environment
  retention_in_days = local.log_retention_days
}

module "observability" {
  source = "../../modules/observability"

  environment          = var.environment
  aws_region           = var.aws_region
  enable_notifications = var.enable_alarm_notifications
  alert_email          = var.budget_alert_email
}

module "s3" {
  source = "../../modules/s3"

  environment = var.environment
  name_prefix = var.name_prefix
}

module "dynamodb" {
  source = "../../modules/dynamodb"

  environment = var.environment
}

module "cognito" {
  source = "../../modules/cognito"

  environment       = var.environment
  retention_in_days = local.log_retention_days
}

module "iam" {
  source = "../../modules/iam"

  environment          = var.environment
  data_bucket_arn      = module.s3.data_bucket_arn
  artifacts_bucket_arn = module.s3.artifacts_bucket_arn
  forecasts_bucket_arn = module.s3.forecasts_bucket_arn
  metadata_table_arn   = module.dynamodb.table_arn
  log_group_arn        = module.cloudwatch.log_group_arn
}

module "lambda_api" {
  source = "../../modules/lambda"

  environment           = var.environment
  role_arn              = module.iam.api_role_arn
  log_group_name        = module.cloudwatch.log_group_name
  environment_variables = local.api_environment
  use_packaged_api      = var.use_packaged_api
}

module "api_gateway" {
  source = "../../modules/api_gateway"

  environment          = var.environment
  lambda_invoke_arn    = module.lambda_api.invoke_arn
  lambda_function_name = module.lambda_api.function_name
}

module "eventbridge" {
  source = "../../modules/eventbridge"

  environment      = var.environment
  enable_schedules = var.enable_schedules
}

module "sagemaker" {
  source = "../../modules/sagemaker"

  environment                = var.environment
  enable_serverless_endpoint = var.enable_serverless_endpoint
}

module "bedrock_permissions" {
  source = "../../modules/bedrock_permissions"

  enable_bedrock        = var.enable_bedrock
  explanation_role_name = module.iam.explanation_role_name
}

module "github_oidc" {
  source = "../../modules/github_oidc"

  environment          = var.environment
  github_repository    = var.github_repository
  aws_region           = var.aws_region
  create_oidc_provider = var.create_oidc_provider
  create_deploy_role   = true
  data_bucket_arn      = module.s3.data_bucket_arn
  artifacts_bucket_arn = module.s3.artifacts_bucket_arn
  forecasts_bucket_arn = module.s3.forecasts_bucket_arn
  metadata_table_arn   = module.dynamodb.table_arn
  log_group_arn        = module.cloudwatch.log_group_arn
  lambda_function_arn  = module.lambda_api.function_arn
  api_id               = module.api_gateway.api_id
  budget_arn           = module.budget.budget_arn
  pass_role_arns = [
    module.iam.api_role_arn,
    module.iam.pipeline_role_arn,
    module.iam.training_role_arn,
    module.iam.inference_role_arn,
    module.iam.explanation_role_arn,
  ]
}

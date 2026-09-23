locals {
  log_retention_days = var.environment == "demo" ? 14 : 7
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

module "s3" {
  source = "../../modules/s3"

  environment = var.environment
  name_prefix = var.name_prefix
}

module "dynamodb" {
  source = "../../modules/dynamodb"

  environment = var.environment
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

  environment    = var.environment
  role_arn       = module.iam.api_role_arn
  log_group_name = module.cloudwatch.log_group_name
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

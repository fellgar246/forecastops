variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "github_repository" {
  description = "GitHub repository allowed to assume the deployment role, as org/name."
  type        = string
}

variable "aws_region" {
  description = "Region used in API Gateway permission ARNs."
  type        = string
}

variable "create_oidc_provider" {
  description = "Create the GitHub OIDC provider. Only one environment in an account should do this."
  type        = bool
}

variable "data_bucket_arn" {
  type = string
}

variable "artifacts_bucket_arn" {
  type = string
}

variable "forecasts_bucket_arn" {
  type = string
}

variable "metadata_table_arn" {
  type = string
}

variable "log_group_arn" {
  type = string
}

variable "lambda_function_arn" {
  type = string
}

variable "api_id" {
  type = string
}

variable "budget_arn" {
  type = string
}

variable "pass_role_arns" {
  description = "Project roles the deployment identity may pass to Lambda or SageMaker."
  type        = list(string)
}

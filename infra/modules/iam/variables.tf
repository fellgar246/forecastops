variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "data_bucket_arn" {
  description = "ARN of the dataset bucket."
  type        = string
}

variable "artifacts_bucket_arn" {
  description = "ARN of the model artifact bucket."
  type        = string
}

variable "forecasts_bucket_arn" {
  description = "ARN of the forecast output bucket."
  type        = string
}

variable "metadata_table_arn" {
  description = "ARN of the metadata table."
  type        = string
}

variable "log_group_arn" {
  description = "ARN of the API log group."
  type        = string
}

variable "aws_region" {
  description = "Region for this environment."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Active environment name."
  type        = string
  default     = "dev"
}

variable "name_prefix" {
  description = "Prefix for globally unique bucket names. Change this before applying if the default is taken."
  type        = string
  default     = "forecastops"
}

variable "budget_alert_email" {
  description = "Inbox for budget alerts. Replace the placeholder before applying."
  type        = string
  default     = "owner@example.com"
}

variable "github_repository" {
  description = "GitHub repository allowed to deploy, as org/name."
  type        = string
  default     = "example/forecastops"
}

variable "enable_serverless_endpoint" {
  description = "Keep the optional serverless endpoint off."
  type        = bool
  default     = false
}

variable "enable_bedrock" {
  description = "Keep Bedrock permissions off."
  type        = bool
  default     = false
}

variable "enable_schedules" {
  description = "Create the daily, weekly, and monthly schedules. The default creates none."
  type        = bool
  default     = false
}

variable "enable_alarm_notifications" {
  description = "Subscribe operational alarms. Stays false until the demo environment should send notifications."
  type        = bool
  default     = false
}

variable "create_oidc_provider" {
  description = "Create the account-wide GitHub OIDC provider from this environment."
  type        = bool
  default     = true
}

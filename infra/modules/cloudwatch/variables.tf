variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "retention_in_days" {
  description = "CloudWatch log retention. Dev uses 7 days and demo uses 14."
  type        = number
}

variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "retention_in_days" {
  description = "Retention for the pre-token function logs."
  type        = number
}

variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "enable_serverless_endpoint" {
  description = "Reserved switch for an optional serverless endpoint. No endpoint is declared."
  type        = bool
}

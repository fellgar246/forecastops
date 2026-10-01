variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "role_arn" {
  description = "Execution role for the control-plane function."
  type        = string
}

variable "log_group_name" {
  description = "Log group created ahead of the function so retention applies."
  type        = string
}

variable "use_packaged_api" {
  description = "Deploy the packaged HTTP API instead of the health stub. The package directory must already exist."
  type        = bool
  default     = false
}

variable "environment_variables" {
  description = "Runtime configuration for the API function. Do not include access keys."
  type        = map(string)
  default     = {}
}

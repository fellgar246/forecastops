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

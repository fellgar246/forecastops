variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "lambda_invoke_arn" {
  description = "Invoke ARN of the control-plane function."
  type        = string
}

variable "lambda_function_name" {
  description = "Name of the control-plane function."
  type        = string
}

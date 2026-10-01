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

variable "web_origin" {
  description = "Browser origin allowed to call the API from a laptop."
  type        = string
  default     = "http://localhost:3000"
}

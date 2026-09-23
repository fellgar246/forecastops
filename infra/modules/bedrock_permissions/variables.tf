variable "enable_bedrock" {
  description = "Attach permission to invoke an explanation model. Off in the default environments."
  type        = bool
}

variable "explanation_role_name" {
  description = "Role that would receive invoke permission when Bedrock is enabled."
  type        = string
}

variable "model_arns" {
  description = "Foundation model ARNs allowed when Bedrock is enabled."
  type        = list(string)
  default     = []
}

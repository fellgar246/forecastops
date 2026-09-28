variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "name_prefix" {
  description = "Prefix for globally unique bucket names."
  type        = string
}

variable "noncurrent_version_expiration_days" {
  description = "Days to keep a non-current object version before it expires."
  type        = number
  default     = 30

  validation {
    condition     = var.noncurrent_version_expiration_days >= 1
    error_message = "Non-current versions must expire after at least one day."
  }
}

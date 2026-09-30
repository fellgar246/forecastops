variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "enable_schedules" {
  description = "Create the daily, weekly, and monthly schedules. Leave this false to create none."
  type        = bool
}

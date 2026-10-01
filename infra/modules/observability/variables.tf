variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "aws_region" {
  description = "Region for the dashboard widgets."
  type        = string
}

variable "enable_notifications" {
  description = "Subscribe alarm email in the demo environment. Other environments stay unsubscribed."
  type        = bool
  default     = false
}

variable "alert_email" {
  description = "Inbox for alarm email when demo notifications are enabled."
  type        = string
  default     = ""
}

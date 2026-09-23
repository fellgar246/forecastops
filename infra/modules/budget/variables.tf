variable "environment" {
  description = "Active environment name, used in resource names and tags."
  type        = string
}

variable "alert_email" {
  description = "Inbox that receives budget alerts. Change the placeholder before applying."
  type        = string
}

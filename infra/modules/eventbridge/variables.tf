variable "environment" {
  description = "Active environment name."
  type        = string
}

variable "enable_schedules" {
  description = "Reserved switch for future schedules. No schedule is declared while this stays false."
  type        = bool
}

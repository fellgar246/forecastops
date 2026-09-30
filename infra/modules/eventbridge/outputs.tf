output "schedules_enabled" {
  description = "Whether the daily, weekly, and monthly schedules are created."
  value       = var.enable_schedules
}

output "enabled_schedule_count" {
  description = "Number of schedules created. Zero when schedules are left off."
  value       = var.enable_schedules ? length(local.schedules) : 0
}

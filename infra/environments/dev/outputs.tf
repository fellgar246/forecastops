output "serverless_endpoint_enabled" {
  value = module.sagemaker.serverless_endpoint_enabled
}

output "bedrock_enabled" {
  value = module.bedrock_permissions.bedrock_enabled
}

output "schedules_enabled" {
  value = module.eventbridge.schedules_enabled
}

output "enabled_schedule_count" {
  value = module.eventbridge.enabled_schedule_count
}

output "budget_arn" {
  value = module.budget.budget_arn
}

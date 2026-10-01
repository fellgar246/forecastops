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

output "alarm_notifications_enabled" {
  value = module.observability.notifications_enabled
}

output "api_endpoint" {
  value = module.api_gateway.api_endpoint
}

output "metadata_table_name" {
  value = module.dynamodb.table_name
}

output "cognito_user_pool_id" {
  value = module.cognito.user_pool_id
}

output "cognito_app_client_id" {
  value = module.cognito.app_client_id
}

output "deploy_role_arn" {
  value = module.github_oidc.deploy_role_arn
}

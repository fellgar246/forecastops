output "dashboard_name" {
  value = aws_cloudwatch_dashboard.operations.dashboard_name
}

output "notifications_enabled" {
  value = local.notifications_enabled
}

output "alarm_names" {
  value = [
    aws_cloudwatch_metric_alarm.training_failure.alarm_name,
    aws_cloudwatch_metric_alarm.repeated_forecast_failure.alarm_name,
    aws_cloudwatch_metric_alarm.stale_dataset.alarm_name,
    aws_cloudwatch_metric_alarm.high_forecast_error.alarm_name,
    aws_cloudwatch_metric_alarm.explanation_error_spike.alarm_name,
  ]
}

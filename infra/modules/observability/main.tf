locals {
  namespace = "ForecastOps"
  # Dev stays unsubscribed. Demo subscribes only when that environment asks.
  notifications_enabled = var.environment == "demo" && var.enable_notifications
  alarm_actions         = aws_sns_topic.alarms[*].arn
  groups = [
    {
      title   = "API"
      metrics = ["api_request_count", "api_error_count", "api_latency_ms"]
    },
    {
      title   = "Training"
      metrics = ["training_job_count", "training_duration_seconds", "training_failures"]
    },
    {
      title   = "Forecast"
      metrics = ["forecast_job_count", "forecast_latency_seconds"]
    },
    {
      title = "Explanation"
      metrics = [
        "bedrock_calls",
        "bedrock_input_tokens",
        "bedrock_output_tokens",
        "bedrock_latency",
        "bedrock_failures",
        "explanation_validation_failures",
      ]
    },
  ]
}

resource "aws_sns_topic" "alarms" {
  count = local.notifications_enabled ? 1 : 0
  name  = "forecastops-${var.environment}-alarms"
}

resource "aws_sns_topic_subscription" "alarms" {
  count     = local.notifications_enabled ? 1 : 0
  topic_arn = aws_sns_topic.alarms[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_dashboard" "operations" {
  dashboard_name = "forecastops-${var.environment}"
  dashboard_body = jsonencode({
    widgets = flatten([
      for index, group in local.groups : [
        {
          type   = "text"
          x      = 0
          y      = index * 7
          width  = 24
          height = 1
          properties = {
            markdown = "## ${group.title}"
          }
        },
        {
          type   = "metric"
          x      = 0
          y      = index * 7 + 1
          width  = 24
          height = 6
          properties = {
            title   = group.title
            region  = var.aws_region
            view    = "timeSeries"
            period  = 300
            stat    = "Sum"
            metrics = [for name in group.metrics : [local.namespace, name]]
          }
        },
      ]
    ])
  })
}

resource "aws_cloudwatch_metric_alarm" "training_failure" {
  alarm_name          = "forecastops-${var.environment}-training-failure"
  alarm_description   = "A training job failed."
  namespace           = local.namespace
  metric_name         = "training_failures"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.notifications_enabled
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "repeated_forecast_failure" {
  alarm_name          = "forecastops-${var.environment}-repeated-forecast-failure"
  alarm_description   = "More than one forecast job failed in the window."
  namespace           = local.namespace
  metric_name         = "forecast_job_count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  datapoints_to_alarm = 1
  threshold           = 2
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.notifications_enabled
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  dimensions = {
    outcome = "failed"
  }
}

resource "aws_cloudwatch_metric_alarm" "stale_dataset" {
  alarm_name          = "forecastops-${var.environment}-stale-dataset"
  alarm_description   = "The newest observation is older than 30 days."
  namespace           = local.namespace
  metric_name         = "latest_data_age_hours"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 720
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.notifications_enabled
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "high_forecast_error" {
  alarm_name          = "forecastops-${var.environment}-high-forecast-error"
  alarm_description   = "Forecast WAPE is at or above 0.5."
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  threshold           = 0.5
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.notifications_enabled
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id          = "wape"
    expression  = "MAX(SEARCH('{ForecastOps,model_version,category,forecast_horizon} MetricName=\"wape\"', 'Maximum', 300))"
    return_data = true
  }
}

resource "aws_cloudwatch_metric_alarm" "explanation_error_spike" {
  alarm_name          = "forecastops-${var.environment}-explanation-error-spike"
  alarm_description   = "Explanation validation or provider errors spiked."
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  threshold           = 5
  treat_missing_data  = "notBreaching"
  actions_enabled     = local.notifications_enabled
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id = "validation"

    metric {
      namespace   = local.namespace
      metric_name = "explanation_validation_failures"
      period      = 300
      stat        = "Sum"
    }
  }

  metric_query {
    id = "provider"

    metric {
      namespace   = local.namespace
      metric_name = "bedrock_failures"
      period      = 300
      stat        = "Sum"
    }
  }

  metric_query {
    id          = "errors"
    expression  = "validation + provider"
    return_data = true
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/forecastops-${var.environment}-api"
  retention_in_days = var.retention_in_days
}

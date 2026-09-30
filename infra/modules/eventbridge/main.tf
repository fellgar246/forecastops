# Daily, weekly, and monthly clocks. Nothing here is created unless
# enable_schedules is true, so the default environment has zero schedules.
# The monthly action asks for confirmation. It does not start training.

terraform {
  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0.0"
    }
  }
}

locals {
  schedules = {
    daily_forecast = {
      expression  = "cron(0 6 * * ? *)"
      description = "Refresh the data marker and generate the latest forecast."
    }
    weekly_evaluation = {
      expression  = "cron(0 6 ? * MON *)"
      description = "Evaluate forecast error against actuals that have arrived."
    }
    monthly_retrain = {
      expression  = "cron(0 6 1 * ? *)"
      description = "Request retraining. A person must confirm before training starts."
    }
  }
}

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "scheduler_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

data "archive_file" "handler" {
  count = var.enable_schedules ? 1 : 0

  type        = "zip"
  source_file = "${path.module}/src/handler.py"
  output_path = "${path.module}/build/handler.zip"
}

resource "aws_iam_role" "schedules" {
  count = var.enable_schedules ? 1 : 0

  name               = "forecastops-${var.environment}-schedules"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role_policy" "schedules_logs" {
  count = var.enable_schedules ? 1 : 0

  name = "logs"
  role = aws_iam_role.schedules[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/lambda/forecastops-${var.environment}-schedules:*"
      }
    ]
  })
}

resource "aws_lambda_function" "schedules" {
  count = var.enable_schedules ? 1 : 0

  function_name    = "forecastops-${var.environment}-schedules"
  description      = "Receives a scheduled action. Monthly retraining stays pending until a person confirms it."
  role             = aws_iam_role.schedules[0].arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.handler[0].output_path
  source_code_hash = data.archive_file.handler[0].output_base64sha256
  memory_size      = 128
  timeout          = 10
}

resource "aws_iam_role" "scheduler" {
  count = var.enable_schedules ? 1 : 0

  name               = "forecastops-${var.environment}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_trust.json
}

resource "aws_iam_role_policy" "scheduler_invoke" {
  count = var.enable_schedules ? 1 : 0

  name = "invoke-schedules"
  role = aws_iam_role.scheduler[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "lambda:InvokeFunction"
        Resource = aws_lambda_function.schedules[0].arn
      }
    ]
  })
}

resource "aws_lambda_permission" "scheduler" {
  count = var.enable_schedules ? 1 : 0

  statement_id  = "AllowEventBridgeScheduler"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.schedules[0].function_name
  principal     = "scheduler.amazonaws.com"
}

resource "aws_scheduler_schedule" "clock" {
  for_each = var.enable_schedules ? local.schedules : {}

  name                = "forecastops-${var.environment}-${each.key}"
  description         = each.value.description
  schedule_expression = each.value.expression
  state               = "ENABLED"

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.schedules[0].arn
    role_arn = aws_iam_role.scheduler[0].arn
    input    = jsonencode({ action = each.key })
  }
}

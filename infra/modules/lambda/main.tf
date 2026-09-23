terraform {
  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0.0"
    }
  }
}

data "archive_file" "handler" {
  type        = "zip"
  source_file = "${path.module}/src/handler.py"
  output_path = "${path.module}/build/handler.zip"
}

resource "aws_lambda_function" "api" {
  function_name    = "forecastops-${var.environment}-api"
  description      = "Control-plane health handler. This function does not train models."
  role             = var.role_arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = data.archive_file.handler.output_path
  source_code_hash = data.archive_file.handler.output_base64sha256
  memory_size      = 128
  timeout          = 10

  environment {
    variables = {
      LOG_GROUP = var.log_group_name
    }
  }
}

terraform {
  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0.0"
    }
  }
}

# Default plans ship the health stub. `make aws-deploy` builds the API package
# and sets use_packaged_api so this function serves the HTTP routes.
data "archive_file" "stub" {
  count = var.use_packaged_api ? 0 : 1

  type        = "zip"
  source_file = "${path.module}/src/handler.py"
  output_path = "${path.module}/build/stub.zip"
}

data "archive_file" "packaged" {
  count = var.use_packaged_api ? 1 : 0

  type        = "zip"
  source_dir  = "${path.module}/build/package"
  output_path = "${path.module}/build/api.zip"
  excludes    = ["__pycache__", "*.pyc"]
}

locals {
  filename = concat(
    data.archive_file.packaged[*].output_path,
    data.archive_file.stub[*].output_path,
  )[0]
  source_code_hash = concat(
    data.archive_file.packaged[*].output_base64sha256,
    data.archive_file.stub[*].output_base64sha256,
  )[0]
}

resource "aws_lambda_function" "api" {
  function_name    = "forecastops-${var.environment}-api"
  description      = "HTTP API for the cloud control plane. It does not declare a training job or a real-time endpoint."
  role             = var.role_arn
  runtime          = "python3.12"
  handler          = "handler.handler"
  filename         = local.filename
  source_code_hash = local.source_code_hash
  memory_size      = 1024
  timeout          = 30

  environment {
    variables = merge(var.environment_variables, {
      LOG_GROUP = var.log_group_name
    })
  }
}

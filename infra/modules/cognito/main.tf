# Cognito user pool for organization access tokens.
# Users are created by an operator. Self-service signup stays off.
# The pre-token function copies custom:tenant_id onto the access token
# as the claim tenant_id.

terraform {
  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0.0"
    }
  }
}

data "aws_region" "current" {}

data "archive_file" "pre_token" {
  type        = "zip"
  source_file = "${path.module}/src/pre_token.py"
  output_path = "${path.module}/build/pre_token.zip"
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

resource "aws_cloudwatch_log_group" "pre_token" {
  name              = "/aws/lambda/forecastops-${var.environment}-pre-token"
  retention_in_days = var.retention_in_days
}

resource "aws_iam_role" "pre_token" {
  name               = "forecastops-${var.environment}-pre-token"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role_policy" "pre_token_logs" {
  name = "logs"
  role = aws_iam_role.pre_token.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "${aws_cloudwatch_log_group.pre_token.arn}:*"
      }
    ]
  })
}

resource "aws_lambda_function" "pre_token" {
  function_name    = "forecastops-${var.environment}-pre-token"
  description      = "Copies the tenant attribute onto Cognito access tokens."
  role             = aws_iam_role.pre_token.arn
  runtime          = "python3.12"
  handler          = "pre_token.handler"
  filename         = data.archive_file.pre_token.output_path
  source_code_hash = data.archive_file.pre_token.output_base64sha256
  memory_size      = 128
  timeout          = 5

  depends_on = [aws_cloudwatch_log_group.pre_token]
}

resource "aws_cognito_user_pool" "tenants" {
  name = "forecastops-${var.environment}"

  admin_create_user_config {
    allow_admin_create_user_only = true
  }

  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = false
    temporary_password_validity_days = 7
  }

  schema {
    attribute_data_type = "String"
    mutable             = true
    name                = "tenant_id"
    required            = false

    string_attribute_constraints {
      min_length = 1
      max_length = 64
    }
  }

  lambda_config {
    pre_token_generation_config {
      lambda_arn     = aws_lambda_function.pre_token.arn
      lambda_version = "V2_0"
    }
  }
}

resource "aws_cognito_user_pool_client" "web" {
  name         = "forecastops-${var.environment}-web"
  user_pool_id = aws_cognito_user_pool.tenants.id

  generate_secret = false
  explicit_auth_flows = [
    "ALLOW_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
  ]
  prevent_user_existence_errors = "ENABLED"
  access_token_validity         = 1
  id_token_validity             = 1
  refresh_token_validity        = 30

  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }
}

resource "aws_lambda_permission" "cognito" {
  statement_id  = "AllowCognitoInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.pre_token.function_name
  principal     = "cognito-idp.amazonaws.com"
  source_arn    = aws_cognito_user_pool.tenants.arn
}

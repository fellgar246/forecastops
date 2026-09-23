data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "api" {
  name               = "forecastops-${var.environment}-api"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role" "pipeline" {
  name               = "forecastops-${var.environment}-pipeline"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role" "training" {
  name               = "forecastops-${var.environment}-training"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

resource "aws_iam_role" "inference" {
  name               = "forecastops-${var.environment}-inference"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

resource "aws_iam_role" "explanation" {
  name = "forecastops-${var.environment}-explanation"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = "sts:AssumeRole"
        Principal = {
          AWS = aws_iam_role.api.arn
        }
      }
    ]
  })
}

resource "aws_iam_policy" "api" {
  name = "forecastops-${var.environment}-api"
  policy = templatefile("${path.module}/../../policies/api.json", {
    metadata_table_arn   = var.metadata_table_arn
    forecasts_bucket_arn = var.forecasts_bucket_arn
    log_group_arn        = var.log_group_arn
    explanation_role_arn = aws_iam_role.explanation.arn
  })
}

resource "aws_iam_role_policy_attachment" "api" {
  role       = aws_iam_role.api.name
  policy_arn = aws_iam_policy.api.arn
}

resource "aws_iam_policy" "pipeline" {
  name = "forecastops-${var.environment}-pipeline"
  policy = templatefile("${path.module}/../../policies/pipeline.json", {
    data_bucket_arn      = var.data_bucket_arn
    artifacts_bucket_arn = var.artifacts_bucket_arn
    metadata_table_arn   = var.metadata_table_arn
  })
}

resource "aws_iam_role_policy_attachment" "pipeline" {
  role       = aws_iam_role.pipeline.name
  policy_arn = aws_iam_policy.pipeline.arn
}

resource "aws_iam_policy" "training" {
  name = "forecastops-${var.environment}-training"
  policy = templatefile("${path.module}/../../policies/training.json", {
    data_bucket_arn      = var.data_bucket_arn
    artifacts_bucket_arn = var.artifacts_bucket_arn
  })
}

resource "aws_iam_role_policy_attachment" "training" {
  role       = aws_iam_role.training.name
  policy_arn = aws_iam_policy.training.arn
}

resource "aws_iam_policy" "inference" {
  name = "forecastops-${var.environment}-inference"
  policy = templatefile("${path.module}/../../policies/inference.json", {
    artifacts_bucket_arn = var.artifacts_bucket_arn
    forecasts_bucket_arn = var.forecasts_bucket_arn
  })
}

resource "aws_iam_role_policy_attachment" "inference" {
  role       = aws_iam_role.inference.name
  policy_arn = aws_iam_policy.inference.arn
}

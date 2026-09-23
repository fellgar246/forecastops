output "api_role_arn" {
  value = aws_iam_role.api.arn
}

output "pipeline_role_arn" {
  value = aws_iam_role.pipeline.arn
}

output "training_role_arn" {
  value = aws_iam_role.training.arn
}

output "inference_role_arn" {
  value = aws_iam_role.inference.arn
}

output "explanation_role_arn" {
  value = aws_iam_role.explanation.arn
}

output "explanation_role_name" {
  value = aws_iam_role.explanation.name
}

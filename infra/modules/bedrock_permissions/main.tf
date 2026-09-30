# Invoke permission is attached only to the explanation role.
# The training role does not receive it.
resource "aws_iam_role_policy" "invoke" {
  count = var.enable_bedrock ? 1 : 0

  name = "bedrock-invoke"
  role = var.explanation_role_name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "InvokeExplanationModel"
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = var.model_arns
      }
    ]
  })

  lifecycle {
    precondition {
      condition     = length(var.model_arns) > 0
      error_message = "At least one Bedrock model ARN is required when Bedrock is enabled."
    }
  }
}

output "api_id" {
  value = aws_apigatewayv2_api.http.id
}

output "execution_arn" {
  value = aws_apigatewayv2_api.http.execution_arn
}

output "api_endpoint" {
  value = aws_apigatewayv2_api.http.api_endpoint
}

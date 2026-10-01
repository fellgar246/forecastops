output "user_pool_id" {
  value = aws_cognito_user_pool.tenants.id
}

output "app_client_id" {
  value = aws_cognito_user_pool_client.web.id
}

output "issuer" {
  value = "https://cognito-idp.${data.aws_region.current.name}.amazonaws.com/${aws_cognito_user_pool.tenants.id}"
}

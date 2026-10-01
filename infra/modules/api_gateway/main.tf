resource "aws_apigatewayv2_api" "http" {
  name          = "forecastops-${var.environment}"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins  = [var.web_origin]
    allow_methods  = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    allow_headers  = ["authorization", "content-type", "accept", "x-correlation-id"]
    expose_headers = ["x-correlation-id"]
    max_age        = 300
  }
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = var.lambda_invoke_arn
  payload_format_version = "2.0"
}

# These route keys match the HTTP API. The function dispatches them.
locals {
  routes = toset([
    "GET /health",
    "GET /health/aws",
    "GET /cost",
    "GET /metrics",
    "POST /datasets",
    "POST /datasets/uploads",
    "POST /datasets/upload-url",
    "GET /datasets",
    "GET /datasets/{dataset_id}",
    "GET /datasets/{dataset_id}/catalog",
    "POST /datasets/{dataset_id}/validate",
    "POST /training-runs",
    "GET /training-runs",
    "GET /training-runs/{run_id}",
    "GET /models",
    "GET /models/{model_id}",
    "POST /models/{model_id}/approve",
    "POST /models/{model_id}/reject",
    "POST /forecasts",
    "GET /forecasts",
    "GET /forecasts/{forecast_id}",
    "GET /forecasts/{forecast_id}/series",
    "POST /forecasts/{forecast_id}/explanation",
    "GET /forecasts/{forecast_id}/explanation",
    "GET /metrics/model-performance",
    "POST /admin/monitoring",
    "GET /metrics/data-quality",
    "POST /admin/retrain",
    "POST /admin/schedules/daily_forecast",
    "POST /admin/schedules/weekly_evaluation",
    "POST /admin/schedules/monthly_retrain",
    "GET /admin/retrain-requests",
    "POST /admin/retrain-requests/{request_id}/confirm",
  ])
}

resource "aws_apigatewayv2_route" "api" {
  for_each = local.routes

  api_id    = aws_apigatewayv2_api.http.id
  route_key = each.value
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    throttling_burst_limit = 5
    throttling_rate_limit  = 2
  }
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowApiGateway"
  action        = "lambda:InvokeFunction"
  function_name = var.lambda_function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*"
}

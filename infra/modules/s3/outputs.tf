output "data_bucket_arn" {
  value = aws_s3_bucket.this["data"].arn
}

output "artifacts_bucket_arn" {
  value = aws_s3_bucket.this["artifacts"].arn
}

output "forecasts_bucket_arn" {
  value = aws_s3_bucket.this["forecasts"].arn
}

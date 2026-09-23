locals {
  buckets = toset(["data", "artifacts", "forecasts"])
}

resource "aws_s3_bucket" "this" {
  for_each      = local.buckets
  bucket        = "${var.name_prefix}-${var.environment}-${each.key}"
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each = aws_s3_bucket.this

  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_policy" "deny_insecure" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  policy = templatefile("${path.module}/../../policies/s3-deny-insecure-transport.json", {
    bucket_arn = each.value.arn
  })

  depends_on = [aws_s3_bucket_public_access_block.this]
}

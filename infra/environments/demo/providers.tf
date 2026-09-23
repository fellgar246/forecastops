terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.50.0, < 6.0.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.0.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ml-demand-forecasting"
      Environment = var.environment
      Owner       = "portfolio"
      ManagedBy   = "terraform"
      CostCenter  = "learning"
      AutoCleanup = "true"
    }
  }
}

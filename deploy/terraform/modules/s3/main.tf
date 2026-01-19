# Builder Engine - S3 Module

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "account_id" {
  type = string
}

variable "create_recordings_bucket" {
  type    = bool
  default = true
}

variable "create_transcripts_bucket" {
  type    = bool
  default = true
}

variable "create_documents_bucket" {
  type    = bool
  default = true
}

variable "create_assets_bucket" {
  type    = bool
  default = true
}

variable "enable_versioning" {
  type    = bool
  default = true
}

variable "enable_encryption" {
  type    = bool
  default = true
}

variable "lifecycle_rules" {
  type = map(object({
    transition_ia_days      = optional(number)
    transition_glacier_days = optional(number)
    expiration_days         = optional(number)
  }))
  default = {}
}

variable "tags" {
  type    = map(string)
  default = {}
}

# =============================================================================
# Local Variables
# =============================================================================

locals {
  buckets = {
    recordings  = var.create_recordings_bucket
    transcripts = var.create_transcripts_bucket
    documents   = var.create_documents_bucket
    assets      = var.create_assets_bucket
  }
}

# =============================================================================
# KMS Key for S3 Encryption
# =============================================================================

resource "aws_kms_key" "s3" {
  count = var.enable_encryption ? 1 : 0

  description             = "S3 encryption key for ${var.name_prefix}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${var.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow S3 Service"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey*",
          "kms:Decrypt"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-s3-key"
  })
}

resource "aws_kms_alias" "s3" {
  count = var.enable_encryption ? 1 : 0

  name          = "alias/${var.name_prefix}-s3"
  target_key_id = aws_kms_key.s3[0].key_id
}

# =============================================================================
# S3 Buckets
# =============================================================================

resource "aws_s3_bucket" "main" {
  for_each = { for k, v in local.buckets : k => v if v }

  bucket = "${var.name_prefix}-${each.key}-${var.account_id}"

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-${each.key}"
  })
}

resource "aws_s3_bucket_versioning" "main" {
  for_each = { for k, v in local.buckets : k => v if v }

  bucket = aws_s3_bucket.main[each.key].id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  for_each = { for k, v in local.buckets : k => v if v && var.enable_encryption }

  bucket = aws_s3_bucket.main[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3[0].arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  for_each = { for k, v in local.buckets : k => v if v }

  bucket = aws_s3_bucket.main[each.key].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "recordings" {
  count = var.create_recordings_bucket && lookup(var.lifecycle_rules, "recordings", null) != null ? 1 : 0

  bucket = aws_s3_bucket.main["recordings"].id

  rule {
    id     = "recordings-lifecycle"
    status = "Enabled"

    dynamic "transition" {
      for_each = var.lifecycle_rules["recordings"].transition_glacier_days != null ? [1] : []
      content {
        days          = var.lifecycle_rules["recordings"].transition_glacier_days
        storage_class = "GLACIER"
      }
    }

    dynamic "expiration" {
      for_each = var.lifecycle_rules["recordings"].expiration_days != null ? [1] : []
      content {
        days = var.lifecycle_rules["recordings"].expiration_days
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "transcripts" {
  count = var.create_transcripts_bucket && lookup(var.lifecycle_rules, "transcripts", null) != null ? 1 : 0

  bucket = aws_s3_bucket.main["transcripts"].id

  rule {
    id     = "transcripts-lifecycle"
    status = "Enabled"

    dynamic "transition" {
      for_each = var.lifecycle_rules["transcripts"].transition_ia_days != null ? [1] : []
      content {
        days          = var.lifecycle_rules["transcripts"].transition_ia_days
        storage_class = "STANDARD_IA"
      }
    }

    dynamic "transition" {
      for_each = var.lifecycle_rules["transcripts"].transition_glacier_days != null ? [1] : []
      content {
        days          = var.lifecycle_rules["transcripts"].transition_glacier_days
        storage_class = "GLACIER"
      }
    }

    dynamic "expiration" {
      for_each = var.lifecycle_rules["transcripts"].expiration_days != null ? [1] : []
      content {
        days = var.lifecycle_rules["transcripts"].expiration_days
      }
    }
  }
}

# =============================================================================
# CORS Configuration for Assets Bucket
# =============================================================================

resource "aws_s3_bucket_cors_configuration" "assets" {
  count = var.create_assets_bucket ? 1 : 0

  bucket = aws_s3_bucket.main["assets"].id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3600
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "recordings_bucket_name" {
  value = var.create_recordings_bucket ? aws_s3_bucket.main["recordings"].id : null
}

output "recordings_bucket_arn" {
  value = var.create_recordings_bucket ? aws_s3_bucket.main["recordings"].arn : null
}

output "transcripts_bucket_name" {
  value = var.create_transcripts_bucket ? aws_s3_bucket.main["transcripts"].id : null
}

output "transcripts_bucket_arn" {
  value = var.create_transcripts_bucket ? aws_s3_bucket.main["transcripts"].arn : null
}

output "documents_bucket_name" {
  value = var.create_documents_bucket ? aws_s3_bucket.main["documents"].id : null
}

output "documents_bucket_arn" {
  value = var.create_documents_bucket ? aws_s3_bucket.main["documents"].arn : null
}

output "assets_bucket_name" {
  value = var.create_assets_bucket ? aws_s3_bucket.main["assets"].id : null
}

output "assets_bucket_arn" {
  value = var.create_assets_bucket ? aws_s3_bucket.main["assets"].arn : null
}

output "kms_key_arn" {
  value = var.enable_encryption ? aws_kms_key.s3[0].arn : null
}

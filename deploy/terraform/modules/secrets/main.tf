# Builder Engine - Secrets Manager Module

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
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

variable "secrets" {
  type = map(object({
    description = string
    values      = map(string)
  }))
  default = {}
}

variable "additional_secrets" {
  type    = list(string)
  default = []
}

variable "tags" {
  type    = map(string)
  default = {}
}

# =============================================================================
# KMS Key for Secrets
# =============================================================================

resource "aws_kms_key" "secrets" {
  description             = "KMS key for ${var.name_prefix} secrets"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-secrets-key"
  })
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/${var.name_prefix}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

# =============================================================================
# Secrets with Values
# =============================================================================

resource "aws_secretsmanager_secret" "main" {
  for_each = var.secrets

  name        = "${var.name_prefix}/${each.key}"
  description = each.value.description
  kms_key_id  = aws_kms_key.secrets.arn

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "main" {
  for_each = var.secrets

  secret_id     = aws_secretsmanager_secret.main[each.key].id
  secret_string = jsonencode(each.value.values)
}

# =============================================================================
# Additional Secrets (placeholders)
# =============================================================================

resource "aws_secretsmanager_secret" "additional" {
  for_each = toset(var.additional_secrets)

  name        = "${var.name_prefix}/${each.value}"
  description = "Secret for ${each.value}"
  kms_key_id  = aws_kms_key.secrets.arn

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "additional" {
  for_each = toset(var.additional_secrets)

  secret_id     = aws_secretsmanager_secret.additional[each.value].id
  secret_string = jsonencode({ "placeholder" = "UPDATE_THIS_VALUE" })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

# =============================================================================
# IAM Policy for Secret Access
# =============================================================================

resource "aws_iam_policy" "secrets_read" {
  name        = "${var.name_prefix}-secrets-read"
  description = "Policy for reading ${var.name_prefix} secrets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = concat(
          [for s in aws_secretsmanager_secret.main : s.arn],
          [for s in aws_secretsmanager_secret.additional : s.arn]
        )
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = [aws_kms_key.secrets.arn]
      }
    ]
  })

  tags = var.tags
}

# =============================================================================
# Outputs
# =============================================================================

output "kms_key_arn" {
  value = aws_kms_key.secrets.arn
}

output "secret_arns" {
  value = merge(
    { for k, v in aws_secretsmanager_secret.main : k => v.arn },
    { for k, v in aws_secretsmanager_secret.additional : k => v.arn }
  )
}

output "secrets_read_policy_arn" {
  value = aws_iam_policy.secrets_read.arn
}

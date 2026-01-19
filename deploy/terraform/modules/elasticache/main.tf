# Builder Engine - ElastiCache Module

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

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "node_type" {
  type = string
}

variable "num_cache_nodes" {
  type = number
}

variable "engine_version" {
  type = string
}

variable "automatic_failover_enabled" {
  type    = bool
  default = true
}

variable "multi_az_enabled" {
  type    = bool
  default = true
}

variable "allowed_security_groups" {
  type = list(string)
}

variable "tags" {
  type    = map(string)
  default = {}
}

# =============================================================================
# Security Group
# =============================================================================

resource "aws_security_group" "redis" {
  name_prefix = "${var.name_prefix}-redis-"
  vpc_id      = var.vpc_id
  description = "Security group for ElastiCache Redis"

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-redis-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group_rule" "redis_ingress" {
  count = length(var.allowed_security_groups)

  security_group_id        = aws_security_group.redis.id
  type                     = "ingress"
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  source_security_group_id = var.allowed_security_groups[count.index]
}

# =============================================================================
# Subnet Group
# =============================================================================

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.name_prefix}-redis"
  subnet_ids = var.subnet_ids

  tags = var.tags
}

# =============================================================================
# Parameter Group
# =============================================================================

resource "aws_elasticache_parameter_group" "main" {
  name   = "${var.name_prefix}-redis-params"
  family = "redis7"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "notify-keyspace-events"
    value = "Ex"
  }

  tags = var.tags
}

# =============================================================================
# Replication Group (Redis Cluster)
# =============================================================================

resource "aws_elasticache_replication_group" "main" {
  replication_group_id = "${var.name_prefix}-redis"
  description         = "Redis cluster for ${var.name_prefix}"

  node_type                  = var.node_type
  num_cache_clusters         = var.num_cache_nodes
  parameter_group_name       = aws_elasticache_parameter_group.main.name
  engine_version            = var.engine_version
  port                      = 6379

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  automatic_failover_enabled = var.automatic_failover_enabled && var.num_cache_nodes > 1
  multi_az_enabled          = var.multi_az_enabled && var.num_cache_nodes > 1

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  kms_key_id                = aws_kms_key.redis.arn

  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  maintenance_window      = "Mon:05:00-Mon:07:00"

  auto_minor_version_upgrade = true

  tags = var.tags

  lifecycle {
    ignore_changes = [num_cache_clusters]
  }
}

# =============================================================================
# KMS Key
# =============================================================================

resource "aws_kms_key" "redis" {
  description             = "Redis encryption key for ${var.name_prefix}"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-redis-key"
  })
}

resource "aws_kms_alias" "redis" {
  name          = "alias/${var.name_prefix}-redis"
  target_key_id = aws_kms_key.redis.key_id
}

# =============================================================================
# Outputs
# =============================================================================

output "cluster_id" {
  value = aws_elasticache_replication_group.main.id
}

output "primary_endpoint" {
  value = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "reader_endpoint" {
  value = aws_elasticache_replication_group.main.reader_endpoint_address
}

output "port" {
  value = aws_elasticache_replication_group.main.port
}

output "security_group_id" {
  value = aws_security_group.redis.id
}

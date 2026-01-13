# Builder Engine - Terraform Variables

# =============================================================================
# General Configuration
# =============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

# =============================================================================
# Networking
# =============================================================================

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

# =============================================================================
# EKS Configuration
# =============================================================================

variable "eks_cluster_version" {
  description = "EKS cluster version"
  type        = string
  default     = "1.28"
}

variable "eks_node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    min_size       = number
    max_size       = number
    desired_size   = number
    disk_size      = number
    labels         = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      instance_types = ["m6i.large", "m6i.xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 2
      max_size       = 10
      desired_size   = 3
      disk_size      = 100
      labels = {
        workload = "general"
      }
      taints = []
    }
    voice = {
      instance_types = ["c6i.xlarge", "c6i.2xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 5
      desired_size   = 2
      disk_size      = 50
      labels = {
        workload = "voice"
      }
      taints = [{
        key    = "workload"
        value  = "voice"
        effect = "NO_SCHEDULE"
      }]
    }
    spot = {
      instance_types = ["m6i.large", "m6i.xlarge", "m5.large", "m5.xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 20
      desired_size   = 2
      disk_size      = 100
      labels = {
        workload = "batch"
      }
      taints = [{
        key    = "spotInstance"
        value  = "true"
        effect = "PREFER_NO_SCHEDULE"
      }]
    }
  }
}

# =============================================================================
# RDS Configuration
# =============================================================================

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "16.1"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

# =============================================================================
# ElastiCache Configuration
# =============================================================================

variable "elasticache_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "elasticache_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.1"
}

# =============================================================================
# Domain & SSL
# =============================================================================

variable "domain_name" {
  description = "Primary domain name"
  type        = string
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for CloudFront"
  type        = string
  default     = ""
}

# =============================================================================
# Monitoring
# =============================================================================

variable "alarm_notification_email" {
  description = "Email for CloudWatch alarm notifications"
  type        = string
}

# =============================================================================
# Feature Flags
# =============================================================================

variable "enable_waf" {
  description = "Enable AWS WAF"
  type        = bool
  default     = true
}

variable "enable_shield" {
  description = "Enable AWS Shield Advanced"
  type        = bool
  default     = false
}

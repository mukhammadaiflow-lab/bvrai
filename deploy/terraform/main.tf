# Builder Engine - AWS Infrastructure
# Main Terraform configuration

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "s3" {
    # Configure in environments/*/backend.tf
  }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "builderengine"
      Environment = var.environment
      ManagedBy   = "terraform"
      Repository  = "builderengine/infrastructure"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# =============================================================================
# Local Variables
# =============================================================================

locals {
  name_prefix = "builderengine-${var.environment}"

  azs = slice(data.aws_availability_zones.available.names, 0, 3)

  common_tags = {
    Project     = "builderengine"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# =============================================================================
# VPC Module
# =============================================================================

module "vpc" {
  source = "./modules/vpc"

  name_prefix = local.name_prefix
  environment = var.environment

  vpc_cidr           = var.vpc_cidr
  availability_zones = local.azs

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment == "dev"
  enable_dns_hostnames   = true
  enable_dns_support     = true

  tags = local.common_tags
}

# =============================================================================
# EKS Module
# =============================================================================

module "eks" {
  source = "./modules/eks"

  name_prefix = local.name_prefix
  environment = var.environment

  cluster_version = var.eks_cluster_version

  vpc_id          = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnet_ids

  node_groups = var.eks_node_groups

  enable_cluster_autoscaler = true
  enable_metrics_server     = true
  enable_aws_load_balancer_controller = true
  enable_external_dns       = true
  enable_cert_manager       = true

  tags = local.common_tags
}

# =============================================================================
# RDS Module (PostgreSQL)
# =============================================================================

module "rds" {
  source = "./modules/rds"

  name_prefix = local.name_prefix
  environment = var.environment

  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.database_subnet_ids

  engine_version    = var.rds_engine_version
  instance_class    = var.rds_instance_class
  allocated_storage = var.rds_allocated_storage

  database_name   = "builderengine"
  master_username = "builderengine"

  multi_az               = var.environment != "dev"
  deletion_protection    = var.environment == "production"
  skip_final_snapshot    = var.environment == "dev"
  backup_retention_period = var.environment == "production" ? 30 : 7

  allowed_security_groups = [module.eks.node_security_group_id]

  tags = local.common_tags
}

# =============================================================================
# ElastiCache Module (Redis)
# =============================================================================

module "elasticache" {
  source = "./modules/elasticache"

  name_prefix = local.name_prefix
  environment = var.environment

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.elasticache_subnet_ids

  node_type       = var.elasticache_node_type
  num_cache_nodes = var.environment == "production" ? 3 : 1
  engine_version  = var.elasticache_engine_version

  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled          = var.environment == "production"

  allowed_security_groups = [module.eks.node_security_group_id]

  tags = local.common_tags
}

# =============================================================================
# S3 Module
# =============================================================================

module "s3" {
  source = "./modules/s3"

  name_prefix = local.name_prefix
  environment = var.environment
  account_id  = data.aws_caller_identity.current.account_id

  create_recordings_bucket   = true
  create_transcripts_bucket  = true
  create_documents_bucket    = true
  create_assets_bucket       = true

  enable_versioning = var.environment == "production"
  enable_encryption = true

  lifecycle_rules = {
    recordings = {
      transition_glacier_days = 90
      expiration_days        = 365
    }
    transcripts = {
      transition_ia_days     = 30
      transition_glacier_days = 180
      expiration_days        = 730
    }
  }

  tags = local.common_tags
}

# =============================================================================
# CloudFront Module
# =============================================================================

module "cloudfront" {
  source = "./modules/cloudfront"

  name_prefix = local.name_prefix
  environment = var.environment

  domain_name     = var.domain_name
  certificate_arn = var.acm_certificate_arn

  api_origin = {
    domain_name = module.eks.alb_dns_name
    origin_id   = "api"
  }

  s3_origins = {
    assets = {
      bucket_name = module.s3.assets_bucket_name
      origin_id   = "assets"
    }
  }

  enable_waf = var.environment == "production"

  tags = local.common_tags
}

# =============================================================================
# Secrets Manager Module
# =============================================================================

module "secrets" {
  source = "./modules/secrets"

  name_prefix = local.name_prefix
  environment = var.environment

  secrets = {
    "database" = {
      description = "PostgreSQL credentials"
      values = {
        host     = module.rds.endpoint
        port     = module.rds.port
        database = "builderengine"
        username = module.rds.master_username
        password = module.rds.master_password
      }
    }
    "redis" = {
      description = "Redis connection details"
      values = {
        host = module.elasticache.primary_endpoint
        port = module.elasticache.port
      }
    }
  }

  # Additional secrets to create (values provided separately)
  additional_secrets = [
    "openai-api-key",
    "anthropic-api-key",
    "elevenlabs-api-key",
    "twilio-credentials",
    "stripe-credentials",
    "jwt-secret"
  ]

  tags = local.common_tags
}

# =============================================================================
# Monitoring Module
# =============================================================================

module "monitoring" {
  source = "./modules/monitoring"

  name_prefix = local.name_prefix
  environment = var.environment

  cluster_name = module.eks.cluster_name

  enable_container_insights = true
  enable_cloudwatch_alarms  = true
  enable_sns_notifications  = true

  alarm_email = var.alarm_notification_email

  dashboard_widgets = {
    eks_cluster     = true
    rds_database    = true
    elasticache     = true
    api_latency     = true
    call_metrics    = true
  }

  rds_identifier         = module.rds.identifier
  elasticache_cluster_id = module.elasticache.cluster_id

  tags = local.common_tags
}

# =============================================================================
# Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.endpoint
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = module.elasticache.primary_endpoint
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    recordings  = module.s3.recordings_bucket_name
    transcripts = module.s3.transcripts_bucket_name
    documents   = module.s3.documents_bucket_name
    assets      = module.s3.assets_bucket_name
  }
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = module.cloudfront.distribution_id
}

output "cloudfront_domain_name" {
  description = "CloudFront domain name"
  value       = module.cloudfront.domain_name
}

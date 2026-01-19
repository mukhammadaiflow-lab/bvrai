# Builder Engine - Production Environment

terraform {
  backend "s3" {
    bucket         = "builderengine-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "builderengine-terraform-locks"
  }
}

module "infrastructure" {
  source = "../../"

  aws_region  = "us-east-1"
  environment = "production"

  # Networking
  vpc_cidr = "10.0.0.0/16"

  # EKS Configuration
  eks_cluster_version = "1.28"
  eks_node_groups = {
    general = {
      instance_types = ["m6i.xlarge", "m6i.2xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 3
      max_size       = 20
      desired_size   = 5
      disk_size      = 100
      labels = {
        workload = "general"
      }
      taints = []
    }
    voice = {
      instance_types = ["c6i.2xlarge", "c6i.4xlarge"]
      capacity_type  = "ON_DEMAND"
      min_size       = 2
      max_size       = 10
      desired_size   = 3
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
      instance_types = ["m6i.xlarge", "m6i.2xlarge", "m5.xlarge", "m5.2xlarge"]
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = 50
      desired_size   = 5
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

  # RDS Configuration
  rds_engine_version    = "16.1"
  rds_instance_class    = "db.r6g.xlarge"
  rds_allocated_storage = 500

  # ElastiCache Configuration
  elasticache_node_type      = "cache.r6g.large"
  elasticache_engine_version = "7.1"

  # Domain & SSL
  domain_name         = "api.builderengine.io"
  acm_certificate_arn = "arn:aws:acm:us-east-1:ACCOUNT_ID:certificate/CERTIFICATE_ID"

  # Monitoring
  alarm_notification_email = "ops@builderengine.io"

  # Security
  enable_waf    = true
  enable_shield = true
}

# Builder Engine - AWS Infrastructure (Terraform)

This directory contains Terraform configurations for deploying Builder Engine to AWS.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         CloudFront                                   │   │
│  │                    (CDN + WAF + Shield)                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────┼────────────────────────────────────┐  │
│  │                              VPC                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Public Subnets                                │ │  │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                   │ │  │
│  │  │  │ NAT GW 1  │  │ NAT GW 2  │  │ NAT GW 3  │                   │ │  │
│  │  │  └───────────┘  └───────────┘  └───────────┘                   │ │  │
│  │  │                     ALB (Application Load Balancer)             │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Private Subnets                               │ │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐   │ │  │
│  │  │  │                     EKS Cluster                          │   │ │  │
│  │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐   │   │ │  │
│  │  │  │  │   API   │  │ Workers │  │  Voice  │  │ Scheduler│   │   │ │  │
│  │  │  │  └─────────┘  └─────────┘  └─────────┘  └──────────┘   │   │ │  │
│  │  │  └─────────────────────────────────────────────────────────┘   │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Database Subnets                              │ │  │
│  │  │  ┌─────────────────────────┐  ┌─────────────────────────┐      │ │  │
│  │  │  │       RDS PostgreSQL    │  │      ElastiCache Redis  │      │ │  │
│  │  │  │       (Multi-AZ)        │  │       (Cluster Mode)    │      │ │  │
│  │  │  └─────────────────────────┘  └─────────────────────────┘      │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐   │
│  │    S3 Buckets        │  │ Secrets Manager  │  │     CloudWatch      │   │
│  │ (Recordings, Assets) │  │   (Credentials)  │  │   (Logs & Metrics)  │   │
│  └──────────────────────┘  └──────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Terraform >= 1.6.0
- AWS CLI configured with appropriate credentials
- S3 bucket and DynamoDB table for state management

## Quick Start

```bash
# 1. Initialize Terraform (from environment directory)
cd environments/production
terraform init

# 2. Review the plan
terraform plan -out=plan.tfplan

# 3. Apply the configuration
terraform apply plan.tfplan
```

## Directory Structure

```
terraform/
├── main.tf              # Root module configuration
├── variables.tf         # Variable definitions
├── modules/
│   ├── vpc/            # VPC, subnets, NAT gateways
│   ├── eks/            # EKS cluster, node groups
│   ├── rds/            # PostgreSQL RDS instance
│   ├── elasticache/    # Redis ElastiCache cluster
│   ├── s3/             # S3 buckets with lifecycle policies
│   ├── cloudfront/     # CDN distribution with WAF
│   ├── secrets/        # AWS Secrets Manager
│   └── monitoring/     # CloudWatch dashboards & alarms
└── environments/
    ├── dev/            # Development environment
    ├── staging/        # Staging environment
    └── production/     # Production environment
```

## Modules

### VPC Module
- Creates VPC with public, private, database, and cache subnets
- NAT Gateways (single for dev, multi-AZ for production)
- VPC Endpoints for AWS services (S3, ECR, Secrets Manager)
- Network ACLs and security groups

### EKS Module
- EKS cluster with encryption at rest
- Managed node groups (general, voice, spot)
- IRSA (IAM Roles for Service Accounts)
- Add-ons: VPC CNI, CoreDNS, kube-proxy, EBS CSI

### RDS Module
- PostgreSQL 16 with encryption
- Multi-AZ for production
- Automated backups with configurable retention
- Performance Insights enabled
- Custom parameter groups

### ElastiCache Module
- Redis 7.1 cluster
- Encryption at rest and in transit
- Automatic failover for production
- Snapshot retention

### S3 Module
- Recordings bucket (with Glacier lifecycle)
- Transcripts bucket (with IA/Glacier lifecycle)
- Documents bucket (for knowledge base)
- Assets bucket (with CloudFront origin access)
- All buckets encrypted with KMS

### CloudFront Module
- Distribution with custom domain
- Origin Access Control for S3
- WAF integration with managed rules
- Rate limiting

### Secrets Module
- Secrets Manager for credentials
- KMS encryption
- IAM policy for EKS access

### Monitoring Module
- CloudWatch dashboards
- Metric alarms for RDS, Redis, EKS
- SNS notifications
- Log groups with retention

## Environment Configuration

### Development
```hcl
environment           = "dev"
rds_instance_class    = "db.t3.medium"
elasticache_node_type = "cache.t3.medium"
single_nat_gateway    = true
```

### Production
```hcl
environment           = "production"
rds_instance_class    = "db.r6g.xlarge"
elasticache_node_type = "cache.r6g.large"
multi_az              = true
enable_waf            = true
enable_shield         = true
```

## State Management

Backend configuration uses S3 with DynamoDB locking:

```hcl
terraform {
  backend "s3" {
    bucket         = "builderengine-terraform-state"
    key            = "<environment>/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "builderengine-terraform-locks"
  }
}
```

Create the state bucket and lock table:
```bash
# Create S3 bucket
aws s3api create-bucket \
  --bucket builderengine-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket builderengine-terraform-state \
  --versioning-configuration Status=Enabled

# Create DynamoDB table
aws dynamodb create-table \
  --table-name builderengine-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

## Cost Optimization

### Spot Instances
The EKS configuration includes spot instance node groups for batch workloads:
- Campaigns processing
- Transcription (non-real-time)
- Analytics processing

### Reserved Capacity
For production, consider:
- RDS Reserved Instances (1-3 year)
- ElastiCache Reserved Nodes
- EC2 Savings Plans for EKS nodes

### Right-sizing
Monitor CloudWatch metrics and adjust:
- EKS node group instance types
- RDS instance class
- ElastiCache node type

## Security

### Encryption
- All data encrypted at rest (S3, RDS, ElastiCache, EKS secrets)
- KMS customer managed keys
- TLS for all data in transit

### Network Security
- Private subnets for all workloads
- VPC Endpoints for AWS services
- Security groups with least privilege
- WAF with managed rule sets

### Access Control
- IRSA for EKS pod permissions
- Secrets Manager for credentials
- No secrets in Terraform state

## Disaster Recovery

### Backups
- RDS: Automated backups, 30-day retention
- ElastiCache: Daily snapshots, 7-day retention
- S3: Versioning enabled, cross-region replication (optional)

### Multi-Region
For multi-region deployment:
1. Deploy infrastructure to secondary region
2. Configure RDS read replica
3. Enable S3 cross-region replication
4. Set up Route 53 failover routing

## Troubleshooting

### EKS Access
```bash
aws eks update-kubeconfig --name builderengine-production-eks
kubectl get nodes
```

### RDS Connection
```bash
# Get credentials from Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id builderengine-production/database \
  --query SecretString --output text | jq
```

### Terraform State
```bash
# List resources
terraform state list

# Show specific resource
terraform state show module.eks.aws_eks_cluster.main

# Import existing resource
terraform import module.vpc.aws_vpc.main vpc-123456
```

## Support

- Documentation: https://docs.builderengine.io
- Infrastructure issues: ops@builderengine.io

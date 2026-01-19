# Builder Engine - Monitoring Module

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

variable "cluster_name" {
  type = string
}

variable "enable_container_insights" {
  type    = bool
  default = true
}

variable "enable_cloudwatch_alarms" {
  type    = bool
  default = true
}

variable "enable_sns_notifications" {
  type    = bool
  default = true
}

variable "alarm_email" {
  type = string
}

variable "dashboard_widgets" {
  type = map(bool)
  default = {}
}

variable "rds_identifier" {
  type = string
}

variable "elasticache_cluster_id" {
  type = string
}

variable "tags" {
  type    = map(string)
  default = {}
}

# =============================================================================
# SNS Topic for Alerts
# =============================================================================

resource "aws_sns_topic" "alerts" {
  count = var.enable_sns_notifications ? 1 : 0

  name = "${var.name_prefix}-alerts"

  tags = var.tags
}

resource "aws_sns_topic_subscription" "email" {
  count = var.enable_sns_notifications && var.alarm_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# =============================================================================
# CloudWatch Log Groups
# =============================================================================

resource "aws_cloudwatch_log_group" "api" {
  name              = "/builderengine/${var.environment}/api"
  retention_in_days = 30

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/builderengine/${var.environment}/worker"
  retention_in_days = 30

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "voice" {
  name              = "/builderengine/${var.environment}/voice"
  retention_in_days = 30

  tags = var.tags
}

# =============================================================================
# CloudWatch Alarms
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-rds-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "RDS CPU utilization is above 80%"

  dimensions = {
    DBInstanceIdentifier = var.rds_identifier
  }

  alarm_actions = var.enable_sns_notifications ? [aws_sns_topic.alerts[0].arn] : []
  ok_actions    = var.enable_sns_notifications ? [aws_sns_topic.alerts[0].arn] : []

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "rds_connections" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-rds-high-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 400
  alarm_description   = "RDS connections are above 400"

  dimensions = {
    DBInstanceIdentifier = var.rds_identifier
  }

  alarm_actions = var.enable_sns_notifications ? [aws_sns_topic.alerts[0].arn] : []

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-redis-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Redis CPU utilization is above 80%"

  dimensions = {
    CacheClusterId = var.elasticache_cluster_id
  }

  alarm_actions = var.enable_sns_notifications ? [aws_sns_topic.alerts[0].arn] : []

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  count = var.enable_cloudwatch_alarms ? 1 : 0

  alarm_name          = "${var.name_prefix}-redis-high-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Redis memory usage is above 80%"

  dimensions = {
    CacheClusterId = var.elasticache_cluster_id
  }

  alarm_actions = var.enable_sns_notifications ? [aws_sns_topic.alerts[0].arn] : []

  tags = var.tags
}

# =============================================================================
# CloudWatch Dashboard
# =============================================================================

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.name_prefix}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "RDS Performance"
          view   = "timeSeries"
          stacked = false
          region = data.aws_region.current.name
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", var.rds_identifier],
            [".", "DatabaseConnections", ".", "."],
            [".", "ReadLatency", ".", "."],
            [".", "WriteLatency", ".", "."]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Redis Performance"
          view   = "timeSeries"
          stacked = false
          region = data.aws_region.current.name
          metrics = [
            ["AWS/ElastiCache", "CPUUtilization", "CacheClusterId", var.elasticache_cluster_id],
            [".", "DatabaseMemoryUsagePercentage", ".", "."],
            [".", "CurrConnections", ".", "."],
            [".", "CacheHitRate", ".", "."]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "EKS Cluster Metrics"
          view   = "timeSeries"
          stacked = false
          region = data.aws_region.current.name
          metrics = [
            ["ContainerInsights", "node_cpu_utilization", "ClusterName", var.cluster_name],
            [".", "node_memory_utilization", ".", "."],
            [".", "pod_cpu_utilization", ".", "."],
            [".", "pod_memory_utilization", ".", "."]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "API Latency & Requests"
          view   = "timeSeries"
          stacked = false
          region = data.aws_region.current.name
          metrics = [
            ["BuilderEngine", "APILatency", "Environment", var.environment, { stat = "p99" }],
            [".", "APILatency", ".", ".", { stat = "p50" }],
            [".", "APIRequests", ".", ".", { stat = "Sum" }]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 8
        height = 6
        properties = {
          title  = "Call Metrics"
          view   = "singleValue"
          region = data.aws_region.current.name
          metrics = [
            ["BuilderEngine", "ActiveCalls", "Environment", var.environment],
            [".", "TotalCalls", ".", "."],
            [".", "CallSuccessRate", ".", "."]
          ]
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 12
        width  = 8
        height = 6
        properties = {
          title  = "Error Rates"
          view   = "timeSeries"
          stacked = false
          region = data.aws_region.current.name
          metrics = [
            ["BuilderEngine", "API5xxErrors", "Environment", var.environment],
            [".", "API4xxErrors", ".", "."],
            [".", "CallFailures", ".", "."]
          ]
        }
      },
      {
        type   = "log"
        x      = 16
        y      = 12
        width  = 8
        height = 6
        properties = {
          title  = "Recent Errors"
          region = data.aws_region.current.name
          query  = "SOURCE '/builderengine/${var.environment}/api' | fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20"
        }
      }
    ]
  })
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_region" "current" {}

# =============================================================================
# Outputs
# =============================================================================

output "sns_topic_arn" {
  value = var.enable_sns_notifications ? aws_sns_topic.alerts[0].arn : null
}

output "dashboard_name" {
  value = aws_cloudwatch_dashboard.main.dashboard_name
}

output "log_groups" {
  value = {
    api    = aws_cloudwatch_log_group.api.name
    worker = aws_cloudwatch_log_group.worker.name
    voice  = aws_cloudwatch_log_group.voice.name
  }
}

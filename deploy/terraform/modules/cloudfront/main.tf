# Builder Engine - CloudFront Module

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

variable "domain_name" {
  type = string
}

variable "certificate_arn" {
  type = string
}

variable "api_origin" {
  type = object({
    domain_name = string
    origin_id   = string
  })
}

variable "s3_origins" {
  type = map(object({
    bucket_name = string
    origin_id   = string
  }))
  default = {}
}

variable "enable_waf" {
  type    = bool
  default = true
}

variable "tags" {
  type    = map(string)
  default = {}
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_s3_bucket" "origins" {
  for_each = var.s3_origins
  bucket   = each.value.bucket_name
}

# =============================================================================
# Origin Access Control
# =============================================================================

resource "aws_cloudfront_origin_access_control" "s3" {
  for_each = var.s3_origins

  name                              = "${var.name_prefix}-${each.key}-oac"
  description                       = "OAC for ${each.key} bucket"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# =============================================================================
# CloudFront Distribution
# =============================================================================

resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "${var.name_prefix} distribution"
  default_root_object = "index.html"
  price_class         = "PriceClass_100"
  http_version        = "http2and3"

  aliases = var.certificate_arn != "" ? [var.domain_name, "www.${var.domain_name}"] : []

  # API Origin
  origin {
    domain_name = var.api_origin.domain_name
    origin_id   = var.api_origin.origin_id

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      origin_read_timeout    = 60
    }

    custom_header {
      name  = "X-Custom-Header"
      value = "builderengine"
    }
  }

  # S3 Origins
  dynamic "origin" {
    for_each = var.s3_origins
    content {
      domain_name              = data.aws_s3_bucket.origins[origin.key].bucket_regional_domain_name
      origin_id                = origin.value.origin_id
      origin_access_control_id = aws_cloudfront_origin_access_control.s3[origin.key].id
    }
  }

  # Default behavior (API)
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = var.api_origin.origin_id

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Origin", "Access-Control-Request-Headers", "Access-Control-Request-Method"]

      cookies {
        forward = "all"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
    compress               = true
  }

  # Assets behavior
  dynamic "ordered_cache_behavior" {
    for_each = contains(keys(var.s3_origins), "assets") ? [1] : []
    content {
      path_pattern     = "/assets/*"
      allowed_methods  = ["GET", "HEAD", "OPTIONS"]
      cached_methods   = ["GET", "HEAD"]
      target_origin_id = "assets"

      forwarded_values {
        query_string = false
        cookies {
          forward = "none"
        }
      }

      viewer_protocol_policy = "redirect-to-https"
      min_ttl                = 0
      default_ttl            = 86400
      max_ttl                = 31536000
      compress               = true
    }
  }

  # WebSocket behavior
  ordered_cache_behavior {
    path_pattern     = "/ws/*"
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = var.api_origin.origin_id

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies {
        forward = "all"
      }
    }

    viewer_protocol_policy = "https-only"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = var.certificate_arn == ""
    acm_certificate_arn           = var.certificate_arn != "" ? var.certificate_arn : null
    ssl_support_method            = var.certificate_arn != "" ? "sni-only" : null
    minimum_protocol_version      = var.certificate_arn != "" ? "TLSv1.2_2021" : null
  }

  web_acl_id = var.enable_waf ? aws_wafv2_web_acl.main[0].arn : null

  tags = var.tags
}

# =============================================================================
# WAF Web ACL
# =============================================================================

resource "aws_wafv2_web_acl" "main" {
  count = var.enable_waf ? 1 : 0

  name        = "${var.name_prefix}-waf"
  description = "WAF for ${var.name_prefix} CloudFront distribution"
  scope       = "CLOUDFRONT"

  default_action {
    allow {}
  }

  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "AWSManagedRulesCommonRuleSetMetric"
      sampled_requests_enabled  = true
    }
  }

  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "AWSManagedRulesKnownBadInputsRuleSetMetric"
      sampled_requests_enabled  = true
    }
  }

  rule {
    name     = "RateLimitRule"
    priority = 3

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name               = "RateLimitRuleMetric"
      sampled_requests_enabled  = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name               = "${var.name_prefix}-waf"
    sampled_requests_enabled  = true
  }

  tags = var.tags
}

# =============================================================================
# Outputs
# =============================================================================

output "distribution_id" {
  value = aws_cloudfront_distribution.main.id
}

output "distribution_arn" {
  value = aws_cloudfront_distribution.main.arn
}

output "domain_name" {
  value = aws_cloudfront_distribution.main.domain_name
}

output "hosted_zone_id" {
  value = aws_cloudfront_distribution.main.hosted_zone_id
}

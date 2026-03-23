###############################################################################
# Terraform configuration for AWS GPU training and inference infrastructure
#
# Resources:
# - VPC with public/private subnets
# - S3 bucket for checkpoints, datasets, and model artifacts
# - EC2 spot fleet for distributed training (p4d.24xlarge)
# - EC2 instances for inference (g5.2xlarge) with autoscaling
# - IAM roles with least-privilege policies
# - Security groups with minimal exposure
# - CloudWatch alarms for spot interruptions
###############################################################################

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
  backend "s3" {
    bucket = "finetune-pipeline-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# =============================================================================
# Variables
# =============================================================================

variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "finetune-pipeline"
}

variable "environment" {
  default = "production"
}

variable "training_instance_type" {
  default = "p4d.24xlarge"
}

variable "training_spot_max_price" {
  default = "15.00"
}

variable "training_num_instances" {
  default = 1
}

variable "inference_instance_type" {
  default = "g5.2xlarge"
}

variable "inference_min_instances" {
  default = 1
}

variable "inference_max_instances" {
  default = 4
}

variable "ami_id" {
  description = "Deep Learning AMI with CUDA and drivers pre-installed"
  default     = "ami-0b20a6f09f8be7f64" # AWS DL AMI Ubuntu 22.04
}

variable "key_pair_name" {
  description = "SSH key pair for instance access"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed SSH access"
  default     = "0.0.0.0/0"
}

# =============================================================================
# Networking (VPC)
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name    = "${var.project_name}-vpc"
    Project = var.project_name
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = { Name = "${var.project_name}-private-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

data "aws_availability_zones" "available" {
  state = "available"
}

# =============================================================================
# Security Groups
# =============================================================================

resource "aws_security_group" "training" {
  name_prefix = "${var.project_name}-training-"
  vpc_id      = aws_vpc.main.id

  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # NCCL (inter-node GPU communication)
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "udp"
    self      = true
  }

  # Prometheus metrics
  ingress {
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-training-sg" }
}

resource "aws_security_group" "inference" {
  name_prefix = "${var.project_name}-inference-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-inference-sg" }
}

# =============================================================================
# S3 Bucket (checkpoints, datasets, model artifacts)
# =============================================================================

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
  force_destroy = false

  tags = {
    Name    = "${var.project_name}-artifacts"
    Project = var.project_name
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "checkpoints-cleanup"
    status = "Enabled"
    filter { prefix = "checkpoints/" }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }
    expiration { days = 180 }
  }

  rule {
    id     = "logs-cleanup"
    status = "Enabled"
    filter { prefix = "logs/" }
    expiration { days = 90 }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_caller_identity" "current" {}

# =============================================================================
# IAM Roles
# =============================================================================

resource "aws_iam_role" "training" {
  name = "${var.project_name}-training-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "training_s3" {
  name = "${var.project_name}-training-s3"
  role = aws_iam_role.training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ]
        Resource = [
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_instance_profile" "training" {
  name = "${var.project_name}-training-profile"
  role = aws_iam_role.training.name
}

resource "aws_iam_instance_profile" "inference" {
  name = "${var.project_name}-inference-profile"
  role = aws_iam_role.training.name
}

# =============================================================================
# EC2 Spot Fleet — Training
# =============================================================================

resource "aws_spot_fleet_request" "training" {
  iam_fleet_role                      = aws_iam_role.spot_fleet.arn
  target_capacity                     = var.training_num_instances
  allocation_strategy                 = "capacityOptimized"
  terminate_instances_with_expiration = true
  excess_capacity_termination_policy  = "Default"

  launch_specification {
    instance_type          = var.training_instance_type
    ami                    = var.ami_id
    key_name               = var.key_pair_name
    vpc_security_group_ids = [aws_security_group.training.id]
    subnet_id              = aws_subnet.public[0].id
    iam_instance_profile   = aws_iam_instance_profile.training.name
    spot_price             = var.training_spot_max_price

    root_block_device {
      volume_size = 500
      volume_type = "gp3"
      iops        = 6000
      throughput  = 400
    }

    ebs_block_device {
      device_name = "/dev/sdf"
      volume_size = 1000
      volume_type = "gp3"
      iops        = 10000
      throughput  = 700
    }

    user_data = base64encode(templatefile("${path.module}/user_data_training.sh.tpl", {
      s3_bucket   = aws_s3_bucket.artifacts.id
      wandb_key   = "PLACEHOLDER"
      hf_token    = "PLACEHOLDER"
      region      = var.aws_region
    }))

    tags = {
      Name = "${var.project_name}-training-spot"
    }
  }

  tags = {
    Name    = "${var.project_name}-training-fleet"
    Project = var.project_name
  }
}

resource "aws_iam_role" "spot_fleet" {
  name = "${var.project_name}-spot-fleet-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "spotfleet.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "spot_fleet" {
  role       = aws_iam_role.spot_fleet.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

# =============================================================================
# Auto Scaling Group — Inference
# =============================================================================

resource "aws_launch_template" "inference" {
  name_prefix   = "${var.project_name}-inference-"
  image_id      = var.ami_id
  instance_type = var.inference_instance_type
  key_name      = var.key_pair_name

  iam_instance_profile {
    name = aws_iam_instance_profile.inference.name
  }

  network_interfaces {
    security_groups             = [aws_security_group.inference.id]
    associate_public_ip_address = true
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 200
      volume_type = "gp3"
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "${var.project_name}-inference"
      Project = var.project_name
    }
  }
}

resource "aws_autoscaling_group" "inference" {
  name                = "${var.project_name}-inference-asg"
  desired_capacity    = var.inference_min_instances
  min_size            = var.inference_min_instances
  max_size            = var.inference_max_instances
  vpc_zone_identifier = aws_subnet.public[*].id

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 1
      on_demand_percentage_above_base_capacity = 0
      spot_allocation_strategy                 = "capacity-optimized"
    }
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.inference.id
        version            = "$Latest"
      }
      override {
        instance_type = "g5.2xlarge"
      }
      override {
        instance_type = "g5.4xlarge"
      }
    }
  }

  health_check_type         = "EC2"
  health_check_grace_period = 300

  tag {
    key                 = "Project"
    value               = var.project_name
    propagate_at_launch = true
  }
}

resource "aws_autoscaling_policy" "inference_scale_up" {
  name                   = "${var.project_name}-inference-scale-up"
  autoscaling_group_name = aws_autoscaling_group.inference.name
  policy_type            = "TargetTrackingScaling"

  target_tracking_configuration {
    customized_metric_specification {
      metric_dimension {
        name  = "AutoScalingGroupName"
        value = aws_autoscaling_group.inference.name
      }
      metric_name = "CPUUtilization"
      namespace   = "AWS/EC2"
      statistic   = "Average"
    }
    target_value = 70.0
  }
}

# =============================================================================
# CloudWatch — Spot Interruption Alarm
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "spot_interruption" {
  alarm_name          = "${var.project_name}-spot-interruption"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  alarm_description   = "Spot instance interrupted — trigger checkpoint save"

  alarm_actions = []  # Add SNS topic ARN for notifications

  tags = { Project = var.project_name }
}

# =============================================================================
# Outputs
# =============================================================================

output "s3_bucket_name" {
  value = aws_s3_bucket.artifacts.id
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "training_security_group_id" {
  value = aws_security_group.training.id
}

output "inference_security_group_id" {
  value = aws_security_group.inference.id
}

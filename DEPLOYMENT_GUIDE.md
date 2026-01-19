# Builder Voice AI Platform - Production Deployment Guide

Complete step-by-step guide to deploy BVRAI to production with Kubernetes.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Options](#infrastructure-options)
3. [Cloud Provider Setup](#cloud-provider-setup)
4. [Kubernetes Cluster Setup](#kubernetes-cluster-setup)
5. [Database Setup](#database-setup)
6. [Secrets Configuration](#secrets-configuration)
7. [Deploy the Platform](#deploy-the-platform)
8. [DNS & SSL Configuration](#dns--ssl-configuration)
9. [Post-Deployment Verification](#post-deployment-verification)
10. [Monitoring & Observability](#monitoring--observability)
11. [CI/CD Pipeline Setup](#cicd-pipeline-setup)
12. [Scaling & Performance](#scaling--performance)
13. [Backup & Disaster Recovery](#backup--disaster-recovery)
14. [Cost Estimation](#cost-estimation)
15. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

Install these on your local machine:

```bash
# Check versions
kubectl version --client    # v1.28+
helm version               # v3.12+
docker version             # v24+
git version                # v2.40+
```

#### Installation Commands

**macOS:**
```bash
# Using Homebrew
brew install kubectl helm docker git
```

**Linux (Ubuntu/Debian):**
```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# docker
curl -fsSL https://get.docker.com | sh
```

**Windows:**
```powershell
# Using Chocolatey
choco install kubernetes-cli helm docker-desktop git
```

### Required API Keys

You must have these ready before deployment:

| Provider | Purpose | Where to Get |
|----------|---------|--------------|
| Deepgram | Speech-to-Text | [console.deepgram.com](https://console.deepgram.com) |
| ElevenLabs | Text-to-Speech | [elevenlabs.io](https://elevenlabs.io) |
| OpenAI | LLM | [platform.openai.com](https://platform.openai.com) |
| Twilio (optional) | Phone calls | [twilio.com](https://twilio.com) |

---

## Infrastructure Options

### Option A: Managed Kubernetes (Recommended)

| Provider | Service | Monthly Cost (Estimate) | Best For |
|----------|---------|------------------------|----------|
| **AWS** | EKS | $73/month + nodes | Enterprise, AWS ecosystem |
| **Google Cloud** | GKE | $0/month + nodes | Best price, auto-scaling |
| **Azure** | AKS | $0/month + nodes | Microsoft ecosystem |
| **DigitalOcean** | DOKS | $12/month + nodes | Startups, simple setup |

**Recommended:** Google Cloud GKE (free control plane, excellent auto-scaling)

### Option B: Self-Managed Kubernetes

For on-premise or VPS deployment:

| Tool | Purpose |
|------|---------|
| k3s | Lightweight Kubernetes |
| kubeadm | Standard K8s installation |
| Rancher | K8s management UI |

### Minimum Node Requirements

| Component | vCPUs | RAM | Instances |
|-----------|-------|-----|-----------|
| Platform API | 2 | 4GB | 2-3 |
| Voice Engine | 2 | 4GB | 2-3 |
| Conversation Engine | 2 | 4GB | 2-3 |
| Workers | 2 | 4GB | 2 |
| PostgreSQL | 2 | 8GB | 1 (managed recommended) |
| Redis | 1 | 2GB | 1 |

**Minimum Total:** 3 nodes with 4 vCPU, 16GB RAM each

---

## Cloud Provider Setup

### AWS EKS Setup

#### Step 1: Install AWS CLI and eksctl

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (e.g., us-east-1)
```

#### Step 2: Create EKS Cluster

```bash
# Create cluster (takes 15-20 minutes)
eksctl create cluster \
  --name bvrai-production \
  --region us-east-1 \
  --version 1.28 \
  --nodegroup-name bvrai-nodes \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 6 \
  --managed \
  --with-oidc

# Verify cluster
kubectl get nodes
```

#### Step 3: Install AWS Load Balancer Controller

```bash
# Install controller for ingress
eksctl utils associate-iam-oidc-provider --cluster bvrai-production --approve

helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=bvrai-production
```

---

### Google Cloud GKE Setup

#### Step 1: Install gcloud CLI

```bash
# Install gcloud
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize and login
gcloud init
gcloud auth login
```

#### Step 2: Create GKE Cluster

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Create cluster
gcloud container clusters create bvrai-production \
  --zone us-central1-a \
  --machine-type e2-standard-4 \
  --num-nodes 3 \
  --min-nodes 2 \
  --max-nodes 6 \
  --enable-autoscaling \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials bvrai-production --zone us-central1-a

# Verify
kubectl get nodes
```

---

### DigitalOcean DOKS Setup (Simplest)

#### Step 1: Install doctl

```bash
# macOS
brew install doctl

# Linux
curl -sL https://github.com/digitalocean/doctl/releases/download/v1.98.0/doctl-1.98.0-linux-amd64.tar.gz | tar xz
sudo mv doctl /usr/local/bin

# Authenticate
doctl auth init
# Paste your API token from DigitalOcean dashboard
```

#### Step 2: Create Cluster

```bash
# Create cluster
doctl kubernetes cluster create bvrai-production \
  --region nyc1 \
  --size s-4vcpu-8gb \
  --count 3 \
  --auto-upgrade

# Get credentials
doctl kubernetes cluster kubeconfig save bvrai-production

# Verify
kubectl get nodes
```

---

## Database Setup

### Option A: Managed Database (Recommended for Production)

#### AWS RDS PostgreSQL

```bash
# Create RDS instance via AWS Console or CLI
aws rds create-db-instance \
  --db-instance-identifier bvrai-postgres \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 15.4 \
  --master-username bvrai \
  --master-user-password "YOUR_SECURE_PASSWORD" \
  --allocated-storage 100 \
  --storage-type gp3 \
  --multi-az \
  --vpc-security-group-ids sg-xxxxxx \
  --db-subnet-group-name your-subnet-group

# Get endpoint after creation
aws rds describe-db-instances --db-instance-identifier bvrai-postgres \
  --query 'DBInstances[0].Endpoint.Address'
```

#### Google Cloud SQL

```bash
gcloud sql instances create bvrai-postgres \
  --database-version=POSTGRES_15 \
  --tier=db-custom-2-4096 \
  --region=us-central1 \
  --root-password="YOUR_SECURE_PASSWORD" \
  --storage-size=100GB \
  --storage-type=SSD \
  --availability-type=REGIONAL

# Create database
gcloud sql databases create bvrai --instance=bvrai-postgres
```

### Option B: In-Cluster Database

Only for development/testing:

```bash
# Deploy PostgreSQL via Helm
helm repo add bitnami https://charts.bitnami.com/bitnami

helm install postgresql bitnami/postgresql \
  --namespace bvrai \
  --set auth.postgresPassword=YOUR_PASSWORD \
  --set auth.database=bvrai \
  --set primary.persistence.size=50Gi
```

---

## Secrets Configuration

### Step 1: Create Namespace

```bash
kubectl create namespace bvrai
```

### Step 2: Create Secrets

Create a file `secrets.yaml` (DO NOT commit this file):

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: bvrai-secrets
  namespace: bvrai
type: Opaque
stringData:
  # Database
  DATABASE_URL: "postgresql://bvrai:PASSWORD@your-db-host:5432/bvrai"

  # Redis
  REDIS_URL: "redis://redis:6379"

  # JWT & Security
  JWT_SECRET: "your-32-character-random-string-here"
  APP_SECRET_KEY: "another-32-char-random-string"

  # Speech-to-Text
  DEEPGRAM_API_KEY: "your-deepgram-key"

  # Text-to-Speech
  ELEVENLABS_API_KEY: "your-elevenlabs-key"

  # LLM
  OPENAI_API_KEY: "your-openai-key"

  # Telephony (optional)
  TWILIO_ACCOUNT_SID: "your-twilio-sid"
  TWILIO_AUTH_TOKEN: "your-twilio-token"
```

Apply secrets:

```bash
kubectl apply -f secrets.yaml

# Verify
kubectl get secrets -n bvrai
```

### Generate Random Secrets

```bash
# Generate JWT_SECRET
openssl rand -hex 32

# Generate APP_SECRET_KEY
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## Deploy the Platform

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/bvrai.git
cd bvrai
```

### Step 2: Install Ingress Controller

```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer
```

### Step 3: Install cert-manager (for SSL)

```bash
helm repo add jetstack https://charts.jetstack.io
helm repo update

helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true
```

### Step 4: Deploy BVRAI

```bash
# Apply Kubernetes manifests
kubectl apply -k deploy/kubernetes

# Wait for deployments
kubectl -n bvrai rollout status deployment/platform-api
kubectl -n bvrai rollout status deployment/voice-engine
kubectl -n bvrai rollout status deployment/conversation-engine
```

### Step 5: Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n bvrai

# Expected output:
# NAME                                    READY   STATUS    RESTARTS   AGE
# platform-api-xxxxx-xxxxx               1/1     Running   0          2m
# voice-engine-xxxxx-xxxxx               1/1     Running   0          2m
# conversation-engine-xxxxx-xxxxx        1/1     Running   0          2m
# redis-xxxxx-xxxxx                       1/1     Running   0          2m
```

---

## DNS & SSL Configuration

### Step 1: Get Load Balancer IP

```bash
# Get external IP
kubectl get svc -n ingress-nginx ingress-nginx-controller

# Output:
# NAME                       TYPE           EXTERNAL-IP     PORT(S)
# ingress-nginx-controller   LoadBalancer   203.0.113.50   80:30080/TCP,443:30443/TCP
```

### Step 2: Configure DNS Records

In your DNS provider (Cloudflare, Route53, etc.), create these A records:

| Subdomain | Type | Value | TTL |
|-----------|------|-------|-----|
| api.yourdomain.com | A | 203.0.113.50 | 300 |
| app.yourdomain.com | A | 203.0.113.50 | 300 |
| ws.yourdomain.com | A | 203.0.113.50 | 300 |

### Step 3: Update Ingress Configuration

Edit `deploy/kubernetes/ingress.yaml` with your domain:

```yaml
spec:
  tls:
    - hosts:
        - api.yourdomain.com
        - app.yourdomain.com
        - ws.yourdomain.com
      secretName: bvrai-tls
  rules:
    - host: api.yourdomain.com
      # ... rest of config
```

Apply changes:

```bash
kubectl apply -f deploy/kubernetes/ingress.yaml
```

### Step 4: Verify SSL

Wait 2-5 minutes for Let's Encrypt certificate:

```bash
# Check certificate status
kubectl get certificate -n bvrai

# Test HTTPS
curl -I https://api.yourdomain.com/health
```

---

## Post-Deployment Verification

### Health Check Script

```bash
#!/bin/bash
# save as verify-deployment.sh

API_URL="${1:-https://api.yourdomain.com}"

echo "=== BVRAI Deployment Verification ==="

# 1. API Health
echo -n "API Health: "
if curl -sf "$API_URL/health" > /dev/null; then
  echo "✅ OK"
else
  echo "❌ FAILED"
fi

# 2. API Version
echo -n "API Version: "
VERSION=$(curl -sf "$API_URL/api/v1/version" | jq -r '.version')
echo "$VERSION"

# 3. Database Connection
echo -n "Database: "
DB_STATUS=$(curl -sf "$API_URL/health/db" | jq -r '.status')
if [ "$DB_STATUS" = "healthy" ]; then
  echo "✅ Connected"
else
  echo "❌ Not Connected"
fi

# 4. Redis Connection
echo -n "Redis: "
REDIS_STATUS=$(curl -sf "$API_URL/health/redis" | jq -r '.status')
if [ "$REDIS_STATUS" = "healthy" ]; then
  echo "✅ Connected"
else
  echo "❌ Not Connected"
fi

echo "=== Verification Complete ==="
```

Run verification:

```bash
chmod +x verify-deployment.sh
./verify-deployment.sh https://api.yourdomain.com
```

---

## Monitoring & Observability

### Install Prometheus & Grafana

```bash
# Add Prometheus community Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack (includes Grafana)
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=your-secure-password
```

### Access Grafana Dashboard

```bash
# Port forward to access locally
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80

# Open in browser: http://localhost:3000
# Username: admin
# Password: your-secure-password
```

### Import BVRAI Dashboards

The monitoring dashboards are in `deploy/kubernetes/monitoring/grafana-dashboards.yaml`.

---

## CI/CD Pipeline Setup

### GitHub Actions Setup

1. **Add Repository Secrets** (Settings → Secrets → Actions):

| Secret Name | Description |
|-------------|-------------|
| `KUBE_CONFIG_STAGING` | Base64-encoded kubeconfig for staging |
| `KUBE_CONFIG_PRODUCTION` | Base64-encoded kubeconfig for production |
| `SLACK_WEBHOOK` | Slack webhook for notifications |
| `SMOKE_TEST_API_KEY` | API key for smoke tests |

2. **Generate kubeconfig secret:**

```bash
# Get kubeconfig and encode
cat ~/.kube/config | base64 -w 0
# Copy output to GitHub secrets
```

3. **Push to trigger deployment:**

```bash
git add .
git commit -m "feat: Deploy to production"
git tag v1.0.0
git push origin main --tags
```

---

## Scaling & Performance

### Horizontal Pod Autoscaler

```bash
# Auto-scale platform API (2-10 replicas based on CPU)
kubectl autoscale deployment platform-api \
  -n bvrai \
  --min=2 \
  --max=10 \
  --cpu-percent=70
```

### Manual Scaling

```bash
# Scale to specific replica count
kubectl scale deployment platform-api -n bvrai --replicas=5
```

### Recommended Production Settings

| Service | Min Replicas | Max Replicas | CPU Request | Memory Request |
|---------|--------------|--------------|-------------|----------------|
| platform-api | 3 | 10 | 500m | 1Gi |
| voice-engine | 3 | 10 | 500m | 1Gi |
| conversation-engine | 3 | 10 | 500m | 1Gi |
| workers | 2 | 5 | 250m | 512Mi |

---

## Backup & Disaster Recovery

### Database Backup

**AWS RDS:**
```bash
# Enable automated backups (7 days retention)
aws rds modify-db-instance \
  --db-instance-identifier bvrai-postgres \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00"

# Manual snapshot
aws rds create-db-snapshot \
  --db-instance-identifier bvrai-postgres \
  --db-snapshot-identifier bvrai-backup-$(date +%Y%m%d)
```

**Google Cloud SQL:**
```bash
# Enable automated backups
gcloud sql instances patch bvrai-postgres \
  --backup-start-time="03:00" \
  --retained-backups-count=7

# Manual backup
gcloud sql backups create --instance=bvrai-postgres
```

### Disaster Recovery Plan

1. **Database Recovery:**
   - Restore from latest backup
   - Point-in-time recovery (RDS/Cloud SQL)

2. **Application Recovery:**
   ```bash
   # Redeploy from Git
   kubectl apply -k deploy/kubernetes
   ```

3. **Multi-Region Setup (Advanced):**
   - Deploy to multiple regions
   - Use global load balancer
   - Database replication

---

## Cost Estimation

### Monthly Cost Breakdown

| Component | AWS (EKS) | GCP (GKE) | DigitalOcean |
|-----------|-----------|-----------|--------------|
| Kubernetes Control Plane | $73 | $0 | $0 |
| 3x Compute Nodes (4vCPU, 16GB) | ~$200 | ~$180 | ~$150 |
| Managed PostgreSQL | ~$50 | ~$50 | ~$50 |
| Load Balancer | ~$20 | ~$20 | ~$12 |
| **Total Infrastructure** | **~$343** | **~$250** | **~$212** |

### API Costs (Variable)

| Provider | Unit | Cost |
|----------|------|------|
| Deepgram | per hour audio | $0.25 |
| ElevenLabs | per 1K characters | $0.30 |
| OpenAI GPT-4o | per 1M tokens | $5.00 |
| Twilio | per minute | $0.014 |

**Example:** 1,000 calls/day × 3 min avg = ~$100/month in API costs

---

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check pod status
kubectl describe pod POD_NAME -n bvrai

# Check logs
kubectl logs POD_NAME -n bvrai

# Common fixes:
# - Check secrets are correctly configured
# - Verify image pull secrets
# - Check resource limits
```

#### Database Connection Issues

```bash
# Test database connectivity from pod
kubectl exec -it deployment/platform-api -n bvrai -- \
  psql "$DATABASE_URL" -c "SELECT 1"

# Common fixes:
# - Check DATABASE_URL secret
# - Verify security groups/firewall rules
# - Ensure database is accessible from cluster
```

#### SSL Certificate Issues

```bash
# Check certificate status
kubectl describe certificate bvrai-tls -n bvrai

# Check cert-manager logs
kubectl logs -n cert-manager -l app=cert-manager

# Common fixes:
# - Verify DNS is pointing to load balancer
# - Check ingress configuration
# - Wait for DNS propagation (up to 48 hours)
```

#### High Latency

```bash
# Check pod resource usage
kubectl top pods -n bvrai

# Scale up if needed
kubectl scale deployment platform-api -n bvrai --replicas=5

# Check HPA status
kubectl get hpa -n bvrai
```

---

## Quick Reference Commands

```bash
# View all pods
kubectl get pods -n bvrai

# View logs (follow)
kubectl logs -f deployment/platform-api -n bvrai

# Shell into pod
kubectl exec -it deployment/platform-api -n bvrai -- /bin/bash

# Restart deployment
kubectl rollout restart deployment/platform-api -n bvrai

# Check deployment status
kubectl rollout status deployment/platform-api -n bvrai

# View secrets
kubectl get secrets -n bvrai

# Port forward for local testing
kubectl port-forward svc/platform-api -n bvrai 8086:8000
```

---

## Support

- **Documentation:** `/docs` folder in repository
- **Issues:** GitHub Issues
- **API Reference:** https://api.yourdomain.com/docs

---

## Deployment Checklist

Before going live, verify:

- [ ] All secrets configured
- [ ] Database migrations run
- [ ] SSL certificate issued
- [ ] DNS records configured
- [ ] Health endpoints responding
- [ ] Monitoring dashboards accessible
- [ ] Backups configured
- [ ] CI/CD pipeline tested
- [ ] Load testing completed
- [ ] Documentation updated

# Cloud Deployment Guide for Khipu Viewer

This guide provides detailed instructions for deploying the cloud-ready Khipu viewer to various cloud platforms.

## Prerequisites

- Docker installed locally (for building and testing)
- Cloud provider account (Azure, AWS, or Google Cloud)
- Exported khipu data in JSON format (see [Data Export](#data-export))

## Data Export

Before deployment, export the khipu data from the database to JSON format:

```bash
# Export all khipus
python scripts/export_from_processed.py --output data/blob_export

# Or export a subset for testing
python scripts/export_from_processed.py --output data/blob_export --limit 50
```

This creates:
- `data/blob_export/khipu_index.json` - Index of all khipus
- `data/blob_export/khipus/*.json` - Individual khipu data files
- `data/blob_export/colors.json` - Color mappings

## Local Testing

Test the application locally before deploying:

```bash
cd cloud_viewer

# Option 1: Using Docker Compose (recommended)
docker-compose up --build

# Option 2: Using Python directly
pip install -r requirements.txt
export STORAGE_TYPE=local
export STORAGE_PATH=../data/blob_export
python app.py
```

Access the viewer at: http://localhost:5000

## Azure Container Apps Deployment

### Step 1: Upload Data to Azure Blob Storage

```bash
# Install Azure CLI if needed
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Create resource group
az group create --name khipu-rg --location eastus

# Create storage account
az storage account create \
  --name khipustorage123 \
  --resource-group khipu-rg \
  --location eastus \
  --sku Standard_LRS

# Get connection string
CONNECTION_STRING=$(az storage account show-connection-string \
  --name khipustorage123 \
  --resource-group khipu-rg \
  --query connectionString -o tsv)

# Create container
az storage container create \
  --name khipu-data \
  --account-name khipustorage123 \
  --connection-string "$CONNECTION_STRING"

# Upload data
az storage blob upload-batch \
  --account-name khipustorage123 \
  --destination khipu-data \
  --source ../data/blob_export \
  --connection-string "$CONNECTION_STRING"
```

### Step 2: Create Container Registry

```bash
# Create Azure Container Registry
az acr create \
  --resource-group khipu-rg \
  --name khipuregistry \
  --sku Basic \
  --admin-enabled true

# Login to ACR
az acr login --name khipuregistry
```

### Step 3: Build and Push Container

```bash
# Build and push to ACR
az acr build \
  --registry khipuregistry \
  --image khipu-viewer:latest \
  --file cloud_viewer/Dockerfile \
  .
```

### Step 4: Create Container App Environment

```bash
# Create Container Apps environment
az containerapp env create \
  --name khipu-env \
  --resource-group khipu-rg \
  --location eastus
```

### Step 5: Deploy Container App

```bash
# Deploy the app
az containerapp create \
  --name khipu-viewer \
  --resource-group khipu-rg \
  --environment khipu-env \
  --image khipuregistry.azurecr.io/khipu-viewer:latest \
  --target-port 5000 \
  --ingress external \
  --registry-server khipuregistry.azurecr.io \
  --query properties.configuration.ingress.fqdn \
  --env-vars \
    STORAGE_TYPE=azure \
    STORAGE_PATH=khipu-data \
    AZURE_CONNECTION_STRING="$CONNECTION_STRING"
```

### Step 6: Access Your App

The command above will output the FQDN (e.g., `https://khipu-viewer.nicewater-12345.eastus.azurecontainerapps.io`)

## AWS ECS Deployment

### Step 1: Upload Data to S3

```bash
# Install AWS CLI if needed
pip install awscli

# Configure AWS credentials
aws configure

# Create S3 bucket
aws s3 mb s3://khipu-data-bucket-123

# Upload data
aws s3 sync ../data/blob_export s3://khipu-data-bucket-123/

# Make bucket files readable by the app (set appropriate IAM policies)
```

### Step 2: Create ECR Repository

```bash
# Create ECR repository
aws ecr create-repository --repository-name khipu-viewer

# Get registry URL
REGISTRY_URL=$(aws ecr describe-repositories \
  --repository-names khipu-viewer \
  --query 'repositories[0].repositoryUri' \
  --output text)
```

### Step 3: Build and Push Container

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $REGISTRY_URL

# Build image
docker build -t khipu-viewer -f cloud_viewer/Dockerfile .

# Tag and push
docker tag khipu-viewer:latest $REGISTRY_URL:latest
docker push $REGISTRY_URL:latest
```

### Step 4: Create ECS Task Definition

Create `ecs-task-definition.json`:

```json
{
  "family": "khipu-viewer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "khipu-viewer",
      "image": "YOUR_REGISTRY_URL:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "STORAGE_TYPE", "value": "aws"},
        {"name": "STORAGE_PATH", "value": "khipu-data-bucket-123"},
        {"name": "AWS_REGION", "value": "us-east-1"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/khipu-viewer",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register the task:

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

### Step 5: Create ECS Service

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name khipu-cluster

# Create service (requires VPC, subnets, and security groups)
aws ecs create-service \
  --cluster khipu-cluster \
  --service-name khipu-viewer-service \
  --task-definition khipu-viewer \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

### Step 6: Setup Load Balancer (Optional)

For production deployments, configure an Application Load Balancer to route traffic to your ECS service.

## Google Cloud Run Deployment

### Step 1: Upload Data to Google Cloud Storage

```bash
# Install gcloud CLI
# Follow: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Create bucket
gsutil mb gs://khipu-data-bucket

# Upload data
gsutil -m cp -r ../data/blob_export/* gs://khipu-data-bucket/
```

### Step 2: Build and Deploy

```bash
# Build and deploy in one command
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/khipu-viewer cloud_viewer/

# Deploy to Cloud Run
gcloud run deploy khipu-viewer \
  --image gcr.io/YOUR_PROJECT_ID/khipu-viewer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars STORAGE_TYPE=local,STORAGE_PATH=./data/blob_export \
  --memory 512Mi \
  --cpu 1

# Note: For GCS integration, you'll need to mount the bucket or use GCS client library
```

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `STORAGE_TYPE` | Yes | Storage backend: `local`, `azure`, or `aws` | `azure` |
| `STORAGE_PATH` | Yes | Container/bucket name or local path | `khipu-data` |
| `AZURE_CONNECTION_STRING` | If Azure | Azure Storage connection string | `DefaultEndpoints...` |
| `AWS_ACCESS_KEY_ID` | If AWS | AWS access key | Your AWS key |
| `AWS_SECRET_ACCESS_KEY` | If AWS | AWS secret key | Your AWS secret |
| `AWS_REGION` | If AWS | AWS region | `us-east-1` |
| `PORT` | No | Server port (default: 5000) | `8080` |
| `DEBUG` | No | Debug mode (default: false) | `true` |

## Cost Estimates

### Azure Container Apps
- **Compute**: ~$0.000012/vCPU-second + $0.000002/GB-second
- **Storage**: ~$0.02/GB/month (Blob Storage)
- **Estimated monthly**: $10-30 for low traffic

### AWS ECS + Fargate
- **Compute**: ~$0.04/hour per task
- **Storage**: ~$0.023/GB/month (S3)
- **Estimated monthly**: $30-50 for low traffic

### Google Cloud Run
- **Compute**: First 2M requests free, then $0.40/million
- **Storage**: ~$0.02/GB/month (Cloud Storage)
- **Estimated monthly**: $10-20 for low traffic

## Scaling Configuration

### Azure Container Apps

```bash
az containerapp update \
  --name khipu-viewer \
  --resource-group khipu-rg \
  --min-replicas 1 \
  --max-replicas 10 \
  --scale-rule-name http-scale \
  --scale-rule-type http \
  --scale-rule-http-concurrency 50
```

### AWS ECS

Update service with desired count:

```bash
aws ecs update-service \
  --cluster khipu-cluster \
  --service khipu-viewer-service \
  --desired-count 3
```

### Google Cloud Run

Cloud Run auto-scales by default. Configure minimum/maximum instances:

```bash
gcloud run services update khipu-viewer \
  --min-instances 1 \
  --max-instances 10
```

## Monitoring

### Azure Container Apps

```bash
# View logs
az containerapp logs show \
  --name khipu-viewer \
  --resource-group khipu-rg \
  --follow

# Enable Application Insights
az monitor app-insights component create \
  --app khipu-viewer-insights \
  --location eastus \
  --resource-group khipu-rg
```

### AWS ECS

```bash
# View logs
aws logs tail /ecs/khipu-viewer --follow
```

### Google Cloud Run

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=khipu-viewer" --limit 50
```

## Security Best Practices

1. **Use managed identities** instead of connection strings when possible
2. **Enable HTTPS** for all production deployments
3. **Set up CORS** properly if using a separate frontend domain
4. **Use secrets management** (Azure Key Vault, AWS Secrets Manager, GCP Secret Manager)
5. **Implement rate limiting** to prevent abuse
6. **Enable logging and monitoring** for security auditing

## Troubleshooting

### Container won't start
- Check logs: Look for error messages in container logs
- Verify environment variables are set correctly
- Ensure data is accessible from the storage backend

### Can't access blob storage
- Check connection string/credentials
- Verify firewall rules allow access
- Ensure container/bucket exists

### High latency
- Enable CDN for blob storage
- Increase container resources (CPU/memory)
- Consider adding caching layer (Redis)

### Out of memory
- Increase memory allocation
- Reduce data export size (limit number of khipus)
- Optimize JSON files (remove unnecessary fields)

## Support

For issues specific to the cloud viewer:
- Check the [main README](README.md) for general usage
- Review the [repository issues](https://github.com/adafieno/khipu-computational-toolkit/issues)
- Contact the maintainer

For cloud provider issues:
- Azure: https://docs.microsoft.com/azure/container-apps/
- AWS: https://docs.aws.amazon.com/ecs/
- Google Cloud: https://cloud.google.com/run/docs

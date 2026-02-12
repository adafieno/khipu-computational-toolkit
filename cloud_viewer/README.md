# Cloud Khipu Viewer

A modern, cloud-ready 3D viewer for Inka khipus with blob storage support and containerized deployment.

## Features

- 🎨 **Modern UI**: Clean, intuitive interface with gradient design
- 🌐 **Cloud-Ready**: Designed for container deployment (Azure, AWS, Docker)
- 💾 **Blob Storage**: Supports Azure Blob Storage, AWS S3, or local filesystem
- 📊 **3D Visualization**: Interactive Plotly-based 3D khipu structures
- 🔍 **Advanced Filtering**: Search and filter by provenance, cord count
- 📱 **Responsive**: Works on desktop and mobile devices
- 🚀 **Fast**: No database queries - all data served from JSON files

## Architecture

### Backend
- **Framework**: Flask (Python)
- **Storage**: Pluggable backend (Local/Azure/AWS)
- **API**: RESTful JSON API

### Frontend
- **Technology**: Pure HTML/CSS/JavaScript
- **Visualization**: Plotly.js for 3D rendering
- **Design**: Modern gradient UI with responsive layout

### Data Format
- **Storage**: JSON files in blob storage
- **Structure**:
  - `khipu_index.json` - List of all khipus with metadata
  - `khipus/*.json` - Individual khipu data files
  - `colors.json` - Color code to RGB mappings

## Quick Start

### 1. Export Data from Database

First, export the khipu data to JSON format:

```bash
# From the repository root
python scripts/export_to_blob_format.py --output data/blob_export

# Or limit to first 50 khipus for testing
python scripts/export_to_blob_format.py --output data/blob_export --limit 50
```

This creates:
- `data/blob_export/khipu_index.json`
- `data/blob_export/khipus/*.json` (one per khipu)
- `data/blob_export/colors.json`

### 2. Run with Docker Compose (Recommended)

```bash
cd cloud_viewer
docker-compose up --build
```

Access the viewer at: http://localhost:5000

### 3. Run Locally (Development)

```bash
# Install dependencies
cd cloud_viewer
pip install -r requirements.txt

# Set environment variables
export STORAGE_TYPE=local
export STORAGE_PATH=../data/blob_export

# Run the app
python app.py
```

Access the viewer at: http://localhost:5000

## Deployment

### Azure Container Apps

1. **Upload data to Azure Blob Storage**:

```bash
# Install Azure CLI and login
az login

# Create storage account
az storage account create --name khipustorage --resource-group myResourceGroup --location eastus

# Create container
az storage container create --name khipu-data --account-name khipustorage

# Upload data
az storage blob upload-batch --account-name khipustorage --destination khipu-data --source data/blob_export
```

2. **Deploy container**:

```bash
# Build and push container
az acr build --registry myregistry --image khipu-viewer:latest cloud_viewer/

# Create container app
az containerapp create \
  --name khipu-viewer \
  --resource-group myResourceGroup \
  --environment myEnvironment \
  --image myregistry.azurecr.io/khipu-viewer:latest \
  --target-port 5000 \
  --ingress external \
  --env-vars \
    STORAGE_TYPE=azure \
    STORAGE_PATH=khipu-data \
    AZURE_CONNECTION_STRING="<connection-string>"
```

### AWS ECS (Elastic Container Service)

1. **Upload data to S3**:

```bash
# Create S3 bucket
aws s3 mb s3://khipu-data-bucket

# Upload data
aws s3 sync data/blob_export s3://khipu-data-bucket/
```

2. **Build and push to ECR**:

```bash
# Create ECR repository
aws ecr create-repository --repository-name khipu-viewer

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t khipu-viewer cloud_viewer/
docker tag khipu-viewer:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/khipu-viewer:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/khipu-viewer:latest
```

3. **Create ECS task and service** (use AWS Console or CLI)

### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/khipu-viewer cloud_viewer/

# Deploy
gcloud run deploy khipu-viewer \
  --image gcr.io/PROJECT_ID/khipu-viewer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars STORAGE_TYPE=local,STORAGE_PATH=./data/blob_export
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STORAGE_TYPE` | Storage backend: `local`, `azure`, or `aws` | `local` |
| `STORAGE_PATH` | Local path or container/bucket name | `../data/blob_export` |
| `AZURE_CONNECTION_STRING` | Azure Storage connection string | - |
| `AWS_ACCESS_KEY_ID` | AWS access key ID | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | - |
| `AWS_REGION` | AWS region | `us-east-1` |
| `PORT` | Server port | `5000` |
| `DEBUG` | Enable debug mode | `false` |

### Storage Backends

#### Local Filesystem
```bash
export STORAGE_TYPE=local
export STORAGE_PATH=../data/blob_export
```

#### Azure Blob Storage
```bash
export STORAGE_TYPE=azure
export STORAGE_PATH=khipu-data  # container name
export AZURE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
```

#### AWS S3
```bash
export STORAGE_TYPE=aws
export STORAGE_PATH=khipu-data-bucket  # bucket name
export AWS_ACCESS_KEY_ID=YOUR_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET
export AWS_REGION=us-east-1
```

## API Reference

### GET /api/health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "storage_type": "local",
  "storage_path": "../data/blob_export"
}
```

### GET /api/stats
Get overall statistics

**Response:**
```json
{
  "total_khipus": 612,
  "total_cords": 54403,
  "total_knots": 110677,
  "provenances": ["Unknown", "Chachapoyas", ...],
  "avg_cords_per_khipu": 88.9
}
```

### GET /api/colors
Get color code to RGB mappings

**Response:**
```json
{
  "AB": {
    "description": "Aberdeen Brown",
    "rgb": { "r": 139, "g": 69, "b": 19 },
    "hex": "#8b4513"
  },
  ...
}
```

### GET /api/khipus
List all khipus with optional filtering

**Query Parameters:**
- `provenance` (optional): Filter by provenance
- `min_cords` (optional): Minimum cord count
- `max_cords` (optional): Maximum cord count

**Response:**
```json
[
  {
    "id": "AS001",
    "provenance": "Chachapoyas",
    "museum_no": "B/1234",
    "creation_date": "1500-1600",
    "cord_count": 45,
    "knot_count": 120
  },
  ...
]
```

### GET /api/khipus/:id
Get detailed data for a specific khipu

**Response:**
```json
{
  "id": "AS001",
  "provenance": "Chachapoyas",
  "museum_no": "B/1234",
  "creation_date": "1500-1600",
  "cords": [
    {
      "cord_id": 1001,
      "position": 1,
      "color_code": "AB",
      "level": 1,
      "length_cm": 45.5
    },
    ...
  ],
  "knots": [
    {
      "cord_id": 1001,
      "knot_id": 5001,
      "ordinal": 1,
      "type": "S",
      "turns": 0
    },
    ...
  ],
  "statistics": {
    "total_cords": 45,
    "total_knots": 120,
    "pendant_count": 40,
    "subsidiary_count": 5
  }
}
```

## Development

### Project Structure

```
cloud_viewer/
├── app.py                 # Flask backend
├── static/
│   └── index.html        # Frontend UI
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Local development setup
└── README.md             # This file
```

### Adding New Features

1. **Backend**: Edit `app.py` to add new API endpoints
2. **Frontend**: Edit `static/index.html` to modify UI
3. **Storage**: Implement new storage backend in `app.py`

### Testing

```bash
# Test data export
python scripts/export_to_blob_format.py --output /tmp/test_export --limit 10

# Test local server
cd cloud_viewer
STORAGE_PATH=/tmp/test_export python app.py

# Test Docker build
docker build -t khipu-viewer-test .
docker run -p 5000:5000 -e STORAGE_PATH=./data/blob_export khipu-viewer-test
```

## Comparison with Original Viewer

| Feature | Original (Streamlit) | Cloud Viewer (Flask) |
|---------|---------------------|---------------------|
| Framework | Streamlit | Flask + HTML/JS |
| Data Source | SQLite Database | JSON Blob Storage |
| Deployment | Single server | Containerized (cloud-ready) |
| UI Style | Streamlit default | Modern gradient design |
| Scalability | Limited | Horizontal scaling |
| Cloud Native | No | Yes |
| Storage Options | Database only | Local/Azure/AWS/custom |

## Advantages

- ✅ **No Database Required**: All data in JSON files
- ✅ **Cloud-Ready**: Easy deployment to any cloud provider
- ✅ **Scalable**: Can handle high traffic with CDN + blob storage
- ✅ **Fast**: No database queries, served from blob storage
- ✅ **Modern UI**: Clean, intuitive interface
- ✅ **Containerized**: Docker support for easy deployment
- ✅ **Flexible Storage**: Multiple storage backend options

## Limitations

- Data is read-only (no editing through UI)
- Requires data export step from database
- No user authentication (add reverse proxy if needed)

## License

MIT License - See main repository LICENSE file

## Contributing

This is part of the Khipu Computational Analysis Toolkit. See main repository for contribution guidelines.

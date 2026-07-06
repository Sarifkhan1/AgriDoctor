#!/bin/bash

# =============================================================================
# AgriDoctor AI - Local-to-VPS Deployer (Multi-Site Edition)
# Run this script from your local Mac terminal to deploy without port conflicts!
# =============================================================================

set -euo pipefail

VPS_IP="69.62.78.148"
VPS_PORT="2222"
VPS_USER="root"
REMOTE_DIR="/root/agri-doctor"

echo "====================================================================="
echo "🌿 AgriDoctor AI - One-Click VPS Deployer (Multi-Site Safe) 🌿"
echo "====================================================================="
echo "Target VPS: $VPS_USER@$VPS_IP:$VPS_PORT"
echo "Target Domain: http://127.0.0.1:8020 (Internal)"
echo "Public Domain: https://agridoctor.cloud"
echo "====================================================================="

# Step 1: Upload files to VPS
echo "📤 Step 1: Uploading project files to VPS..."
ssh -p "$VPS_PORT" "$VPS_USER@$VPS_IP" "mkdir -p $REMOTE_DIR"
scp -P "$VPS_PORT" -r \
  backend \
  config \
  data \
  docs \
  frontend \
  src \
  tools \
  Dockerfile \
  docker-compose.yml \
  nginx.conf \
  requirements.txt \
  requirements-ml.txt \
  start.sh \
  "$VPS_USER@$VPS_IP:$REMOTE_DIR/"


echo "✅ Upload complete!"

# Step 2: Provision VPS and Launch Services
echo "🛠️ Step 2: Connecting to VPS to run containerized services..."
ssh -p "$VPS_PORT" "$VPS_USER@$VPS_IP" "bash -s" << 'EOF'
  set -euo pipefail
  
  # Install Docker if not present
  if ! [ -x "$(command -v docker)" ]; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
  fi
  
  # Install Docker Compose if not present
  if ! docker compose version &>/dev/null && ! [ -x "$(command -v docker-compose)" ]; then
    echo "🐳 Installing Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin || apt-get install -y docker-compose
  fi
  
  # Go to project directory
  cd /root/agri-doctor
  chmod +x start.sh
  
  # Spin up services in background
  echo "🚀 Launching containerized application stack on alternative ports..."
  if docker compose version &>/dev/null; then
    docker compose up -d --build
  else
    docker-compose up -d --build
  fi
  
  echo "📊 Container status:"
  docker ps
EOF

echo "====================================================================="
echo "🎉 Containers Deployed Successfully!"
echo "====================================================================="
echo "Your AgriDoctor container is running internally at:"
echo "👉 Frontend Host Port: 8020"
echo "👉 Backend Host Port:  8002"
echo "👉 Annotator Host Port: 8502"
echo ""
echo "Next Step: Configure your VPS host reverse proxy to route traffic for"
echo "agridoctor.cloud -> http://127.0.0.1:8020 and provision SSL on the host."
echo "====================================================================="

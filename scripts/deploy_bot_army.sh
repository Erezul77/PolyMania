#!/bin/bash
# PolyMania Bot Army - Deployment Script for Oracle VM
# ====================================================

set -e

# Configuration
REMOTE_USER="ubuntu"
REMOTE_HOST="${VPS_HOST:-your-oracle-vm-ip}"
REMOTE_DIR="/home/ubuntu/polymania"
SSH_KEY="${SSH_KEY:-~/.ssh/oci_polymania.key}"

echo "🤖 PolyMania Bot Army Deployment"
echo "================================="
echo ""

# Check SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found at $SSH_KEY"
    exit 1
fi

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf /tmp/bot_army_deploy.tar.gz \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='logs/*.log*' \
    --exclude='*.session' \
    bot_army/ \
    config/ \
    grafana/ \
    init_db/ \
    docker-compose.oracle.yml \
    Dockerfile.oracle \
    Dockerfile.dashboard \
    requirements.oracle.txt \
    requirements.dashboard.txt

# Upload to server
echo "📤 Uploading to $REMOTE_HOST..."
scp -i "$SSH_KEY" /tmp/bot_army_deploy.tar.gz "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

# Deploy on server
echo "🚀 Deploying on server..."
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" << 'ENDSSH'
    cd /home/ubuntu/polymania
    
    # Extract new code
    echo "Extracting deployment package..."
    tar -xzf bot_army_deploy.tar.gz
    rm bot_army_deploy.tar.gz
    
    # Stop existing services
    echo "Stopping existing services..."
    docker compose -f docker-compose.oracle.yml down || true
    
    # Build and start services
    echo "Building and starting Bot Army..."
    docker compose -f docker-compose.oracle.yml build
    docker compose -f docker-compose.oracle.yml up -d
    
    # Check status
    echo ""
    echo "Service Status:"
    docker compose -f docker-compose.oracle.yml ps
    
    echo ""
    echo "✅ Deployment complete!"
ENDSSH

# Cleanup
rm /tmp/bot_army_deploy.tar.gz

echo ""
echo "🎉 Bot Army deployed successfully!"
echo ""
echo "Access points:"
echo "  📊 Dashboard:  http://$REMOTE_HOST:8501"
echo "  📈 Grafana:    http://$REMOTE_HOST:3000"
echo "  🐘 Postgres:   $REMOTE_HOST:5432"
echo ""

#!/bin/bash
# Setup ComfyUI as a systemd service on junkpile
# Run this script ON junkpile as root or with sudo

set -euo pipefail

echo "=== ComfyUI Setup Script ==="

# 1. Create comfyui user and group
echo "[1/6] Creating comfyui user and group..."
if ! getent group comfyui > /dev/null 2>&1; then
    groupadd --system comfyui
    echo "  Created group: comfyui"
else
    echo "  Group comfyui already exists"
fi

if ! id comfyui > /dev/null 2>&1; then
    useradd --system --gid comfyui --home-dir /opt/comfyui --shell /usr/sbin/nologin comfyui
    echo "  Created user: comfyui"
else
    echo "  User comfyui already exists"
fi

# 2. Create directory structure
echo "[2/6] Creating directory structure..."
mkdir -p /opt/comfyui
chown comfyui:comfyui /opt/comfyui

# 3. Move ComfyUI installation
echo "[3/6] Moving ComfyUI installation..."
SOURCE_DIR="/home/chi/Projects/ComfyUI"
DEST_DIR="/opt/comfyui/app"

if [[ -d "$SOURCE_DIR" ]]; then
    if [[ -d "$DEST_DIR" ]]; then
        echo "  Destination already exists, backing up..."
        mv "$DEST_DIR" "${DEST_DIR}.bak.$(date +%Y%m%d%H%M%S)"
    fi
    mv "$SOURCE_DIR" "$DEST_DIR"
    echo "  Moved $SOURCE_DIR -> $DEST_DIR"
else
    if [[ -d "$DEST_DIR" ]]; then
        echo "  ComfyUI already at $DEST_DIR"
    else
        echo "  ERROR: Source directory not found: $SOURCE_DIR"
        exit 1
    fi
fi

# 4. Set ownership
echo "[4/6] Setting ownership..."
chown -R comfyui:comfyui /opt/comfyui
# Add chi to comfyui group for model access
usermod -aG comfyui chi || true
echo "  Ownership set to comfyui:comfyui"
echo "  Added chi to comfyui group"

# 5. Create systemd service
echo "[5/6] Creating systemd service..."
cat > /etc/systemd/system/comfyui.service << 'EOF'
[Unit]
Description=ComfyUI Image Generation Server
After=network.target

[Service]
Type=simple
User=comfyui
Group=comfyui
WorkingDirectory=/opt/comfyui/app
Environment=HOME=/opt/comfyui
Environment=PATH=/usr/local/bin:/usr/bin:/bin

# Run with uv - listen on all interfaces
ExecStart=/usr/local/bin/uv run python main.py --listen 0.0.0.0 --port 8188

# Restart on failure
Restart=on-failure
RestartSec=10

# GPU access
SupplementaryGroups=video render

[Install]
WantedBy=multi-user.target
EOF

echo "  Created /etc/systemd/system/comfyui.service"

# 6. Enable and start service
echo "[6/6] Enabling and starting service..."
systemctl daemon-reload
systemctl enable comfyui
systemctl start comfyui

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Service status:"
systemctl status comfyui --no-pager || true
echo ""
echo "Check logs with: journalctl -u comfyui -f"
echo "ComfyUI URL: http://$(hostname):8188"

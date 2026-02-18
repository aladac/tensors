#!/bin/bash
# Tensors Reinstall Script
# Updates version with git hash, reinstalls locally and on junkpile
set -euo pipefail

TENSORS_ROOT="/Users/chi/Projects/tensors"
JUNKPILE_HOST="chi@junkpile"
JUNKPILE_PATH="/opt/tensors/app"

cd "$TENSORS_ROOT"

# Get current version from __init__.py
CURRENT_VERSION=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' tensors/__init__.py)
BASE_VERSION=$(echo "$CURRENT_VERSION" | sed 's/+.*//')

# Get the LAST commit hash BEFORE we make any changes
LAST_HASH=$(git rev-parse --short HEAD)

# New version with hash
NEW_VERSION="${BASE_VERSION}+${LAST_HASH}"

echo "=== Tensors Reinstall ==="
echo "Current version: $CURRENT_VERSION"
echo "Base version:    $BASE_VERSION"
echo "Last commit:     $LAST_HASH"
echo "New version:     $NEW_VERSION"
echo ""

# Check if version already matches (avoid loop)
if [[ "$CURRENT_VERSION" == "$NEW_VERSION" ]]; then
    echo "[1/5] Version already at $NEW_VERSION, skipping version bump"
else
    echo "[1/5] Updating version to $NEW_VERSION..."
    sed -i '' "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" tensors/__init__.py
    echo "  Updated tensors/__init__.py"

    # Commit version change if there are changes
    if ! git diff --quiet; then
        echo ""
        echo "[2/5] Committing version update..."
        git add tensors/__init__.py
        git commit -m "Version $NEW_VERSION"
        echo "  Committed."
    fi
fi

# Push if ahead of remote
if git status | grep -q "Your branch is ahead"; then
    echo ""
    echo "Pushing to remote..."
    git push
fi

echo ""
echo "[3/5] Installing locally via uv..."
uv pip install -e .

echo ""
echo "[4/5] Pulling on junkpile..."
ssh "$JUNKPILE_HOST" "cd $JUNKPILE_PATH && sudo -u tensors git checkout . && sudo -u tensors git pull"

echo ""
echo "[5/5] Installing on junkpile..."
ssh "$JUNKPILE_HOST" "cd $JUNKPILE_PATH && sudo -u tensors uv sync && sudo systemctl restart tensors"

echo ""
echo "=== Verification ==="
uv run tsr --version

echo ""
echo "=== Done ==="
echo "Version: $NEW_VERSION"
echo "Installed locally and on junkpile."

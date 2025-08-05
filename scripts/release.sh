#!/bin/bash
# Release script for gds_fdtd
# Usage: ./scripts/release.sh [patch|minor|major]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default to patch if no argument provided
BUMP_TYPE=${1:-patch}

echo -e "${BLUE}Starting release process for gds_fdtd${NC}"
echo -e "${BLUE}Bump type: ${BUMP_TYPE}${NC}"

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}Error: Must be on main branch for release. Currently on: $CURRENT_BRANCH${NC}"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: Uncommitted changes detected. Please commit or stash changes first.${NC}"
    exit 1
fi

# Pull latest changes
echo -e "${YELLOW}Pulling latest changes...${NC}"
git pull origin main

# Show current version
echo -e "${BLUE}Current version information:${NC}"
make check-version

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
make test

# Build documentation
echo -e "${YELLOW}Building documentation...${NC}"
make docs

# Bump version
echo -e "${YELLOW}Bumping ${BUMP_TYPE} version...${NC}"
bump2version $BUMP_TYPE

# Get new version
NEW_VERSION=$(grep '__version__' gds_fdtd/__init__.py | cut -d'"' -f2)
echo -e "${GREEN}Version bumped to: v${NEW_VERSION}${NC}"

# Push changes and tags
echo -e "${YELLOW}Pushing changes and tags...${NC}"
git push origin main --tags

echo -e "${GREEN}Release process completed!${NC}"
echo -e "${GREEN}Version v${NEW_VERSION} has been tagged and pushed.${NC}"
echo -e "${BLUE}GitHub release will be created automatically.${NC}"
echo -e "${BLUE}Documentation will be updated at: https://siepic.github.io/gds_fdtd/${NC}"

# Optional: Open GitHub releases page
if command -v open &> /dev/null; then
    echo -e "${BLUE}Opening GitHub releases page...${NC}"
    open "https://github.com/SiEPIC/gds_fdtd/releases"
elif command -v xdg-open &> /dev/null; then
    echo -e "${BLUE}Opening GitHub releases page...${NC}"
    xdg-open "https://github.com/SiEPIC/gds_fdtd/releases"
fi 
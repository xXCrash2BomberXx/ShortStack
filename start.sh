#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Change to that directory
cd "$SCRIPT_DIR"

# Pull required images
podman-compose pull

# Execute docker-compose
podman-compose up

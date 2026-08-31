#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="ollama"
PODMAN_ROOT="${SCRIPT_DIR}/containers-storage"
PODMAN="podman --root ${PODMAN_ROOT}"

pushd ./models
CURRENT_MODELS=$($PODMAN exec "$CONTAINER_NAME" ollama list | awk '{print $1}' | cut -d: -f1)
for file in *.gguf; do
    # Handle case where no .gguf files exist
    [ -e "$file" ] || continue

    model_name="${file%.*}"
    model_name="${model_name//[._]/-}"

    if echo "$CURRENT_MODELS" | grep -qx "$model_name"; then
        echo "ok: $model_name is already registered."
    else
        echo "new: Registering $model_name..."

        $PODMAN exec "$CONTAINER_NAME" sh -c "echo 'FROM /root/models/$file' > /root/models/tmp_modelfile"
        $PODMAN exec "$CONTAINER_NAME" ollama create "$model_name" -f /root/models/tmp_modelfile
        $PODMAN exec "$CONTAINER_NAME" rm /root/models/tmp_modelfile
    fi
done
popd

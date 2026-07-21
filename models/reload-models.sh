#!/bin/bash

CONTAINER_NAME="ollama"

echo "Checking for new models in /mnt/data/LLM..."

# Get current models, removing the 'latest' tag for cleaner comparison
CURRENT_MODELS=$(docker exec $CONTAINER_NAME ollama list | awk '{print $1}' | cut -d: -f1)

for file in *.gguf; do
    # Clean up model name: remove extension and replace dots/underscores with dashes
    model_name="${file%.*}"
    model_name="${model_name//[._]/-}"

    if echo "$CURRENT_MODELS" | grep -qx "$model_name"; then
        echo "ok: $model_name is already registered."
    else
        echo "new: Registering $model_name..."
        
        # 1. Create a physical Modelfile inside the shared volume
        # We use /root/models/ because that's where the container sees this folder
        docker exec $CONTAINER_NAME sh -c "echo 'FROM /root/models/$file' > /root/models/tmp_modelfile"
        
        # 2. Point Ollama to that specific file
        docker exec $CONTAINER_NAME ollama create "$model_name" -f /root/models/tmp_modelfile
        
        # 3. Clean up the temporary file
        docker exec $CONTAINER_NAME rm /root/models/tmp_modelfile
    fi
done

echo "Done! Refresh your WebUI to see the changes."

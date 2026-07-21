# SmallStack

## Setup

1. Create a `.env` file with a `TS_AUTHKEY` from [Tailscale](https://console.tailscale.com/admin/settings/keys)
2. Download desired `.gguf` text model(s) to `./models/`
3. `docker-compose up`
4. `./reload-models.sh`
    - You might need to `sudo chmod +x ./reload-models.sh` beforehand
5. Download desired `.safetensors` image model(s) to `./image`
6. Consult the [sacred texts](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/comfyui#create-image-image-generation)
7. 
8. Profit

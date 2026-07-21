# SmallStack

## Setup

1. Create a `.env` file with a `TS_AUTHKEY` from [Tailscale](https://console.tailscale.com/admin/settings/keys)
2. Download desired `.gguf` [text model(s)](https://huggingface.co/models?pipeline_tag=text-generation&library=gguf&sort=trending) to `./models/`
    - This can be easily changed to other formats and model types 
4. Run `docker-compose up`
5. Run `./reload-models.sh`
    - You might need to `sudo chmod +x ./reload-models.sh` beforehand
6. Download desired `.safetensors` [image model(s)](https://huggingface.co/models?pipeline_tag=text-to-image&library=safetensors&sort=trending) to `./image`
    - This can be easily changed to other formats and model types 
7. Consult the [sacred texts](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/comfyui#create-image-image-generation)
8. 
9. Profit?

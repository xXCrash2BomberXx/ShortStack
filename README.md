# ShortStack

I wanted a lightweight and containerized LLM stack with drag and drop models... so here's this.
Ask an AI to debug it for you, it's fewer tokens than you'd spend not hosting yourself after this.
`ollama` is exposed over `port 11434` for agentic work through your `tailscale`/`localhost` with `open-webui` exposed at root (`port 443`).
I get around 40 tk/s on Gemma 4 (4.6B) and 20 tk/s on Qwen 3.5 (4.2B) with an Nvidia RTX 2060 GPU.

## Setup

1. Create a `.env` file with a `TS_AUTHKEY` from [Tailscale](https://console.tailscale.com/admin/settings/keys)
   - There's also a `GLOBAL_CONTEXT_LENGTH` variable you can set for `ollama`
3. Download desired `.gguf` [text model(s)](https://huggingface.co/models?pipeline_tag=text-generation&library=gguf&sort=trending) to `./models/`
   - This can be easily changed to other formats and model types 
4. Run `docker-compose up`
5. Run `./reload-models.sh`
   - You might need to `sudo chmod +x ./reload-models.sh` beforehand
6. Download desired `.safetensors` [image model(s)](https://huggingface.co/models?pipeline_tag=text-to-image&library=safetensors&sort=trending) to `./image/`
   - This can be easily changed to other formats and model types 
7. Consult the [sacred texts](https://docs.openwebui.com/features/chat-conversations/image-generation-and-editing/comfyui#create-image-image-generation)
8. 
9. Profit?

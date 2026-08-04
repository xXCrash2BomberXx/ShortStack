# ShortStack

I wanted a lightweight and containerized LLM stack with drag and drop models... so here's this.
Hardly anything new, but maybe easier than doing it from scratch for likely the same final result.
Ask an AI to debug it for you, it's fewer tokens than you'd spend not hosting yourself after this.
`ollama` is exposed over `port 11434` for agentic work through your `tailscale`/`localhost` with `open-webui` exposed at root (`port 443`).

## Setup

1. Create a `.env` file with a `TS_AUTHKEY` from [Tailscale](https://console.tailscale.com/admin/settings/keys)
   - You'll later need to approve the device using the key
3. Download desired `.gguf` [text model(s)](https://huggingface.co/models?pipeline_tag=text-generation&library=gguf&sort=trending) to `./models/`
   - This can be easily changed to other formats and model types 
4. Run `docker-compose up`
5. Run `reload-models.sh` in `./models`
   - You might need to `sudo chmod +x reload-models.sh` beforehand
6. Download desired `.safetensors` [image model(s)](https://huggingface.co/models?pipeline_tag=text-to-image&library=safetensors&sort=trending) to `./image/`
   - This can be easily changed to other formats and model types
   - You'll need to configure the model in the Open WebUI Admin Panel for images
8. 
9. Profit.

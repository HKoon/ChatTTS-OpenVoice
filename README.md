__ChatTTS x OpenVoice__

Enhance the authenticity of speech by utilizing ChatTTS for more natural voice generation, complemented with the voice timber simulation module from Openvoice for seamless tone transplantation.

Have a try on huggingface!
https://huggingface.co/spaces/Hilley/ChatTTS-OpenVoice

---
__Experimental LLM Integration__

`ChatTTS/experimental/llm.py` provides an OpenAI-compatible LLM API wrapper for text pre-processing (e.g. text normalisation before TTS inference). Supported providers:

| Provider | `base_url` | Prompt versions |
|----------|-----------|----------------|
| Kimi (Moonshot AI) | `https://api.moonshot.cn/v1` | `kimi` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek`, `deepseek_TN` |
| **MiniMax** | `https://api.minimax.io/v1` | `minimax`, `minimax_TN` |

**MiniMax quick start** (models: `MiniMax-M3`, `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`; all with 204K context):

```python
import os
from ChatTTS.experimental.llm import create_minimax_client

client = create_minimax_client(os.environ["MINIMAX_API_KEY"])

# Conversational reply (TTS-friendly tone, ≤100 chars)
reply = client.call("今天北京天气怎么样？", prompt_version="minimax")

# Text normalisation before TTS
normalized = client.call("We paid $123 for this desk.", prompt_version="minimax_TN")
```

Get a MiniMax API key at https://www.minimaxi.com/

<img width="1792" alt="image" src="https://github.com/HKoon/ChatTTS-OpenVoice/assets/24382626/9d9592f1-b527-4c7a-b7f8-caf2cd25bc1d">



---
__Notice:__

We need to download the OpenVoice Checkpoint and save it into the __./OpenVoice/checkpoint__ folder.

__OpenVoice Checkpoint:__ https://huggingface.co/myshell-ai/OpenVoice/tree/main/checkpoints
<img width="933" alt="image" src="https://github.com/user-attachments/assets/8ff87528-805e-4ba9-82fb-4571cc456fd6">

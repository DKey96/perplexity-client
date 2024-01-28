# Unofficial Perplexity API Client

This is an UNOFFICIAL Perplexity API Client for interaction with Perplexity AI API.

## Usage 

```python
from peplexity_client.client import PerplexityClient, PerplexityModels


client = PerplexityClient(api_key="YOUR-API-KEY")
messages = [
  {"role": "user", "text": "Give me 20 most visited places on Earth"}
]

# Here we can use online model or leave it with the default mistral-7b-instruct model
places = client.chat_completion(messages=messages, model=PerplexityModels.PPLX_70B_ONLINE)
```

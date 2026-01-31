
import os
import json
import base64
import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

def encode_video_to_base64(video_path: str) -> str:
    """Encodes a video file to a base64 string."""
    with open(video_path, "rb") as video_file:
        video_encoded = base64.b64encode(video_file.read()).decode('utf-8')
    return f"data:video/mp4;base64,{video_encoded}"

def call_openrouter(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Calls the OpenRouter API.
    
    Args:
        model: The model identifier (e.g., "openai/gpt-4").
        messages: A list of message dictionaries.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        response_format: Optional response format (e.g., {"type": "json_object"}).
        **kwargs: Additional parameters to pass to the API (e.g., top_p, top_k, seed).
    
    Returns:
        The JSON response from OpenRouter.
    """
    api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPEN_ROUTER_API_KEY not found in environment variables.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    if response_format:
        payload["response_format"] = response_format

    # Add any additional parameters
    payload.update(kwargs)

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"OpenRouter HTTP Error: {e}")
        if response.text:
            logger.error(f"Response content: {response.text}")
        raise
    except Exception as e:
        logger.error(f"OpenRouter Request Failed: {e}")
        raise

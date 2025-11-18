import os
import numpy as np
import base64
import requests
import json
from io import BytesIO
from typing import List, Union, Optional, Dict, Any
from PIL import Image

# Define supported models - now just a reference list since LiteLLM handles all models
SUPPORTED_MODELS = [
    # OpenAI models
    "gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo", "gpt-5-nano",
    # Ollama models
    "llava", "bakllava", "llava-next",
    # Can easily add more models supported by LiteLLM
    "claude-3-sonnet", "gemini-pro-vision", "gpt-4o-mini",

    "gemma3:12b", "gemma3:4b", "qwen2.5vl", "minicpm-v:8b"
]

def encode_image_to_base64(image) -> str:
    """
    Encodes an image into a base64-encoded string in JPEG format.

    Parameters:
        image (Union[str, Image.Image, np.ndarray]): The image to be encoded.
            This will be a string of the image path, a PIL image, or a numpy array.

    Returns:
        str: A base64-encoded string representing the image in JPEG format.
    """
    # Function to encode the image from file path
    def _encode_image_from_file(image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Function to encode the image from PIL Image
    def _encode_image_from_pil(image):
        buffered = BytesIO()
        image.save(buffered, format='JPEG')
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    if isinstance(image, str):
        return _encode_image_from_file(image)
    elif isinstance(image, Image.Image):
        return _encode_image_from_pil(image)
    elif isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
        return _encode_image_from_pil(image_pil)
    else:
        raise ValueError(f"Unknown option for image {type(image)}")


class LiteLLMRequestHandler:
    """Handler for LiteLLM proxy requests - supports all models through unified API."""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize LiteLLM request handler.
        
        Args:
            api_url: LiteLLM proxy URL. Defaults to http://localhost:4000
            api_key: API key for LiteLLM proxy (if authentication is enabled)
        """
        self.api_url = api_url or "http://localhost:4000"
        # Ensure the URL has the correct endpoint
        if not self.api_url.endswith('/v1/chat/completions'):
            self.api_url = self.api_url.rstrip('/') + '/v1/chat/completions'
        
        # Try to get API key from various sources
        self.api_key = (
            api_key or 
            os.environ.get('LITELLM_API_KEY') or 
            os.environ.get('OPENAI_API_KEY') or
            'dummy-key'  # LiteLLM proxy might not require auth in some setups
        )
    
    def prepare_messages(self, 
                        images: List[Union[str, Image.Image, np.ndarray]],
                        prompt: str,
                        system_prompt: str,
                        in_context_examples: Optional[List[Dict]] = None) -> List[Dict]:
        """Prepare messages in OpenAI chat format for LiteLLM."""
        messages = []
        
        # Add system message
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Prepare user message content
        user_content = []
        
        # Add in-context examples if provided
        if in_context_examples:
            for example in in_context_examples:
                # Add example prompt text
                if example.get('prompt'):
                    user_content.append({
                        'type': 'text',
                        'text': example['prompt']
                    })
                
                # Add example images
                for img in example.get('images', []):
                    base64_image = encode_image_to_base64(img)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                
                # Add expected response
                if example.get('response'):
                    user_content.append({
                        'type': 'text',
                        'text': f"The answer should be: {example['response']}\n"
                    })
        
        # Add user prompt text
        if prompt:
            user_content.append({
                'type': 'text',
                'text': prompt
            })
        else:
            # Ensure we have at least images if no text prompt
            assert len(images) > 0, "Both images and text prompts are empty."
        
        # Add user images
        for image in images:
            base64_image = encode_image_to_base64(image)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def request(self, 
           images: Union[Union[str, Image.Image, np.ndarray], List[Union[str, Image.Image, np.ndarray]]], 
           prompt: str, 
           system_prompt: str,
           model_name: str = "gpt-4o",
           **kwargs) -> Union[str, List[str]]:
        """
        Make a request to any model through LiteLLM proxy.
        """

        # Convert single image to list for consistency
        if not isinstance(images, list):
            images = [images]
        
        # Prepare messages
        messages = self.prepare_messages(
            images, 
            prompt, 
            system_prompt, 
            kwargs.get("in_context_examples")
        )

        # Base payload
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "n": kwargs.get("n", 1),
        }

        # ✅ Handle model-specific token parameter
        max_tokens = kwargs.get("max_tokens", 256)
        model_name_lower = model_name.lower()

        if any(key in model_name_lower for key in ["gpt-4o", "gpt-4.1", "gpt-5", "gpt-5-nano"]):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        temperature = kwargs.get("temperature", 0.0)

        if "gpt-5-nano" in model_name_lower:
            # Either remove the key entirely or force it to 1
            payload["temperature"] = 1
        else:
            payload["temperature"] = temperature

        # Optional parameters
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs.get("seed")

        if kwargs.get("return_logprobs"):
            payload["logprobs"] = True

        # Make request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(
                url=self.api_url, 
                headers=headers, 
                json=payload,
                timeout=kwargs.get("timeout", 120)
            )
            response.raise_for_status()
            response_data = response.json()
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"LiteLLM proxy request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LiteLLM proxy: {str(e)}")

        # Handle errors in response
        if 'error' in response_data:
            error_msg = response_data['error'].get('message', 'Unknown error')
            raise ValueError(f"LiteLLM proxy error: {error_msg}")
        
        # Extract responses
        try:
            responses = [choice['message']['content'] for choice in response_data['choices']]
            return responses[0] if len(responses) == 1 else responses
            
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected response format from LiteLLM proxy: {str(e)}")



# Global handler instance
_litellm_handler = None

def get_litellm_handler(api_url: Optional[str] = None, api_key: Optional[str] = None) -> LiteLLMRequestHandler:
    """Get or create a LiteLLM handler instance."""
    global _litellm_handler
    
    # Create new handler if none exists or if different parameters are provided
    if (_litellm_handler is None or 
        (api_url and api_url != _litellm_handler.api_url) or
        (api_key and api_key != _litellm_handler.api_key)):
        _litellm_handler = LiteLLMRequestHandler(api_url, api_key)
    
    return _litellm_handler


def request_model(
    images: Union[Union[str, Image.Image, np.ndarray], List[Union[str, Image.Image, np.ndarray]]],
    prompt: str,
    system_prompt: str,
    model_name: str = "gpt-4o",
    litellm_api_url: Optional[str] = None,
    litellm_api_key: Optional[str] = None,
    **kwargs
) -> Union[str, List[str]]:
    """
    Make a request to any model through LiteLLM proxy.
    
    Args:
        images: Single image or list of images (file paths, PIL Images, or numpy arrays)
        prompt: Text prompt to send to the model
        system_prompt: System instructions for the model
        model_name: Name of the model to use (any model supported by LiteLLM)
        litellm_api_url: LiteLLM proxy URL (default: http://localhost:4000)
        litellm_api_key: API key for LiteLLM proxy
        **kwargs: Additional arguments:
            - temperature: Sampling temperature (default: 0.0)
            - max_tokens: Maximum number of tokens to generate (default: 256)
            - n: Number of completions to generate (default: 1)
            - return_logprobs: Whether to return log probabilities (default: False)
            - in_context_examples: List of examples for in-context learning
            - seed: Optional seed for reproducible outputs
            - timeout: Request timeout in seconds (default: 120)
    
    Returns:
        Model response as a string or list of strings
    """
    handler = get_litellm_handler(litellm_api_url, litellm_api_key)
    return handler.request(images, prompt, system_prompt, model_name, **kwargs)


# Backward compatibility function
def request_gpt(
    images: Union[np.ndarray, List[np.ndarray]],
    prompt: str,
    system_prompt: str,
    detail: str = "auto", 
    temp: float = 0.0,
    n_tokens: int = 256,
    n: int = 1,
    return_logprobs: bool = True,
    in_context_examples: List[Dict] = None,
    model_name: str = "gpt-4o",
    seed: Optional[int] = None,
    litellm_api_url: Optional[str] = None,
    litellm_api_key: Optional[str] = None
) -> Union[str, List[str]]:
    """
    Backward compatible wrapper for the original request_gpt function.
    Now routes through LiteLLM proxy for any model.
    
    Args:
        images: Single image or list of images (numpy arrays, PIL Images, or file paths)
        prompt: Text prompt to send to the model
        system_prompt: System instructions for the model
        detail: Image detail level (maintained for compatibility)
        temp: Sampling temperature
        n_tokens: Maximum number of tokens to generate
        n: Number of completions to generate
        return_logprobs: Whether to return log probabilities
        in_context_examples: List of examples for in-context learning
        model_name: Name of the model to use
        seed: Optional seed for reproducible outputs
        litellm_api_url: LiteLLM proxy URL
        litellm_api_key: API key for LiteLLM proxy
        
    Returns:
        Model response as a string or list of strings
    """
    # Map parameters to the new API
    kwargs = {
        "temperature": temp,
        "max_tokens": n_tokens,
        "n": n,
        "return_logprobs": return_logprobs,
        "in_context_examples": in_context_examples,
        "seed": seed,
        "litellm_api_url": litellm_api_url,
        "litellm_api_key": litellm_api_key
    }
    
    return request_model(images, prompt, system_prompt, model_name, **kwargs)
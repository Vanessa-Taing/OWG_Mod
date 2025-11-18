import os
import numpy as np
import base64
import requests
import json
from io import BytesIO
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod
from PIL import Image

# Define supported models by provider
GPT_MODELS = ["gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo"]
OLLAMA_MODELS = ["llava", "bakllava", "llava-next"]

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


class ModelRequestHandler(ABC):
    """Abstract base class for handling model requests."""
    
    @abstractmethod
    def prepare_payload(self, 
                        images: List[Union[str, Image.Image, np.ndarray]], 
                        prompt: str, 
                        system_prompt: str,
                        **kwargs) -> Dict[str, Any]:
        """Prepare the payload for API request."""
        pass
    
    @abstractmethod
    def make_request(self, payload: Dict[str, Any]) -> str:
        """Make the API request and return the response."""
        pass
    
    def request(self, 
               images: Union[Union[str, Image.Image, np.ndarray], List[Union[str, Image.Image, np.ndarray]]], 
               prompt: str, 
               system_prompt: str,
               **kwargs) -> str:
        """Process request with the model."""
        # Convert single image to list for consistency
        if not isinstance(images, list):
            images = [images]
        
        payload = self.prepare_payload(images, prompt, system_prompt, **kwargs)
        return self.make_request(payload)


class GPTRequestHandler(ModelRequestHandler):
    """Handler for OpenAI GPT model requests."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize GPT request handler.
        
        Args:
            model_name: Name of the GPT model to use
            api_key: OpenAI API key. If None, will try to get from environment.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required for GPT models. Provide it or set OPENAI_API_KEY environment variable.")
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def prepare_prompt(self, 
                      images: List[Union[str, Image.Image, np.ndarray]],
                      prompt: Optional[str] = None,
                      in_context_examples: Optional[List[Dict]] = None, 
                      **kwargs) -> Dict:
        """Prepare the prompt for GPT models."""
        set_prompt = {
            'role': 'user',
            'content': []
        }
        
        # Include in-context examples if provided
        if in_context_examples:
            for example in in_context_examples:
                # Add example prompt text if available
                if example.get('prompt'):
                    set_prompt['content'].append({
                        'type': 'text',
                        'text': example['prompt']
                    })
                
                # Add example images if available
                for img in example.get('images', []):
                    base64_image = encode_image_to_base64(img)
                    set_prompt['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                
                # Add expected response
                if example.get('response'):
                    set_prompt['content'].append({
                        'type': 'text',
                        'text': f"The answer should be: {example['response']}\n"
                    })
        
        # Add user prompt text if available
        if prompt:
            set_prompt['content'].append({
                'type': 'text',
                'text': prompt
            })
        else:
            # Ensure we have at least one image if no text prompt
            assert len(images) > 0, "Both images and text prompts are empty."
        
        # Add user images
        for image in images:
            base64_image = encode_image_to_base64(image)
            set_prompt['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    # "detail": kwargs.get("detail", "auto")
                }
            })
        
        return set_prompt
    
    def prepare_payload(self, 
                       images: List[Union[str, Image.Image, np.ndarray]], 
                       prompt: str, 
                       system_prompt: str,
                       **kwargs) -> Dict[str, Any]:
        """Prepare payload for GPT models."""
        # Prepare system message
        system_msg = {
            "role": "system",
            "content": system_prompt
        }
        
        # Prepare user message with images and text
        user_msg = self.prepare_prompt(
            images, 
            prompt, 
            kwargs.get("in_context_examples")
        )
        
        payload = {
            "model": self.model_name,
            "messages": [system_msg, user_msg],
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.0),
            "n": kwargs.get("n", 1),
        }
        
        # Add optional parameters
        if kwargs.get("return_logprobs"):
            payload["logprobs"] = True
        
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs.get("seed")
            
        return payload
    
    def make_request(self, payload: Dict[str, Any]) -> str:
        """Make request to GPT API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(url=self.api_url, headers=headers, json=payload).json()
        
        if 'error' in response:
            raise ValueError(response['error']['message'])
        
        responses = [r['message']['content'] for r in response['choices']]
        return responses[0] if len(responses) == 1 else responses


class OllamaRequestHandler(ModelRequestHandler):
    """Handler for Ollama model requests."""
    
    def __init__(self, model_name: str, api_url: Optional[str] = None):
        """
        Initialize Ollama request handler.
        
        Args:
            model_name: Name of the Ollama model to use
            api_url: Ollama API URL. Defaults to http://localhost:11434/api/generate
        """
        self.model_name = model_name
        self.api_url = api_url or "http://localhost:11434/api/generate"
    
    def prepare_payload(self, 
                       images: List[Union[str, Image.Image, np.ndarray]], 
                       prompt: str, 
                       system_prompt: str,
                       **kwargs) -> Dict[str, Any]:
        """Prepare payload for Ollama models."""
        # For Ollama's format, we need to combine the system prompt and user prompt
        combined_prompt = f"{system_prompt}\n\n{prompt}" if prompt else system_prompt
        
        # Encode images to base64
        image_data = [encode_image_to_base64(img) for img in images]
        
        payload = {
            "model": self.model_name,
            "prompt": combined_prompt,
            "stream": False,
            "images": image_data,
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_predict": kwargs.get("max_tokens", 256)
            }
        }

        # logging
        print(payload)
        
        # Add seed if provided
        if kwargs.get("seed") is not None:
            payload["options"]["seed"] = kwargs.get("seed")
            
        return payload
    
    def make_request(self, payload: Dict[str, Any]) -> str:
        """Make request to Ollama API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url=self.api_url, headers=headers, json=payload).json()
        
        if 'error' in response:
            raise ValueError(f"Ollama API error: {response['error']}")
        
        return response.get('response', '')


def get_request_handler(model_name: str, **kwargs) -> ModelRequestHandler:
    """
    Factory function to get the appropriate request handler based on model name.
    
    Args:
        model_name: Name of the model to use
        **kwargs: Additional arguments to pass to the handler constructor
    
    Returns:
        An instance of ModelRequestHandler
    """
    if model_name in GPT_MODELS:
        return GPTRequestHandler(model_name, api_key=kwargs.get("api_key"))
    elif model_name in OLLAMA_MODELS:
        return OllamaRequestHandler(model_name, api_url=kwargs.get("api_url"))
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models are: {GPT_MODELS + OLLAMA_MODELS}")


def request_model(
    images: Union[Union[str, Image.Image, np.ndarray], List[Union[str, Image.Image, np.ndarray]]],
    prompt: str,
    system_prompt: str,
    model_name: str = "gpt-4o",
    **kwargs
) -> str:
    """
    Make a request to a model with images and text.
    
    Args:
        images: Single image or list of images (file paths, PIL Images, or numpy arrays)
        prompt: Text prompt to send to the model
        system_prompt: System instructions for the model
        model_name: Name of the model to use (from GPT_MODELS or OLLAMA_MODELS)
        **kwargs: Additional arguments:
            - temperature: Sampling temperature (default: 0.0)
            - max_tokens: Maximum number of tokens to generate (default: 256)
            - n: Number of completions to generate (GPT only, default: 1)
            - detail: Image detail level for GPT models (default: "auto")
            - return_logprobs: Whether to return log probabilities (GPT only, default: False)
            - in_context_examples: List of examples for in-context learning (GPT only)
            - seed: Optional seed for reproducible outputs
            - api_key: OpenAI API key (for GPT models)
            - api_url: Ollama API URL (for Ollama models)
    
    Returns:
        Model response as a string
    """
    handler = get_request_handler(model_name, **kwargs)
    return handler.request(images, prompt, system_prompt, **kwargs)
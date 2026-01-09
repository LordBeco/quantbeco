"""
OpenRouter AI Client

A Python client for OpenRouter API, providing access to various AI models
including free models that don't require API keys.
"""

import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from config import Config


@dataclass
class OpenRouterResponse:
    """Represents a response from OpenRouter AI"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class OpenRouterClient:
    """Python client for OpenRouter AI services"""
    
    # Available free models (prioritized by reliability)
    FREE_MODELS = [
        'nex-agi/deepseek-v3.1-nex-n1:free',
        'tngtech/deepseek-r1t-chimera:free',
        'deepseek/deepseek-r1-0528:free',
        'tngtech/deepseek-r1t2-chimera:free',
        'qwen/qwen3-4b:free'
    ]
    
    def __init__(self, api_key: Optional[str] = None, site_url: str = None, site_name: str = None):
        """Initialize OpenRouter client"""
        self.api_key = api_key
        self.site_url = site_url or Config.OPENROUTER_SITE_URL
        self.site_name = site_name or Config.OPENROUTER_SITE_NAME
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = requests.Session()
        
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "nex-agi/deepseek-v3.1-nex-n1:free",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs
    ) -> OpenRouterResponse:
        """
        Create a chat completion using OpenRouter API
        """
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        
        # Add authorization header if API key is provided
        if self.api_key and self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare request data
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            # Make API request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=60
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract content
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                usage = response_data.get('usage', {})
                
                return OpenRouterResponse(
                    content=content,
                    model=response_data.get('model', model),
                    usage=usage
                )
            else:
                raise ValueError("No valid response from OpenRouter API")
                
        except requests.exceptions.RequestException as e:
            # Handle network errors
            raise ConnectionError(f"Failed to connect to OpenRouter API: {str(e)}")
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            raise ValueError(f"Invalid response from OpenRouter API: {str(e)}")
        except Exception as e:
            # Handle other errors
            raise RuntimeError(f"OpenRouter API error: {str(e)}")


class OpenRouterClientWrapper:
    """Wrapper to make OpenRouterClient compatible with OpenAI client interface"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenRouterClient(api_key=api_key)
        
    @property
    def chat(self):
        return self
        
    @property  
    def completions(self):
        return self
        
    def create(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 2000, 
               temperature: float = 0.1, **kwargs):
        """Create completion compatible with OpenAI interface"""
        
        response = self.client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Return object that mimics OpenAI response structure
        class MockChoice:
            def __init__(self, content: str):
                self.message = type('Message', (), {'content': content})()
        
        class MockResponse:
            def __init__(self, content: str, model: str, usage: dict):
                self.choices = [MockChoice(content)]
                self.model = model
                self.usage = type('Usage', (), usage)() if usage else None
        
        return MockResponse(response.content, response.model, response.usage)
    
    @classmethod
    def get_available_models(cls) -> List[Dict[str, str]]:
        """Get list of available free models"""
        return [
            {
                'id': 'nex-agi/deepseek-v3.1-nex-n1:free',
                'name': 'DeepSeek v3.1 Nex N1 (Free)',
                'description': 'Latest DeepSeek model with enhanced performance - RECOMMENDED'
            },
            {
                'id': 'tngtech/deepseek-r1t-chimera:free',
                'name': 'DeepSeek R1T Chimera (Free)',
                'description': 'Reasoning-focused model for complex tasks'
            },
            {
                'id': 'deepseek/deepseek-r1-0528:free',
                'name': 'DeepSeek R1 (Free)',
                'description': 'Original DeepSeek reasoning model'
            },
            {
                'id': 'tngtech/deepseek-r1t2-chimera:free',
                'name': 'DeepSeek R1T2 Chimera (Free)',
                'description': 'Advanced reasoning model (may have intermittent issues)'
            },
            {
                'id': 'qwen/qwen3-4b:free',
                'name': 'Qwen3 4B (Free)',
                'description': 'Efficient Chinese-English bilingual model'
            }
        ]
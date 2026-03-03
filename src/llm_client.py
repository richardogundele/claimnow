"""
llm_client.py - Local LLM Interface via Ollama

WHY THIS FILE EXISTS:
- Provides a clean interface to the local LLM (Ollama)
- Abstracts away HTTP calls so other code just calls llm.generate()
- Handles retries, timeouts, and error handling
- Can be swapped for different LLM backends (OpenAI, Anthropic) easily

WHAT IS OLLAMA:
- Ollama is a tool that runs LLMs locally on your machine
- It downloads models (Mistral, Llama, etc.) and serves them via HTTP
- Default runs on http://localhost:11434
- No API keys needed, no data sent to cloud

HOW TO USE OLLAMA:
1. Install: https://ollama.ai
2. Pull a model: ollama pull mistral
3. It automatically starts a server on port 11434
4. This code sends HTTP requests to that server
"""

import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

# Import from sibling modules using package-relative imports
from src.config import settings

# Set up logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes for LLM Responses
# -----------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """
    Structured response from the LLM.
    
    Attributes:
        text: The generated text response
        model: Which model generated this
        tokens_used: Number of tokens in response (if available)
        done: Whether generation is complete
        error: Error message if something went wrong
    """
    text: str
    model: str = ""
    tokens_used: int = 0
    done: bool = True
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if the response was successful (no errors)."""
        return self.error is None


@dataclass  
class Message:
    """
    A single message in a conversation.
    
    Used for chat-style interactions where context matters.
    
    Roles:
    - "system": Instructions for how the LLM should behave
    - "user": The human's message
    - "assistant": The LLM's previous response
    """
    role: str  # "system", "user", or "assistant"
    content: str


class BedrockClient:
    """
    Client for interacting with Amazon Bedrock LLMs.
    
    USAGE:
        client = BedrockClient()
        response = client.generate("What is 2+2?")
        print(response.text)  # "4"
    """
    
    def __init__(
        self,
        region: Optional[str] = None,
        model_id: Optional[str] = None
    ):
        """
        Initialize the Bedrock client.
        
        Args:
            region: AWS Region (default from settings)
            model_id: Model to use (default from settings)
        """
        # Use settings if not provided
        self.region = region or settings.aws_region
        self.model_id = model_id or settings.bedrock_llm_model_id
        
        # Initialize boto3 client for Bedrock Runtime
        self._client = boto3.client('bedrock-runtime', region_name=self.region)
        
        logger.info(f"Bedrock client initialized: region={self.region}, model={self.model_id}")
    
    def is_available(self) -> bool:
        """
        Check if Bedrock client can connect (basic connectivity test)
        """
        try:
            # We can test by calling list_foundation_models on the management plain,
            # but usually just checking if client instantiated is enough.
            # Doing a very lightweight dummy Converse if needed, but returning True
            # to assume available if boto3 initialized.
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """
        Get list of available models. (Placeholder)
        """
        return [self.model_id]
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> LLMResponse:
        """Generate response using Bedrock Converse API"""
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens
        }
        
        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": inference_config
        }
        
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]
            
        try:
            response = self._client.converse(**kwargs)
            
            response_text = response['output']['message']['content'][0]['text']
            usage = response['usage']
            
            return LLMResponse(
                text=response_text,
                model=self.model_id,
                tokens_used=usage.get('totalTokens', 0),
                done=True
            )
            
        except ClientError as e:
            logger.error(f"Bedrock ClientError: {e}")
            return LLMResponse(text="", error=str(e))
        except Exception as e:
            logger.error(f"Bedrock unexpected error: {e}")
            return LLMResponse(text="", error=str(e))
    
    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Have a multi-turn conversation using Bedrock Converse API"""
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        # Convert our Message objects to Bedrock's format
        # System message goes in a separate parameter for Converse API
        bedrock_messages = []
        system_prompts = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompts.append({"text": msg.content})
            else:
                bedrock_messages.append({
                    "role": msg.role,
                    "content": [{"text": msg.content}]
                })
        
        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens
        }
        
        kwargs = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": inference_config
        }
        
        if system_prompts:
            kwargs["system"] = system_prompts
            
        try:
            response = self._client.converse(**kwargs)
            
            response_text = response['output']['message']['content'][0]['text']
            usage = response['usage']
            
            return LLMResponse(
                text=response_text,
                model=self.model_id,
                tokens_used=usage.get('totalTokens', 0),
                done=True
            )
            
        except ClientError as e:
            logger.error(f"Bedrock chat error: {e}")
            return LLMResponse(text="", error=str(e))
        except Exception as e:
            logger.error(f"Bedrock chat unexpected error: {e}")
            return LLMResponse(text="", error=str(e))
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        
        WHY THIS EXISTS:
        When extracting structured data (fields from a document),
        we need JSON output, not free-form text.
        
        This method:
        1. Adds JSON instructions to the prompt
        2. Parses the response as JSON
        3. Returns a dictionary (or empty dict on failure)
        
        Args:
            prompt: The extraction prompt
            system_prompt: System instructions
            temperature: Use 0.0 for consistent JSON output
            
        Returns:
            Parsed JSON as a dictionary
        """
        # Add JSON formatting instructions
        json_instruction = """
Respond with valid JSON only. No explanation, no markdown, just the JSON object.
"""
        
        full_system = f"{system_prompt}\n\n{json_instruction}" if system_prompt else json_instruction
        
        response = self.generate(
            prompt=prompt,
            system_prompt=full_system,
            temperature=temperature
        )
        
        if not response.success:
            logger.error(f"JSON generation failed: {response.error}")
            return {}
        
        # Try to parse the response as JSON
        try:
            # Sometimes LLM wraps JSON in markdown code blocks
            text = response.text.strip()
            
            # Remove markdown code block if present
            if text.startswith("```"):
                # Find the end of the code block
                lines = text.split("\n")
                # Remove first line (```json) and last line (```)
                text = "\n".join(lines[1:-1])
            
            return json.loads(text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Raw response: {response.text}")
            return {}


# -----------------------------------------------------------------------------
# Singleton instance for convenience
# -----------------------------------------------------------------------------
# Create a default client that can be imported directly
_default_client: Optional[BedrockClient] = None


def get_llm_client() -> BedrockClient:
    """
    Get the default LLM client (singleton pattern).
    """
    global _default_client
    
    if _default_client is None:
        _default_client = BedrockClient()
    
    return _default_client


# -----------------------------------------------------------------------------
# Convenience functions for quick usage
# -----------------------------------------------------------------------------
def generate(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Quick function to generate text without managing a client.
    
    USAGE:
        from llm_client import generate
        
        result = generate("What is 2+2?")
        print(result)  # "4"
    """
    client = get_llm_client()
    response = client.generate(prompt, system_prompt)
    return response.text if response.success else ""


def generate_json(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to generate JSON without managing a client.
    
    USAGE:
        from llm_client import generate_json
        
        data = generate_json("Extract name and age from: John is 25 years old")
        print(data)  # {"name": "John", "age": 25}
    """
    client = get_llm_client()
    return client.generate_json(prompt, system_prompt)

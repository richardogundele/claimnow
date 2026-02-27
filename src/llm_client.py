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

import httpx

from config import settings

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


class OllamaClient:
    """
    Client for interacting with Ollama's local LLM server.
    
    USAGE:
        client = OllamaClient()
        response = client.generate("What is 2+2?")
        print(response.text)  # "4"
    
    CHAT USAGE (with context):
        messages = [
            Message("system", "You are a helpful assistant."),
            Message("user", "What is the capital of France?")
        ]
        response = client.chat(messages)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Ollama server URL (default from settings)
            model: Model to use (default from settings)
            timeout: Request timeout in seconds (LLMs can be slow)
        """
        # Use settings if not provided
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout
        
        # httpx is a modern HTTP client (like requests but better)
        # We create a client instance for connection pooling
        self._client = httpx.Client(timeout=timeout)
        
        logger.info(f"Ollama client initialized: {self.base_url}, model={self.model}")
    
    def __del__(self):
        """Clean up HTTP client when object is destroyed."""
        if hasattr(self, '_client'):
            self._client.close()
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is running and reachable.
        
        WHY: Good to check before processing a whole document
        Returns True if server responds, False otherwise
        """
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns names of models that have been pulled/downloaded.
        """
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            # Extract model names from response
            models = [model["name"] for model in data.get("models", [])]
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        This is the MAIN METHOD for single-turn generation.
        
        Args:
            prompt: The user's input/question
            system_prompt: Instructions for how LLM should behave
            temperature: Randomness (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum response length
            stream: Whether to stream the response (not implemented here)
            
        Returns:
            LLMResponse with the generated text
            
        EXAMPLE:
            response = client.generate(
                prompt="Extract the date from: Invoice dated March 15, 2024",
                system_prompt="You are a data extraction assistant. Return only the extracted value.",
                temperature=0.0
            )
            print(response.text)  # "March 15, 2024"
        """
        # Use defaults from settings if not provided
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        # Build the request payload
        # Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get complete response at once
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,  # Ollama calls it num_predict
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            # Send POST request to Ollama's generate endpoint
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            return LLMResponse(
                text=data.get("response", ""),
                model=data.get("model", self.model),
                tokens_used=data.get("eval_count", 0),
                done=data.get("done", True)
            )
            
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return LLMResponse(
                text="",
                error="Request timed out. The model may be loading or overloaded."
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            return LLMResponse(
                text="",
                error=f"HTTP error: {e.response.status_code}"
            )
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return LLMResponse(
                text="",
                error=str(e)
            )
    
    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Have a multi-turn conversation with the LLM.
        
        This preserves context from previous messages.
        
        Args:
            messages: List of Message objects (conversation history)
            temperature: Randomness setting
            max_tokens: Maximum response length
            
        Returns:
            LLMResponse with the assistant's reply
            
        EXAMPLE:
            messages = [
                Message("system", "You are an insurance claims analyst."),
                Message("user", "Is £80/day fair for a Group B car?"),
            ]
            response = client.chat(messages)
            
            # Continue the conversation
            messages.append(Message("assistant", response.text))
            messages.append(Message("user", "What about in London specifically?"))
            response = client.chat(messages)
        """
        temperature = temperature if temperature is not None else settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        # Convert our Message objects to Ollama's format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Chat endpoint returns message in different structure
            message = data.get("message", {})
            
            return LLMResponse(
                text=message.get("content", ""),
                model=data.get("model", self.model),
                tokens_used=data.get("eval_count", 0),
                done=data.get("done", True)
            )
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
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
_default_client: Optional[OllamaClient] = None


def get_llm_client() -> OllamaClient:
    """
    Get the default LLM client (singleton pattern).
    
    USAGE:
        from llm_client import get_llm_client
        
        client = get_llm_client()
        response = client.generate("Hello!")
    
    WHY SINGLETON:
    - Reuses HTTP connection pool
    - Avoids creating multiple clients
    - Consistent configuration
    """
    global _default_client
    
    if _default_client is None:
        _default_client = OllamaClient()
    
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

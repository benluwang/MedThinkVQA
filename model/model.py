# model.py
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import logging
import warnings

logger = logging.getLogger(__name__)

class LLM(ABC):
    """
    Abstract base class for Large Language Models.
    Defines the interface that all LLM implementations should follow.
    """
    def __init__(self, model_name: str,
                 max_tokens: int = 20000,
                 temperature: float = 0.6,
                 seed: Optional[int] = None,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.0):
        """
        Initialize an LLM instance.
        
        Args:
            model_name: Name of the model in the format 'Provider/ModelName'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            seed: Random seed for reproducibility
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
        """

        if '/' not in model_name:
            warnings.warn("model_name should be in the format 'Provider/ModelName'", UserWarning)
        
        self.model_name = model_name.split('/')[-1] if '/' in model_name else model_name
        self.model_type = model_name.split('/')[0]  # e.g., 'OpenAI'
        self.model_name_full = model_name
        self.max_tokens = max(max_tokens,17)
        self.temperature = temperature
        self.seed = seed
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
    
    @abstractmethod
    def generate(self, prompts: List[Tuple[str, str]], batch_size: int = 1) -> List[Tuple[str, int, int]]:
        """
        Generate responses for a list of prompts.
        
        Args:
            prompts: List of (system_msg, usr_msg) tuples
            batch_size: Number of prompts to process at once
        
        Returns:
            List of generated responses, input tokens, and output tokens.
        """
        pass
    
    def compute_tokens(self, response) -> int:
        """
        Compute the number of tokens in a string or a tuple of strings using the model's tokenizer.
        
        Args:
            response: The string or tuple of strings to compute tokens for.
            
        Returns:
            Number of tokens in the input.
        """
        try:
            # If it's a tuple, recurse into each element
            if isinstance(response, tuple):
                return sum(self.compute_tokens(item) for item in response)

            # Ensure we are working with a string to avoid encoding errors
            response_str = str(response)

            # Tokenize and count
            tokens = self.tokenizer.encode(response_str)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error computing tokens: {str(e)}")
            return 0

    
    def get_model_name(self) -> str:
        """
        Returns the full model name that should be used for logging purposes.
        
        Returns:
            Full model name string
        """
        return self.model_name_full
    
    def get_model_type(self) -> str:
        """
        Returns the model type (e.g., OpenAI, HuggingFace) for logging purposes.
        
        Returns:
            Model type string
        """
        return self.model_type
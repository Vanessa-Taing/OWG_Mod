from typing import Dict, Any, List, Optional
import os

class SystemPromptLibrary:
    """
    Manages loading and formatting of prompt templates, including support for
    system prompt variants (e.g., for uncertainty-aware evaluation).
    """
    def __init__(self, prompt_dir: str):
        """
        Args:
            prompt_dir (str): Directory where all `.txt` prompt templates are stored.
        """
        self.prompt_dir = prompt_dir
        print(f"Debugging: {prompt_dir}")
        self.prompts = self._load_all_prompts()
    
    def _load_all_prompts(self) -> Dict[str, str]:
        """Loads all .txt prompt files in the prompt directory into a dictionary."""
        prompts = {}
        for filename in os.listdir(self.prompt_dir):
            if filename.endswith(".txt"):
                name = filename.replace(".txt", "")
                with open(os.path.join(self.prompt_dir, filename), 'r') as f:
                    prompts[name] = f.read()
                print(f"[PromptLibrary] Loaded prompt: {name}")
        return prompts
    
    def list_available_prompts(self) -> List[str]:
        """Returns a list of all loaded prompt names (without `.txt`)."""
        return list(self.prompts.keys())
    
    def load_prompt(self, name: str) -> Optional[str]:
        """
        Retrieves the raw prompt text by name.
        Args:
            name (str): The name of the prompt (excluding `.txt`).
        Returns:
            Optional[str]: The raw prompt string if found, else None.
        """
        # Strip .txt extension if provided
        prompt_name = name.replace(".txt", "")
        return self.prompts.get(prompt_name)
    
    def inject_uncertainty_info(self, text: str) -> str:
        """
        Hook to inject uncertainty annotations or metadata into prompt text.
        Override as needed.
        Args:
            text (str): The prompt string after formatting.
        Returns:
            str: Final prompt string with uncertainty info (if any).
        """
        return text
    
    def prepare_prompt(self, name: str, variables: Dict[str, Any]) -> str:
        """
        Loads and formats a prompt with the given variables.
        Args:
            name (str): Name of the prompt (without `.txt`).
            variables (Dict[str, Any]): Mapping for formatting placeholders.
        Returns:
            str: Formatted and processed prompt string.
        Raises:
            ValueError: If the prompt is not found or formatting fails.
        """
        prompt = self.load_prompt(name)
        if not prompt:
            raise ValueError(f"Prompt '{name}' not found in the prompt directory.")
        try:
            filled = prompt.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable for prompt '{name}': {e}")
        return self.inject_uncertainty_info(filled)
    
    def prepare_variant_prompts(
        self,
        base_name: str,
        variants: Optional[List[str]] = None,
        variables: Dict[str, Any] = {}
    ) -> Dict[str, str]:
        """
        Prepares variant system prompts based on a base prompt name and formatting variables.
        Args:
            base_name (str): Base prompt name prefix (e.g., 'referring_segmentation').
            variants (Optional[List[str]]): List of suffixes to use (e.g., ['_hedging', '_confidence']).
                                            If None, defaults to ['_base'].
            variables (Dict[str, Any]): Variables to inject into each prompt template.
        Returns:
            Dict[str, str]: Dictionary mapping variant suffixes to fully formatted prompts.
        """
        if variants is None:
            variants = ["_base"]
        prompts = {}
        for suffix in variants:
            prompt_name = f"{base_name}{suffix}"
            try:
                prompts[suffix] = self.prepare_prompt(prompt_name, variables)
            except ValueError as e:
                print(f"[PromptLibrary] Error loading variant '{suffix}': {e}")
                continue
        return prompts
    
    def read_prompt_from_file(self, name: str) -> Optional[str]:
        filename = name if name.endswith(".txt") else f"{name}.txt"
        path = os.path.join(self.prompt_dir, filename)

        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[PromptLibrary] read error {name}: {e}")
            return None

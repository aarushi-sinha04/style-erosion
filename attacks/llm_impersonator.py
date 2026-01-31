import os
import random

class LLMImpersonator:
    """
    Impersonation Attack using LLMs (e.g., GPT-4 or Local Llama).
    Goal: Rewrite a text to mimic the style of a target author/domain.
    """
    def __init__(self, provider="openai", model_name="gpt-4", api_key=None):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if provider == "openai" and not self.api_key:
            print("Warning: No API Key found for OpenAI.")
            
    def impersonate(self, source_text, target_samples, method="few-shot"):
        """
        Rewrite source_text to match style of target_samples.
        
        Args:
            source_text: The text to rewrite (content source).
            target_samples: List[str] of texts from the target author.
            method: 'few-shot' or 'style-description'.
        """
        if method == "few-shot":
            prompt = self._construct_few_shot_prompt(source_text, target_samples)
        else:
            prompt = self._construct_desc_prompt(source_text)
            
        return self._generate(prompt)
    
    def _construct_few_shot_prompt(self, source, targets):
        """
        Constructs a prompt with examples of the target style.
        """
        # Select 1-3 random samples
        samples = random.sample(targets, min(len(targets), 3))
        
        prompt = "You are a skilled copyist. Rewrite the Input Text to mimic the writing style of the Reference Samples exactly.\n\n"
        prompt += "Reference Samples:\n"
        for i, s in enumerate(samples):
            prompt += f"Sample {i+1}: {s[:300]}...\n" # Truncate for token limit
            
        prompt += f"\nInput Text: {source}\n"
        prompt += "Rewritten Text:"
        return prompt
    
    def _generate(self, prompt):
        """
        Calls the LLM API.
        """
        if self.provider == "openai":
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM Generation Failed: {e}")
                return "[FAILED]"
        else:
            # Placeholder for local models
            return "[LOCAL MODEL NOT IMPLEMENTED]"

if __name__ == "__main__":
    # Test
    attacker = LLMImpersonator()
    src = "I really enjoyed this movie, it was great."
    tgts = ["Verily, the cinematic experience was of the utmost quality.", "The film was exquisite, truly."]
    print(attacker.impersonate(src, tgts))

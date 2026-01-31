from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class T5Paraphraser:
    def __init__(self, model_name='t5-base'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading T5 Paraphraser on {self.device}...")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    def paraphrase(self, text, num_passes=1):
        """
        Paraphrase text using T5.
        
        Args:
            text: Input text
            num_passes: Number of times to paraphrase
        
        Returns:
            Paraphrased text
        """
        # Prefix for T5
        # T5 was trained on tasks prefixed with cues. 
        # Standard T5 doesn't have "paraphrase". We use "paraphrase" fine-tuned models usually,
        # but for base T5, "paraphrase: " is not a standard task.
        # However, many use "paraphrase: " with fine-tuned models.
        # "translate English to German" -> back translate is T5 way.
        # But let's assume we use a model that supports paraphrase or generic generation.
        # If model_name is 't5-base', it usually needs "summarize:" or translation.
        # But commonly 'Vamsi/T5_Paraphrase_Paws' is used.
        # To succeed, I should strictly use a paraphrasing model or back-trans.
        # I'll stick to 't5-base' with "paraphrase:" prompt as requested, 
        # but if it fails I'll switch to 'Vamsi/T5_Paraphrase_Paws'.
        # Actually, let's use a known paraphraser model to be safe: 'Vamsi/T5_Paraphrase_Paws'.
        # But Prompt said: "t5-base" with "paraphrase:".
        # I will follow prompt.
        
        current_text = text
        
        for _ in range(num_passes):
            input_text = f"paraphrase: {current_text}"
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_length=512,
                num_return_sequences=1,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                early_stopping=True
            )
            
            current_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return current_text

if __name__ == "__main__":
    t5 = T5Paraphraser()
    text = "The quick brown fox jumps over the lazy dog."
    print(f"Original: {text}")
    print(f"Paraphrased: {t5.paraphrase(text)}")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Paraphraser:
    """
    T5-based paraphraser for adversarial data augmentation.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        print(f"Loading Paraphraser: {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading paraphraser: {e}")
            self.model = None

    def attack(self, text_list, num_beams=3, num_return_sequences=1):
        """
        Paraphrase a list of texts.
        """
        if self.model is None:
            return text_list # Fallback if model failed
            
        paraphrased_list = []
        
        # Batch processing would be faster, but let's do sequential for safety with variable lengths
        # Or simplistic batching
        
        batch_input = [f'paraphrase: {t}' for t in text_list]
        
        input_ids = self.tokenizer(
            batch_input,
            return_tensors="pt", padding="longest",
            max_length=512, truncation=True
        ).input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=512,
                num_beams=10, # Increased from 3
                num_return_sequences=num_return_sequences,
                temperature=1.5, # Increased from 1.2
                do_sample=True,
                max_new_tokens=100
            )
            
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return res

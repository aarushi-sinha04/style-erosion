import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer

PARAPHRASE_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"
TRANSLATE_EN_DE = "Helsinki-NLP/opus-mt-en-de"
TRANSLATE_DE_EN = "Helsinki-NLP/opus-mt-de-en"
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class MultiPassParaphraser:
    """Multi-Pass Paraphrasing: Apply T5 paraphraser multiple times."""
    def __init__(self, device=DEVICE):
        self.device = device
        print(f"Loading Paraphraser: {PARAPHRASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to(self.device)
        self.model.eval()
        
    def attack(self, text, passes=3):
        result = text
        for _ in range(passes):
            inputs = self.tokenizer(f'paraphrase: {result}', return_tensors="pt",
                                    max_length=512, truncation=True).input_ids.to(self.device)
            outputs = self.model.generate(inputs, max_length=512, num_beams=5,
                                          do_sample=True, temperature=2.0,
                                          top_k=50, top_p=0.95)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

class BackTranslationAttack:
    """Back-Translation: EN -> German -> EN"""
    def __init__(self, device=DEVICE):
        self.device = device
        print("Loading Translation Models (EN<->DE)...")
        self.en_de_tok = MarianTokenizer.from_pretrained(TRANSLATE_EN_DE)
        self.en_de_model = MarianMTModel.from_pretrained(TRANSLATE_EN_DE).to(self.device)
        self.de_en_tok = MarianTokenizer.from_pretrained(TRANSLATE_DE_EN)
        self.de_en_model = MarianMTModel.from_pretrained(TRANSLATE_DE_EN).to(self.device)
        
    def attack(self, text):
        # EN -> DE
        inputs = self.en_de_tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        de_ids = self.en_de_model.generate(**inputs, max_length=512)
        de_text = self.en_de_tok.decode(de_ids[0], skip_special_tokens=True)
        
        # DE -> EN
        inputs = self.de_en_tok(de_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        en_ids = self.de_en_model.generate(**inputs, max_length=512)
        return self.de_en_tok.decode(en_ids[0], skip_special_tokens=True)

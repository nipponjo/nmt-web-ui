import torch
import numpy as np
import huggingface_hub
from transformers import MarianMTModel, MarianTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from utils.iso_dict import code_dict, rtl_languages


def get_model(src, trg, mod='mt', cache_dir=None, device='cpu'):
    prefix = {'mt': 'mt', 'big': 'mt-tc-big', 
              'tatoeba': 'tatoeba', 'base': 'mt-tc-base'}[mod]
    model_name = f"Helsinki-NLP/opus-{prefix}-{src}-{trg}"
    tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device)
    return model, tokenizer

def infer_nmt(model, tokenizer, text_src, max_length=512, device='cpu'):
    token_dict = tokenizer(text_src, return_tensors="pt", padding=True)
    if device != 'cpu': token_dict = {k:v.to(device) for k, v in token_dict.items()}
    translated = model.generate(**token_dict, max_length=max_length)
    text_trg = tokenizer.decode(translated[0], skip_special_tokens=True)
    return text_trg


class LanguageInformer:
    def __init__(self) -> None:
        opus_models = list(huggingface_hub.list_models(author='Helsinki-NLP'))        
        self.minfo_triples = [model_info.modelId.split('-')[-3:] for model_info in opus_models][::-1]        
        self.code_dict = code_dict

        src_langs_ = set([x[1] for x in self.minfo_triples])
        self.src_langs = sorted([f"{ self.code_dict[c][0]} [{c}]" for c in src_langs_ if c in self.code_dict])

    def get_src_languages(self):
        return self.src_langs

    def get_trg_languages(self, src_lang):
        trg_langs_ = [(x[2], x[0]) for x in self.minfo_triples if x[1] == src_lang]
        trg_langs = sorted([f"{self.code_dict[c][0]} [{c}, {mod}]" 
                           for c, mod in trg_langs_ if c in self.code_dict])
        return trg_langs
    
    def is_rtl_language(self, language):
        return language in rtl_languages


class SaveLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.score_list = []
    
    def _reset_list(self, cutoff=0):
        self.score_list = self.score_list[:cutoff]

    def __call__(self, input_ids, scores):
        self.score_list.append(
            {'ids': input_ids.cpu(),  'logits': scores.cpu()})
        return scores

class TranslationManager:
    def __init__(self, device=None) -> None:

        model_dict = {}

        self.model_dict = model_dict

        self.current_model_id = None
        self.current_src_text = None
        self.current_decoder_ids = None
        self.current_tokens = None
        self.current_tokenizer = None
        self.current_num_beams = None
        
        self.token_idx = None

        self.logits_store = SaveLogitsProcessor()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' \
                             if device is None else device


    def load_model(self, src, trg, mod='mt'):        
        model_id = f'{src}->{trg} ({mod})'
        if model_id in self.model_dict:
            self.current_model_id = model_id
            return

        print(f'Loading model {model_id} ...')
        self.model_dict[model_id] = get_model(src, trg, mod, 
                                              device=self.device)
        print(f"Loaded {model_id} model!")
        self.current_model_id = model_id
    
    def unload_model(self, src=None, trg=None, mod='mt'):
        if src is None or trg is None:
            model_id = self.current_model_id
        else:
            model_id = f'{src}->{trg} ({mod})'
        model, tokenizer = self.model_dict[model_id]
        del model
        del tokenizer
        del self.model_dict[model_id]
        torch.cuda.empty_cache()
        print(f"Unloaded {model_id} model!")

    def _store_tokens(self, decoder_ids, tokenizer) -> None:
        self.current_decoder_ids = decoder_ids.cpu()
        tokens_out = [tokenizer._convert_id_to_token(id) for id in decoder_ids[0].tolist()]
        print(tokens_out)
        tokens_out = [tok for tok in tokens_out if tok not in tokenizer.all_special_tokens]

        self.current_tokens = tokens_out

    def _get_token_idx_at(self, pos: int) -> int:
        tokens_len = np.array([len(tok) for tok in self.current_tokens])
        tokens_cumsum = np.cumsum(tokens_len)
        token_idx = np.searchsorted(tokens_cumsum, pos)
        return token_idx
    
    def _alternative_tokens(self, token_idx, topk: int = 10):
        if self.current_num_beams == 1:
            scores = self.logits_store.score_list[token_idx]['logits'][0]
            topk_res = scores.topk(topk)
        else:
            entry_prev = self.logits_store.score_list[token_idx]
            match_prev = entry_prev['ids'] == self.current_decoder_ids[:,:token_idx+1]
            match_prev = match_prev.all(1).int().argmax()
            scores =  entry_prev['logits'][match_prev]
        
        topk_res = scores.topk(topk)
        alternative_tokens = [self.current_tokenizer._convert_id_to_token(id) 
                              for id in topk_res.indices.tolist()]            

        self.token_idx = token_idx
        return alternative_tokens
    
    def _alternative_tokens_at(self, pos: int, topk: int = 10):
        token_idx = self._get_token_idx_at(pos)
        return self._alternative_tokens(token_idx, topk=topk)
    
    def _change_token(self, new_token):
        if self.token_idx is None:
            return
        
        new_token_id = self.current_tokenizer.encoder.get(new_token)
        new_decoder_ids = self.current_decoder_ids[:, :self.token_idx+2]
        new_decoder_ids[:, -1] = new_token_id

        return self.__call__(decoder_input_ids=new_decoder_ids)        
                
    def __call__(self, text_src=None, num_beams=4, 
                 max_length=512, decoder_input_ids=None) -> str:
        device = self.device

        if text_src is None: text_src = self.current_src_text 
        self.current_src_text = text_src
        self.current_num_beams = num_beams

        model, tokenizer = self.model_dict[self.current_model_id]
        self.current_tokenizer = tokenizer

        token_dict = tokenizer(text_src, return_tensors="pt", padding=True)

        store_ofx = 0
        if decoder_input_ids is not None:
            token_dict['decoder_input_ids'] = decoder_input_ids
            store_ofx = decoder_input_ids.size(-1) - 1

        if device != 'cpu': token_dict = {k: v.to(device) for k, v in token_dict.items()}

        self.logits_store._reset_list(store_ofx)
        logits_processor = LogitsProcessorList([self.logits_store])

        decoder_ids = model.generate(**token_dict, 
                                     max_length=max_length, 
                                     logits_processor=logits_processor, 
                                     num_beams=num_beams)

        text_trg = tokenizer.decode(decoder_ids[0], skip_special_tokens=True)

        self._store_tokens(decoder_ids, tokenizer)

        return text_trg
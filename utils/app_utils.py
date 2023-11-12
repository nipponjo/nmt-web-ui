import torch
import numpy as np
import huggingface_hub
from transformers import MarianMTModel, MarianTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from utils.iso_dict import code_dict, rtl_languages
from typing import Optional
from torch import Tensor

def get_model(src: str, trg: str, 
              version: str = 'mt', 
              cache_dir: Optional[str] = None, 
              device: torch.device = 'cpu'
              ) -> tuple[MarianMTModel, MarianTokenizer]:
    """
    Loads a https://huggingface.co/Helsinki-NLP model from the Huggingface hub

    Args:
        src: Language code for the source language.
        trg: Language code for the target language.
        version: Model version, options: `[mt, big, tatoeba, base]`
        cache_dir: Path to a directory in which a downloaded predefined 
            tokenizer vocabulary files should be cached if the 
            standard cache should not be used.            
        device: Device to which the model should be moved.
    """
    prefix = {'mt': 'mt', 'big': 'mt-tc-big', 
              'tatoeba': 'tatoeba', 'base': 'mt-tc-base'}[version]
    model_name = f"Helsinki-NLP/opus-{prefix}-{src}-{trg}"
    tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device)
    return model, tokenizer

def remove_underscore(tokens: list[str]) -> list[str]:
    """Removes the `▁` from tokens or otherwise prepends a `-`"""
    return [token[1:] if token.startswith('▁') else f"-{token}" for token in tokens]


def get_caa(cattns: list[list[Tensor]], 
            tokens_in: list[str],
            tokens_out: list[str],
            beam_ids: Optional[Tensor] = None,
            layer: int = 0,
            top_k: int = 4,
            ) -> list[str]:
    """
        attns: [out_id][layer][batch/beam,head,.,in_id]
        tokens_in:
        tokens_out:
        beam_ids:
        layer:
        top_k:
    """
    # 
    tokens_in = remove_underscore(tokens_in)
    tokens_out = remove_underscore(tokens_out)
    top_k = min(top_k, len(tokens_in))

    otoken_to_cattn_list = []

    dec_step = 0
    for out_token_idx in range(len(tokens_out)-1):  
        beam_id = beam_ids[dec_step] if beam_ids is not None else 0

        heads_per_oid = cattns[dec_step][layer][beam_id].cpu()
        n_dec_in = heads_per_oid.size(1)
        didx = out_token_idx - dec_step if n_dec_in > 1 else 0    
    
        heads_per_oid = heads_per_oid[:, didx]
        if didx == n_dec_in - 1: dec_step += 1    
        
        mean_ca_per_oid = heads_per_oid.mean(0)

        ca_sort_values, ca_sort_ids = mean_ca_per_oid.topk(top_k)
        ca_sort_tokens = np.array(tokens_in)[ca_sort_ids.numpy()]

        tok_dict = {tok: val for tok, val in 
                    zip(ca_sort_tokens, ca_sort_values.tolist()) 
                    if tok != '-</s>'}
        
        otoken_to_cattn_list.append((tokens_out[out_token_idx], tok_dict))

    caa_res = otoken_to_cattn_list

    caa_res_ = [(tok, ', '.join(f"{t} ({v:.2f})" for t,v in ca.items())) for tok, ca in caa_res]

    return caa_res_


class LanguageInformer:
    def __init__(self) -> None:
        opus_models = list(huggingface_hub.list_models(author='Helsinki-NLP'))
        # modelId: e.g. Helsinki-NLP/opus-mt-rn-de or opus-mt-tc-base-uk-hu
        # -> minfo_triple: ('mt', 'de', 'rn') or ('base', 'uk', 'hu')
        self.minfo_triples = [model_info.modelId.split('-')[-3:] for model_info in opus_models]      
        self.code_dict = code_dict # entries like 'ko': ['Korean', 'ko', 'kor']

        src_langs_ = set([src for _, src, _ in self.minfo_triples])
        self.src_langs = sorted([
            {'code': c, 'name': f"{ self.code_dict[c][0]} [{c}]"}
            for c in src_langs_ if c in self.code_dict], key=lambda x: x['name'])

    def get_src_languages(self) -> list[dict]:
        return self.src_langs

    def get_trg_languages(self, src_lang: str) -> list[dict]:
        trg_langs_ = [(trg, mod) for mod, src, trg in self.minfo_triples if src == src_lang]
        trg_langs = sorted([
            {'code': f"{c}-{mod}", 'name': f"{self.code_dict[c][0]} [{c}, {mod}]"}
            for c, mod in trg_langs_ if c in self.code_dict], key=lambda x: x['name'])
        return trg_langs


class SaveLogitsProcessor(LogitsProcessor):
    def __init__(self) -> None:
        self.score_list = []
    
    def _reset_list(self, cutoff: int = 0) -> None:
        self.score_list = self.score_list[:cutoff]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.score_list.append(
            {'ids': input_ids.cpu(),  'logits': scores.cpu()})
        return scores

class TranslationManager:
    def __init__(self, device: torch.device = None) -> None:

        model_dict = {}

        self.model_dict = model_dict

        self.current_model_id = None
        self.current_src_text = None
        self.current_decoder_ids = None
        self.current_tokens = None
        self.current_tokenizer = None
        self.current_num_beams = None        
        
        self.token_idx = None

        self.caa_active = False # cross-attention analysis (caa)
        self.current_caa = []

        self.logits_store = SaveLogitsProcessor()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' \
                             if device is None else device

    def load_model(self, src: str, trg: str, mod: str = 'mt') -> None:        
        model_id = f'{src}->{trg} ({mod})'
        if model_id in self.model_dict:
            self.current_model_id = model_id
            return

        print(f'Loading model {model_id} ...')
        self.model_dict[model_id] = get_model(src, trg, mod, 
                                              device=self.device)
        print(f"Loaded {model_id} model!")        
        self.current_model_id = model_id
        if self.caa_active: self._set_caa(on=True)
    
    def unload_model(self, 
                     src: Optional[str] = None, 
                     trg: Optional[str] = None, 
                     mod: str = 'mt') -> None:
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

    def _store_tokens(self, 
                      decoder_ids: Tensor, 
                      tokenizer: MarianTokenizer) -> None:
        """Stores decoder_ids and tokens"""
        self.current_decoder_ids = decoder_ids.cpu()
        tokens_out = [tokenizer._convert_id_to_token(id) for id in decoder_ids[0].tolist()]
        print(tokens_out)
        tokens_out = [tok for tok in tokens_out if tok not in tokenizer.all_special_tokens]

        self.current_tokens = tokens_out

    def _get_token_idx_at(self, char_pos: int) -> int:
        """Returns token index at character position `char_pos`"""
        tokens_len = np.array([len(token) for token in self.current_tokens])
        if self.current_tokens[0].startswith('▁'): tokens_len[0] -= 1 # _ at beginning is ignored
        tokens_cumsum = np.cumsum(tokens_len)
        token_idx = np.searchsorted(tokens_cumsum, char_pos)

        return token_idx
    
    def _alternative_tokens(self, 
                            token_idx: int, 
                            topk: int = 10) -> list[str]:
        """Returns alternative tokens for the token at token index `token_idx`"""
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
    
    def _alternative_tokens_at(self, 
                               char_pos: int, 
                               topk: int = 10) -> list[str]:
        """Returns alternative tokens for the token at character position `char_pos`"""
        token_idx = self._get_token_idx_at(char_pos)
        return self._alternative_tokens(token_idx, topk=topk)
    
    def _change_token(self, new_token: str) -> dict:
        if self.token_idx is None:
            return
        
        new_token_id = self.current_tokenizer.encoder.get(new_token)
        new_decoder_ids = self.current_decoder_ids[:, :self.token_idx+2] # +2 because of <pad> token at start 
        new_decoder_ids[:, -1] = new_token_id # replace token id

        return self.__call__(decoder_input_ids=new_decoder_ids)   

    def _set_caa(self, on: bool = True) -> None:     
        """Switch cross-attention analysis on or off"""
        if not self.model_dict:
            return
        model, _ = self.model_dict[self.current_model_id]
        model.generation_config.output_attentions = on
        model.generation_config.output_scores = on
        model.generation_config.return_dict_in_generate = on
        model.translated = on
        self.caa_active = on     
                
    def __call__(self,
                 text_src: Optional[str] = None, 
                 num_beams: int = 4, 
                 max_length: int = 512, 
                 decoder_input_ids: Optional[torch.LongTensor] = None
                 ) -> dict:
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

        output = model.generate(**token_dict, 
                                max_length=max_length, 
                                logits_processor=logits_processor, 
                                num_beams=num_beams)
        decoder_ids = output.sequences if self.caa_active else output

        return_dict = {}

        text_trg = tokenizer.decode(decoder_ids[0], skip_special_tokens=True)
        return_dict['text_trg'] = text_trg

        self._store_tokens(decoder_ids, tokenizer)

        if self.caa_active:
            tokens_in = [tokenizer._convert_id_to_token(id) for id in token_dict['input_ids'][0].tolist()]
            tokens_out = [tokenizer._convert_id_to_token(id) for id in decoder_ids[0].tolist()]
            cattn = output.cross_attentions

            beam_ids = output.beam_indices[0].cpu().tolist() if num_beams > 1 else None
            caa = get_caa(cattn, tokens_in, tokens_out, beam_ids)            
            self.current_caa = caa   

            return_dict['caa'] = self.current_caa
        
        return return_dict
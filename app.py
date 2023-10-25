import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from utils.app_utils import TranslationManager, LanguageInformer

class TranslRequest(BaseModel):
    text: str
    src_lang: str
    trg_lang: str
    n_beams: int

app = FastAPI()

lang_informer = LanguageInformer()
translator = TranslationManager()

# app.add_middleware(
#     CORSMiddleware,
#         allow_origins=['*'],
#         allow_credentials=True,
#         allow_methods=['*'], 
#         allow_headers=['*'],
#         )

@app.get('/')
def index():
    return FileResponse('./app/index.html')

@app.get('/api/get-src-langs')
def get_src_languages():
    src_langs = lang_informer.get_src_languages()
    return {'src_langs': src_langs}

@app.get('/api/get-trg-langs')
def get_trg_languages(srclang: str):
    lang_code = srclang.split('[')[-1].removesuffix(']')
    trg_langs = lang_informer.get_trg_languages(lang_code)
    return {'trg_langs': trg_langs}

@app.post('/api/translate')
async def translate(request: TranslRequest):
    print(request)
    src = request.src_lang.split('[')[-1].removesuffix(']')
    trg_ = request.trg_lang.split('[')[-1].removesuffix(']')
    trg, mod = trg_.split(', ')
    translator.load_model(src, trg, mod)
    text_src = request.text
    text_trg = translator(text_src, num_beams=request.n_beams)
    return {'text_trg': text_trg}

@app.get('/api/get-alt-tokens')
def get_alt_tokens(pos: int):
    alternative_tokens = translator._alternative_tokens_at(pos)
    return {'alt': alternative_tokens}

@app.get('/api/change-token')
def change_token(token: str):
    text_trg = translator._change_token(token)
    return {'alt': text_trg}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

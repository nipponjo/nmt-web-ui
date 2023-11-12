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
async def index():
    return FileResponse(f'./app/index.html')

@app.get('/static/{filename}')
async def static_file(filename: str):
    return FileResponse(f'./app/static/{filename}')

@app.get('/api/get-src-langs')
async def get_src_languages():
    src_langs = lang_informer.get_src_languages()
    return {'src_langs': src_langs}

@app.get('/api/get-trg-langs')
async def get_trg_languages(srclang: str):    
    trg_langs = lang_informer.get_trg_languages(srclang)
    return {'trg_langs': trg_langs}

@app.get('/api/set-caa')
async def set_caa(mode: str):
    translator._set_caa(on=mode=='on')
    return {}

@app.post('/api/translate')
async def translate(request: TranslRequest):
    print(request)        
    trg, mod = request.trg_lang.split('-')
    translator.load_model(request.src_lang, trg, mod)
    text_src = request.text
    return_dict = translator(text_src, num_beams=request.n_beams)
    return return_dict

@app.get('/api/get-alt-tokens')
async def get_alt_tokens(pos: int):
    alternative_tokens = translator._alternative_tokens_at(pos)
    return {'alt': alternative_tokens}

@app.get('/api/change-token')
async def change_token(token: str):
    return_dict = translator._change_token(token)
    return return_dict


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)

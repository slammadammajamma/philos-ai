from typing import Union

from fastapi import FastAPI, Query

from model import semantic_search, get_next, get_prev

app = FastAPI()

@app.get('/')
async def root():
    return {
        'endpoints': {
            'get': [
                'next/{corpus_id}/?count=', 
                'search/{query_string}/?count=',
            ]
        }
    }

@app.get('/search/{query_string}')
async def search(query_string : str, count: Union[int, None] = Query(default=10, ge=1, le=50)):
    return semantic_search(query_string, count)

@app.get('/next/{corpus_id}')
async def next(corpus_id: int, count: Union[int, None] = Query(default=1, ge=1, le=50)):
    return get_next(corpus_id, count)

@app.get('/prev/{corpus_id}')
async def prev(corpus_id: int, count: Union[int, None] = Query(default=1, ge=1, le=50)):
    return get_prev(corpus_id, count)
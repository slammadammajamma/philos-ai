from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pickle
import gzip
import torch
import io
import requests
import os

if not os.path.isfile('embeddings.gz'):
    r = requests.get('https://storage.googleapis.com/philos-ai-embeddings/embeddings.gz', timeout=2)
    open('embeddings.gz', 'wb').write(r.content)
    print('here')

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

df = None
corpus_embeddings=None

# gpu
if torch.cuda.is_available():
    with gzip.open('embeddings.gz', 'rb') as fIn:
        embeddings = pickle.load(fIn)
        df = embeddings['df']
        corpus_embeddings = embeddings['embeddings']
# cpu
else:
    with gzip.open('embeddings.gz', 'rb') as fIn:
        embeddings = CPU_Unpickler(fIn).load()
        df = embeddings['df']
        corpus_embeddings = embeddings['embeddings']
        print(df)

# # pretrained encoders from sentence transformers
# bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
# bi_encoder.max_seq_length = 256

# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# # search function
# def semantic_search(query, count):
# 	# initial question embedding and search with bi encoder
#   question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
	
#   hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=32 if count is None else count)
#   hits = hits[0]

# 	# reordering using cross encoder
#   cross_inp = [[query,  df['sentence_str'][hit['corpus_id']]] for hit in hits]
#   cross_scores = cross_encoder.predict(cross_inp)

#   for idx in range(len(cross_scores)):
#     hits[idx]['cross-score'] = cross_scores[idx]

#   hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)

# 	# formatting results
#   results = []

#   for hit in hits:
#     value = {
# 				'id': hit['corpus_id'],
#         'passage': df['sentence_str'][hit['corpus_id']],
#         'author': df['author'][hit['corpus_id']],
#         'title': df['title'][hit['corpus_id']],
#         'school': df['school'][hit['corpus_id']],
#         'score': str(hit['cross-score'])
#     }

#     results.append(value)

#   return results

def get_next(id, count):
	return df['sentence_str'][id+1:id+count+1]

def get_prev(id, count):
	return df['sentence_str'][id-count:id]
import json


EMBEDDING_SIZE = 2000
DATASET = "data/wiki.txt"
INDEX_FILE = "data/indexed_lines.txt"
SAVED_STATE = "saved_state.txt"
TOKENIZER_FILENAME = "data/bpe_tokenizer_500_50.json"

bpe_json = open(TOKENIZER_FILENAME).read()
bpe_json = json.loads(bpe_json)
VOCAB = list(bpe_json["model"]["vocab"])
TOKEN_TO_ID = dict((c, i) for i, c in enumerate(VOCAB))

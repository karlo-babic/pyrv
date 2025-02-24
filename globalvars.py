import json


EMBEDDING_SIZE = 2000
DATASET = "../../wiki2.txt"
INDEX_FILE = "../indexed_lines_all2.txt"
SAVED_STATE = "saved_state.txt"
TOKENIZER_FILENAME = "../bpe_tokenizer_500_50.json"

bpe_json = open(TOKENIZER_FILENAME).read()
bpe_json = json.loads(bpe_json)
VOCAB = list(bpe_json["model"]["vocab"])
TOKEN_TO_ID = dict((c, i) for i, c in enumerate(VOCAB))

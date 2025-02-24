from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="<unk>"))



from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()


from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=['<subwordlvl>', '<phraselvl>', '<word.beg>', '<word.end>', '<unk>', '<->'], vocab_size=500, limit_alphabet=50, show_progress=True)
tokenizer.train(files=["wiki.txt", "wiki.txt", "wiki.txt"], trainer=trainer)


tokenizer.save("bpe_tokenizer_500_50.json")

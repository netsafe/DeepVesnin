import gc
#gc.set_threshold(200,10,10)
gc.enable()
#import spacy
#spacy.prefer_gpu()
from deeppavlov.core.common.file import read_json
from deeppavlov import configs, train_evaluate_model_from_config
#train_evaluate_model_from_config(read_json("/ai/jupyter/.deeppavlov/configs/doc_retrieval/ru_ranker_tfidf_wiki_custom.json"), download=False)
train_evaluate_model_from_config(read_json("/ai/jupyter/.deeppavlov/configs/doc_retrieval/en_ranker_tfidf_wiki_custom.json"), download=False)


import pandas
from tabulate import tabulate
import cupy
import spacy

#from thinc.api import set_active_gpu
#set_active_gpu(1)


def pbool(x):
    return '+' if x else '-'


def entity_at(t):
    # print(t.i, t.idx, dir(t))
    entity = [e for e in t.doc.ents if e.start == t.i]
    if entity:
        return "{}: {}".format(t.ent_type_, entity[0].text)
    return ''


def print_tokens(nlp, doc):
    for s in doc.sents:
        print('Sentence: "{}"'.format(s))
        df = pandas.DataFrame(columns=['Shape', 'Vocab', 'POS', 'Text', 'Lemma', 'Entity', 'Dep', 'Head'],
                              data=[(t.shape_, pbool(t.orth_ in nlp.vocab), t.pos_,
                                     t.text, t.lemma_, entity_at(t), t.dep_, t.head) for t in s])
        print(tabulate(df, showindex=False, headers=df.columns))


sample_sentences = "Привет России и миру! Как твои дела? Сегодня неплохая погода. Сергей как твои дела?"
if __name__ == '__main__':
    for i in range(0,32):
        print("probing device {}...".format(i))
        try:
          device=cupy.cuda.Device(i)
          print("meminfo {}".format(device.mem_info))
        except:
          print("no device {}".format(i))

    spacy.prefer_gpu(1)
    nlp = spacy.load('ru2',disable=["parser","ner"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    print("Pipeline: {}".format(nlp.pipe_names))
    doc = nlp(sample_sentences)
    print("Sample sentences: {}".format(sample_sentences))
    print("\nResults for ru2 model: ")
    print_tokens(nlp, doc)
    print("lemmatizing")
    for token in doc:
        print(token, token.lemma, token.lemma_)
    exit()
    nlp = spacy.load('ru2', disable=['tagger', 'parser', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(sample_sentences)
    print("\n"+"~"*70)
    print('\nSwitched to lemmatizer and POS from pymorphy2')
    print("Results for empty model: ")
    print_tokens(nlp, doc)

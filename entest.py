import pandas
import spacy
from tabulate import tabulate


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
sample_sentences="Hi George! How are you doing? How are you today?"

if __name__ == '__main__':
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    print("Pipeline: {}".format(nlp.pipe_names))
    doc = nlp(sample_sentences)
    print("Sample sentences: {}".format(sample_sentences))
    print("\nResults for en model: ")
    print_tokens(nlp, doc)

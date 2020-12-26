import multiprocessing as mp
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

corpus = { ( { f_id: nltk.corpus.gutenberg.raw(f_id)}, constant)
          for f_id in nltk.corpus.gutenberg.fileids()}

def tokenize_and_pos_tag(pair, constant):
    f_id, doc = pair
    t=ToktokTokenizer()
    tokens=t.tokenize(doc)
    print(constant)
    return f_id, tokens


if __name__ == '__main__':
    nltk.download('gutenberg')
    nltk.download('averaged_perceptron_tagger')
    constant={'qq','ee'}
    # automatically uses mp.cpu_count() as number of workers
    # mp.cpu_count() is 4 -> use 4 jobs
    with mp.Pool() as pool:
        tokens = pool.map(tokenize_and_pos_tag, corpus.items())

    print(tokens)
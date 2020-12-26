# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
import itertools
import time
import traceback 

from logging import getLogger
from typing import List, Generator, Any, Optional, Union, Tuple

# from nltk.corpus import stopwords
# STOPWORDS = stopwords.words('russian')
import pymorphy2
from nltk.tokenize.toktok import ToktokTokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.tokenizers.utils import detokenize, ngramize, ngramize_step, ngramize_final

import multiprocessing as mp

import cupy

logger = getLogger(__name__)
lck=mp.Lock()
pidmap={}

def tokenize_parallel(pair):
    i, doc = pair
    t=ToktokTokenizer()
    tokens=t.tokenize(doc)
    del t
    return i,tokens

def get_free_gpu(lck,cudadev,cudamax,cudadevs):
    cdv=-1
    try:
        lck.acquire()
        #logger.info("[{}]:dev mask is {}".format(int(mp.current_process().pid),cudadevs))
        if(cudadev.value < cudamax.value):
            for i in range(0,cudamax.value):
                if cudadevs[i]['in_use'] < 1 and cdv < 0:
                    cudadevs[i]['in_use']=1
                    gpu_memory=cudadevs[i]['memory']
                    cdv=i
                    cudadev.value+=1
    except: 
        traceback.print_exc() 
    finally:
        lck.release()
    return cdv

def release_gpu(cdv,cudadev,cudadevs,lck):
    if lck.acquire():
        cudadevs[cdv]['in_use']=0
        cudadev.value-=1
        #logger.info("[{}]:dev mask is {} after release".format(int(mp.current_process().pid),cudadevs))
        lck.release()

def handle_lemmas(q,dataset,cdv):
    mypid=int(mp.current_process().pid)
    try:
        import spacy
        spacy.prefer_gpu(cdv)
        nlp = spacy.load(model)
        hlp=list(dataset.keys())
        lemmas=dict()
        for j, doc in enumerate(nlp.pipe(dataset.values())):
            lemmas_item = [t.lemma_ for t in doc]
            lemmas[hlp[j]] = lemmas_item
    except:
        pass
    q.put(lemmas)


def lemmatize_parallel(pair):
    #logger.info(pair)
    i, (tok2morph, model, cudadev, cudamax, cudadevs,lck,cuda_memory_map) = pair
    #logger.info(tok2morph[i])
    #logger.info("ndata is ".format(len(tok2morph[i])))
    rCount=0
    needReinit=False
    mypid=int(mp.current_process().pid)
    processed_size=0
    job_size=len(tok2morph[i].values())
    batch_memsize=0
    # we will iterate until all the documents are processed
    while job_size > processed_size:
        try:
            cdv=pidmap[mypid]
        except KeyError as error:
            needReinit=True

        if needReinit:
            cdv,gpu_memory = get_free_gpu(lck,cudadev,cudamax,cudadevs)
            pidmap[mypid]=cdv

        if(cdv >= 0):
            # OKay, we've acquired free GPU
            try:
                try:
                    max_documents_size=cuda_memory_map[gpu_memory]
                except KeyError:
                    max_documents_size=gpu_memory-600 # spaCy model loading takes roughly 600 megabytes after spacy_load()
                    cuda_memory_map[gpu_memory]=max_documents_size # and take a blind shot to initialize the value

                ctx = mp.get_context('spawn') # we need a brand new fresh instance to release GPU memory for sure in case of the problem
                q = ctx.Queue()
                p = ctx.Process(target=handle_lemmas, args=(q,tok2morph[i],cdv,))
                p.start()
                lemmas=q.get()
                p.join()
                #if( > len(lemmas.values()))
                
            except:
                logger.info("[ {} ]: lemmatizer worker exception".format(mypid))
                traceback.print_exc()

            release_gpu(cdv,cudadev,cudadevs,lck)
        else:
            logger.info("[{}]: unable to use any CUDA device".format(mypid))
    #logger.info("returning for i {}".format(i))
    return i, lemmas


def ngramize_parallel(pair):
    i, f = pair
    x=ngramize_step(f[0],f[1])
    return x

@register('ru_tokenizer')
class RussianTokenizer(Component):
    """Tokenize or lemmatize a list of documents for Russian language. Default models are
    :class:`ToktokTokenizer` tokenizer and :mod:`pymorphy2` lemmatizer.
    Return a list of tokens or lemmas for a whole document.
    If is called onto ``List[str]``, performs detokenizing procedure.

    Args:
        stopwords: a list of stopwords that should be ignored during tokenizing/lemmatizing
         and ngrams creation
        ngram_range: size of ngrams to create; only unigrams are returned by default
        lemmas: whether to perform lemmatizing or not
        lowercase: whether to perform lowercasing or not; is performed by default by :meth:`_tokenize`
         and :meth:`_lemmatize` methods
        alphas_only: whether to filter out non-alpha tokens; is performed by default by :meth:`_filter`
         method
        save_path: tokenizer lemma cache save path
        load_path: tokenizer lemma cache load path

    Attributes:
        stopwords: a list of stopwords that should be ignored during tokenizing/lemmatizing
         and ngrams creation
        tokenizer: an instance of :class:`ToktokTokenizer` tokenizer class
        lemmatizer: an instance of :class:`pymorphy2.MorphAnalyzer` lemmatizer class
        ngram_range: size of ngrams to create; only unigrams are returned by default
        lemmas: whether to perform lemmatizing or not
        lowercase: whether to perform lowercasing or not; is performed by default by :meth:`_tokenize`
         and :meth:`_lemmatize` methods
        alphas_only: whether to filter out non-alpha tokens; is performed by default by :meth:`_filter`
         method
         tok2morph: token-to-lemma cache

    """

    def __init__(self, stopwords: Optional[List[str]] = None, ngram_range: List[int] = None,
                 lemmas: bool = False, lowercase: Optional[bool] = None,
                 alphas_only: Optional[bool] = None, save_path: Optional[str] = None, load_path: Optional[str] = None, spacy_model: str = 'en_core_web_sm', **kwargs):

        if ngram_range is None:
            ngram_range = [1, 1]
        self.stopwords = stopwords or []
        #self.tokenizer = ToktokTokenizer()
        self.spacy_model=spacy_model
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.ngram_range = tuple(ngram_range)  # cast JSON array to tuple
        self.lemmas = lemmas
        self.lowercase = lowercase
        self.alphas_only = alphas_only
        self.manager = mp.Manager()
        #self.tok2morph = self.manager.list()
        #self.ndata = self.manager.dict()
        self.load_path = load_path
        self.save_path = save_path
        self.cudamap = {}
        if load_path :
            load(load_path)
        self.nglist=self.manager.dict()

    def __call__(self, batch: Union[List[str], List[List[str]]]) -> \
            Union[List[List[str]], List[str]]:
        """Tokenize or detokenize strings, depends on the type structure of passed arguments.

        Args:
            batch: a batch of documents to perform tokenizing/lemmatizing;
             or a batch of lists of tokens/lemmas to perform detokenizing

        Returns:
            a batch of lists of tokens/lemmas; or a batch of detokenized strings

        Raises:
            TypeError: If the first element of ``batch`` is neither ``List``, nor ``str``.

        """
        try:
            if isinstance(batch[0], str):
                if self.lemmas:
                    return list(self._lemmatize(batch))
                else:
                    return list(self._tokenize(batch))
            if isinstance(batch[0], list):
                return [detokenize(doc) for doc in batch]
        except:
            self.save(self.save_path)
            self.manager.shutdown()

        raise TypeError(
            "StreamSpacyTokenizer.__call__() is not implemented for `{}`".format(type(batch[0])))


    def _tokenize(self, data: List[str], ngram_range: Tuple[int, int] = (1, 1), lowercase: bool = True) \
            -> Generator[List[str], Any, None]:
        """Tokenize a list of documents.

       Args:
           data: a list of documents to tokenize
           ngram_range: size of ngrams to create; only unigrams are returned by default
           lowercase: whether to perform lowercasing or not; is performed by default by
           :meth:`_tokenize` and :meth:`_lemmatize` methods

       Yields:
           list of lists of ngramized tokens or list of detokenized strings

        Returns:
            None

       """
        # DEBUG
        size = len(data)
        _ngram_range = self.ngram_range or ngram_range

        if self.lowercase is None:
            _lowercase = lowercase
        else:
            _lowercase = self.lowercase

        self.self_lowercase=_lowercase

        dataset = { i: doc for i, doc in enumerate(data) }

        tok_pool=self.manager.Pool()
        processed_doc_list=tok_pool.map(tokenize_parallel,dataset.items())
        tok_pool.close()
        tok_pool.terminate()

        del dataset
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        dataset = {}

        logger.info("parallel done")
        for p in processed_doc_list:
            id, tokens = p
            if _lowercase:
                tokens = [t.lower() for t in tokens]
            filtered = self._filter(tokens)
            dataset[id] = (self._filter(tokens), _ngram_range)

        logger.info("filtered serially")

        if len(self.nglist) > 0:
            self.nglist.clear()

        ng_pool=self.manager.Pool()
        self.nglist=ng_pool.map(ngramize_parallel,dataset.items())
        ng_pool.close()
        ng_pool.terminate()

        del dataset
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        logger.info("pre-ngrammized")

        for i in self.nglist:
            processed_doc=ngramize_final(i)
            yield from processed_doc

    def _lemmatize(self, data: List[str], ngram_range: Tuple[int, int] = (1, 1)) -> \
            Generator[List[str], Any, None]:
        """Lemmatize a list of documents.

        Args:
            data: a list of documents to tokenize
            ngram_range: size of ngrams to create; only unigrams are returned by default

        Yields:
            list of lists of ngramized tokens or list of detokenized strings

        Returns:
            None

        """
        # DEBUG
        size = len(data)
        _ngram_range = self.ngram_range or ngram_range

        #tokenized_data = list(self._tokenize(data))
        #logger.info("starting lemmatizing {} docs".format(len(tokenized_data)))

        #logger.info("data zero is {} of type {}".format(str(data[0]), type(str(data[0]))))
        #data1=data.copy()
        #print(data1)
        
        
        cuda_devices=0
        cuda_memory={}

        for i in range(0,32):
           try:
              device=cupy.cuda.Device(i)
              logger.info("meminfo {}".format(device.mem_info))
              cuda_memory[i]=device.mem_info[1]
           except:
              #print("no cuda device #{}".format(i))
              cuda_devices=i
              break
        cudadev=self.manager.Value('i',0)
        cudamax=self.manager.Value('i',cuda_devices)
        lck = self.manager.Lock()
        cudadevs=self.manager.dict({})
        cuda_memory_map=self.manager.dict(self.cudamap)
        tok2morph = self.manager.list([self.manager.dict({}) for i in range(0,cuda_devices)])
        for i in range(0,cuda_devices):
            cudadevs[i]=self.manager.dict({})
            cudadevs[i]['in_use']=0
            cudadevs[i]['memory']=cuda_memory[i]
        del cuda_memory
        logger.info("ii {} len is ".format(cudadevs,len(tok2morph)))
        
        ncount=0
        for di,dk in enumerate(data):
            if ncount == cuda_devices:
                ncount=0
            tok2morph[ncount][di]=dk
            ncount+=1
        #logger.info("data is {}".format(self.tok2morph[0]))
        dataset = { i: (tok2morph,self.spacy_model,cudadev,cudamax,cudadevs,lck,cuda_memory_map) for i in range(0,cuda_devices) }
        logger.info("dataset size is {}".format(len(dataset)))
        lemma_pool=self.manager.Pool(processes=cuda_devices)
        tokens = lemma_pool.map(lemmatize_parallel, dataset.items())
        #logger.info(tokens)
        #exit()
        #self.tok2morph=dict(_tok2morph)
        del tok2morph
        lemma_pool.close()
        lemma_pool.terminate()

        #logger.info("parallel lemmas done, cache is {}".format(len(_tok2morph)))
        #del _tok2morph

        #logger.info("parallel lemmas done, cache {} is {} {}:{}".format(type(tokens[0]),len(tokens[0]),type(tokens[0][0]),type(tokens[0][1])))
        del dataset
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        dataset = {}
        tokens2=list()
        for lj in range(0,cuda_devices):
            #logger.info(type(tokens[lj][1]))
            for value in tokens[lj][1].values():
                tokens2.append(value)
        lsize=len(tokens2)
        tokens2.sort()
        #logger.info("size before sort {} after {} ".format(lsize,len(list(token for token,_ in itertools.groupby(tokens)))))

        #for i,p in enumerate(list(token for token,_ in itertools.groupby(tokens))):
        #logger.info(tokens2)
        for i,lemmas in enumerate(tokens2):
            #id1, lemmas = p
            dataset[i] = (self._filter(lemmas), _ngram_range)

        #logger.info("filtered serially")
        del tokens
        del tokens2

        if len(self.nglist) > 0:
            self.nglist.clear()

        ng_pool=self.manager.Pool()
        self.nglist=ng_pool.map(ngramize_parallel,dataset.items())
        ng_pool.close()
        ng_pool.terminate()

        del dataset
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        #logger.info("pre-ngrammized")

        for i in self.nglist:
            processed_doc=ngramize_final(i)
            yield from processed_doc

    def _filter(self, items: List[str], alphas_only: bool = True) -> List[str]:
        """Filter a list of tokens/lemmas.

        Args:
            items: a list of tokens/lemmas to filter
            alphas_only: whether to filter out non-alpha tokens

        Returns:
            a list of filtered tokens/lemmas

        """
        if self.alphas_only is None:
            _alphas_only = alphas_only
        else:
            _alphas_only = self.alphas_only

        if _alphas_only:
            filter_fn = lambda x: x.isalpha() and not x.isspace() and x not in self.stopwords
        else:
            filter_fn = lambda x: not x.isspace() and x not in self.stopwords

        return list(filter(filter_fn, items))

    def set_stopwords(self, stopwords: List[str]) -> None:
        """Redefine a list of stopwords.

       Args:
           stopwords: a list of stopwords

       Returns:
           None

       """
        self.stopwords = stopwords

    def save(self,save_path: str):
        opts = {'stopwords': self.stopwords,
                'ngram_range': self.ngram_range,
                'lowercase': self.lowercase,
                'alphas_only': self.alphas_only}

        data = {
            'opts': opts
        }
        if save_path is None:
            raise FileNotFoundError("{} path doesn't exist!".format(__name__))
        if not save_path.exists():
            raise FileNotFoundError("{} path doesn't exist!".format(__name__))
        with open(save_path, 'w') as json_file:
            json.dump(my_details, json_file)

    def load(self,load_path: str):
        if not load_path.exists():
            raise FileNotFoundError("{} path doesn't exist!".format(__name__))

        logger.info("Loading lemma cache from {}".format(self.load_path))
        with open(load_path) as json_file:
            data = json.load(json_file)

        #self.tok2morph=data["data"]



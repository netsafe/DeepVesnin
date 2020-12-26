BERT in DeepPavlov
==================
BERT (Bidirectional Encoder Representations from Transformers) is a Transformer pre-trained on masked language model
and next sentence prediction tasks. This approach showed state-of-the-art results on a wide range of NLP tasks in
English.

| BERT paper: https://arxiv.org/abs/1810.04805
| Google Research BERT repository: https://github.com/google-research/bert

There are several pre-trained BERT models released by Google Research, more details about these pre-trained models could be found here: https://github.com/google-research/bert#pre-trained-models

-  BERT-base, English, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/cased_L-12_H-768_A-12.zip>`__
-  BERT-base, English, uncased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/uncased_L-12_H-768_A-12.zip>`__
-  BERT-large, English, cased, 24-layer, 1024-hidden, 16-heads, 340M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip>`__
-  BERT-base, multilingual, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12.zip>`__, `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12_pt.tar.gz>`__
-  BERT-base, Chinese, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: download from `[google] <https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip>`__,
   `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/chinese_L-12_H-768_A-12.zip>`__, `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/chinese_L-12_H-768_A-12_pt.tar.gz>`__

We have trained BERT-base model for other languages and domains:

-  RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_v2.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz>`__
-  SlavicBERT, Slavic (bg, cs, pl, ru), cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12_v1.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt.tar.gz>`__
-  Conversational BERT, English, cased, 12-layer, 768-hidden, 12-heads, 110M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/conversational_cased_L-12_H-768_A-12_v1.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/conversational_cased_L-12_H-768_A-12_pt.tar.gz>`__
-  Conversational RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/ru_conversational_cased_L-12_H-768_A-12.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/ru_conversational_cased_L-12_H-768_A-12_pt.tar.gz>`__
-  Sentence Multilingual BERT, 101 languages, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_multi_cased_L-12_H-768_A-12.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_multi_cased_L-12_H-768_A-12_pt.tar.gz>`__
-  Sentence RuBERT, Russian, cased, 12-layer, 768-hidden, 12-heads, 180M parameters: `[deeppavlov] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_ru_cased_L-12_H-768_A-12.tar.gz>`__,
   `[deeppavlov_pytorch] <http://files.deeppavlov.ai/deeppavlov_data/bert/sentence_ru_cased_L-12_H-768_A-12_pt.tar.gz>`__

The ``deeppavlov_pytorch`` models are designed to be run with the `HuggingFace's Transformers <https://huggingface.co/transformers/>`__ library.

RuBERT was trained on the Russian part of Wikipedia and news data. We used this training data to build vocabulary of Russian subtokens and took
multilingual version of BERT-base as initialization for RuBERT [1]_.

SlavicBERT was trained on Russian News and four Wikipedias: Bulgarian, Czech, Polish, and Russian.
Subtoken vocabulary was built using this data. Multilingual BERT was used as an initialization for SlavicBERT.
The model is described in our ACL paper [2]_.

Conversational BERT was trained on the English part of Twitter, Reddit, DailyDialogues [4]_, OpenSubtitles [5]_, Debates [6]_, Blogs [7]_, Facebook News Comments.
We used this training data to build the vocabulary of English subtokens and took
English cased version of BERT-base as initialization for English Conversational BERT.

Conversational RuBERT was trained on OpenSubtitles [5]_, Dirty, Pikabu, and Social Media segment of Taiga corpus [8]_.
We assembled new vocabulary for Conversational RuBERT model on this data and initialized model with RuBERT.

Sentence Multilingual BERT is a representation-based sentence encoder for 101 languages of Multilingual BERT.
It is initialized with Multilingual BERT and then fine-tuned on english MultiNLI [9]_ and on dev set of multilingual XNLI [10]_.
Sentence representations are mean pooled token embeddings in the same manner as in Sentence-BERT [12]_.

Sentence RuBERT is a representation-based sentence encoder for Russian.
It is initialized with RuBERT and fine-tuned on SNLI [11]_ google-translated to russian and on russian part of XNLI dev set [10]_.
Sentence representations are mean pooled token embeddings in the same manner as in Sentence-BERT [12]_.

Here, in DeepPavlov, we made it easy to use pre-trained BERT for downstream tasks like classification, tagging, question answering and
ranking. We also provide pre-trained models and examples on how to use BERT with DeepPavlov.

BERT as Embedder
----------------

:class:`~deeppavlov.models.embedders.transformers_embedder.TransformersBertEmbedder` allows for using BERT
model outputs as token, subtoken and sentence level embeddings.

Additionaly the embeddings can be easily used in DeepPavlov. To get text level, token level and subtoken level representations,
you can use or modify a :config:`BERT embedder configuration <embedder/bert_embedder.json>`:

.. code:: python
    
    from deeppavlov.core.common.file import read_json
    from deeppavlov import build_model, configs
    
    bert_config = read_json(configs.embedder.bert_embedder)
    bert_config['metadata']['variables']['BERT_PATH'] = 'path/to/bert/directory'

    m = build_model(bert_config)

    texts = ['Hi, i want my embedding.', 'And mine too, please!']
    tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = m(texts)

Examples of using these embeddings in model training pipelines can be found in :config:`Sentiment Twitter <classifiers/sentiment_twitter_bert_emb.json>`
and :config:`NER Ontonotes <ner/ner_ontonotes_bert_emb.json>` configuration files.


BERT for Classification
-----------------------

:class:`~deeppavlov.models.bert.bert_classifier.BertClassifierModel` and
:class:`~deeppavlov.models.torch_bert.torch_bert_classifier.TorchBertClassifierModel`
provide easy to use solution for classification problem
using pre-trained BERT on TensorFlow and PyTorch correspondingly.
One can use several pre-trained English, multi-lingual and Russian BERT models that are
listed above.

Two main components of BERT classifier pipeline in DeepPavlov are
:class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertPreprocessor` on TensorFlow
(:class:`~deeppavlov.models.preprocessors.torch_bert_preprocessor.TorchBertPreprocessor` on PyTorch) and
:class:`~deeppavlov.models.bert.bert_classifier.BertClassifierModel` on TensorFlow
(:class:`~deeppavlov.models.torch_bert.torch_bert_classifier.TorchBertClassifierModel` on PyTorch).
Non-processed texts should be given to ``bert_preprocessor`` (or ``torch_bert_preprocessor``) for tokenization on subtokens,
encoding subtokens with their indices and creating tokens and segment masks.
In case of using one-hot encoded classes in the pipeline, set ``one_hot_labels`` to ``true``.

``bert_classifier`` and ``torch_bert_classifier`` have a dense layer of number of classes size upon pooled outputs of Transformer encoder,
it is followed by ``softmax`` activation (``sigmoid`` if ``multilabel`` parameter is set to ``true`` in config).


BERT for Named Entity Recognition (Sequence Tagging)
----------------------------------------------------

Pre-trained BERT model can be used for sequence tagging. Examples of BERT application to sequence tagging
can be found :doc:`here </features/models/ner>`. The modules used for tagging
are :class:`~deeppavlov.models.bert.bert_sequence_tagger.BertSequenceTagger` on TensorFlow and
:class:`~deeppavlov.models.torch_bert.torch_bert_sequence_tagger.TorchBertSequenceTagger` on PyTorch.
The tags are obtained by applying a dense layer to the representation of
the first subtoken of each word. There is also an optional CRF layer on the top for TensorFlow implementation.

Multilingual BERT model allows to perform zero-shot transfer across languages. To use our 19 tags NER for over a
hundred languages see :ref:`ner_multi_bert`.

BERT for Morphological Tagging
------------------------------

Since morphological tagging is also a sequence labeling task, it can be solved in a similar fashion.
The only difference is that we may use the last subtoken of each word in case word morphology
is mostly defined by its suffixes, not prefixes (that is the case for most Indo-European languages,
such as Russian, Spanish, German etc.). See :doc:`also </features/models/morphotagger>`.

BERT for Syntactic Parsing
--------------------------

You can use BERT for syntactic parsing also. As most modern parsers, we use the biaffine model
over the embedding layer, which is the output of BERT. The model outputs the index of syntactic
head and the dependency type for each word. See :doc:`the parser documentation </features/models/syntaxparser>`
for more information about model performance and algorithm.


BERT for Context Question Answering (SQuAD)
-------------------------------------------
Context Question Answering on `SQuAD <https://rajpurkar.github.io/SQuAD-explorer/>`__ dataset is a task
of looking for an answer on a question in a given context. This task could be formalized as predicting answer start
and end position in a given context. :class:`~deeppavlov.models.bert.bert_squad.BertSQuADModel` on TensorFlow and
:class:`~deeppavlov.models.torch_bert.torch_bert_squad.TorchBertSQuADModel` on PyTorch use two linear
transformations to predict probability that current subtoken is start/end position of an answer. For details check
:doc:`Context Question Answering documentation page </features/models/squad>`.

BERT for Ranking
----------------
There are two main approaches in text ranking. The first one is interaction-based which is relatively accurate but
works slow and the second one is representation-based which is less accurate but faster [3]_.
The interaction-based ranking based on BERT is represented in the DeepPavlov with two main components
:class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertRankerPreprocessor` on TensorFlow
(:class:`~deeppavlov.models.preprocessors.torch_bert_preprocessor.TorchBertRankerPreprocessor` on PyTorch)
and :class:`~deeppavlov.models.bert.bert_ranker.BertRankerModel` on TensorFlow
(:class:`~deeppavlov.models.torch_bert.torch_bert_ranker.TorchBertRankerModel` on PyTorch)
and the representation-based ranking with components
:class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertSepRankerPreprocessor`
and :class:`~deeppavlov.models.bert.bert_ranker.BertSepRankerModel` on TensorFlow.
Additional components
:class:`~deeppavlov.models.preprocessors.bert_preprocessor.BertSepRankerPredictorPreprocessor`
and :class:`~deeppavlov.models.bert.bert_ranker.BertSepRankerPredictor` (on TensorFlow) are for usage in the ``interact`` mode
where the task for ranking is to retrieve the best possible response from some provided response base with the help of
the trained model. Working examples with the trained models are given :doc:`here </features/models/neural_ranking>`.
Statistics are available :doc:`here </features/overview>`.

BERT for Extractive Summarization
---------------------------------
The BERT model was trained on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) tasks.
NSP head was trained to detect in ``[CLS] text_a [SEP] text_b [SEP]`` if text_b follows text_a in original document.
This NSP head can be used to stack sentences from a long document, based on a initial sentence. The first sentence in
a document can be used as initial one. :class:`~deeppavlov.models.bert.bert_as_summarizer.BertAsSummarizer` on TensorFlow
and :class:`~deeppavlov.models.torch_bert.torch_bert_as_summarizer.TorchBertAsSummarizer` on PyTorch rely on
pretrained BERT models and does not require training on summarization dataset. 
We have three configuration files:

- :config:`BertAsSummarizer <summarization/bert_as_summarizer.json>` in Russian takes first sentence in document as initialization.
- :config:`BertAsSummarizer with init <summarization/bert_as_summarizer_with_init.json>` in Russian uses provided initial sentence.
- :config:`TorchBertAsSummarizer <summarization/torch_bert_as_en_summarizer.json>` in English takes first sentence in document as initialization.

Using custom BERT in DeepPavlov
-------------------------------

The previous sections describe the BERT based models implemented in DeepPavlov.
To change the BERT model used for initialization in any downstream task mentioned above the following parameters of
the :doc:`config </intro/configuration>` file must be changed to match new BERT path:

* download URL in the ``metadata.download.url`` part of the config
* ``bert_config_file``, ``pretrained_bert`` in the BERT based Component. In case of PyTorch BERT, ``pretrained_bert`` can be assigned to
    string name of used pre-trained BERT (e.g. ``"bert-base-uncased"``) and then ``bert_config_file`` is set to ``None``.
* ``vocab_file`` in the ``bert_preprocessor`` (``torch_bert_preprocessor``). In case of PyTorch BERT, ``vocab_file`` can be assigned to
    string name of used pre-trained BERT (e.g. ``"bert-base-uncased"``).

.. [1] Kuratov, Y., Arkhipov, M. (2019). Adaptation of Deep Bidirectional Multilingual Transformers for Russian Language. arXiv preprint arXiv:1905.07213.
.. [2] Arkhipov M., Trofimova M., Kuratov Y., Sorokin A. (2019). `Tuning Multilingual Transformers for Language-Specific Named Entity Recognition <https://www.aclweb.org/anthology/W19-3712/>`__ . ACL anthology W19-3712.
.. [3] McDonald, R., Brokos, G. I., & Androutsopoulos, I. (2018). Deep relevance ranking using enhanced document-query interactions. arXiv preprint arXiv:1809.01682.
.. [4] Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP 2017.
.. [5] P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
.. [6] Justine Zhang, Ravi Kumar, Sujith Ravi, Cristian Danescu-Niculescu-Mizil. Proceedings of NAACL, 2016.
.. [7] J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006). Effects of Age and Gender on Blogging in Proceedings of 2006 AAAI Spring Symposium on Computational Approaches for Analyzing Weblogs.
.. [8] Shavrina T., Shapovalova O. (2017) TO THE METHODOLOGY OF CORPUS CONSTRUCTION FOR MACHINE LEARNING: «TAIGA» SYNTAX TREE CORPUS AND PARSER. in proc. of “CORPORA2017”, international conference , Saint-Petersbourg, 2017.
.. [9] Williams A., Nangia N. & Bowman S. (2017) A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. arXiv preprint arXiv:1704.05426
.. [10] Williams A., Bowman S. (2018) XNLI: Evaluating Cross-lingual Sentence Representations. arXiv preprint arXiv:1809.05053
.. [11] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. (2015) A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326
.. [12] N. Reimers, I. Gurevych (2019) Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv preprint arXiv:1908.10084

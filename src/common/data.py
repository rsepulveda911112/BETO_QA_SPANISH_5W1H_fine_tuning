import itertools as itt
import string
import numpy as np
import pandas as pd
import spacy
import random
import yaml
import logging

logger = logging.getLogger("autobrat.data")

from functools import lru_cache
from pathlib import Path
from src.common.utils import Collection, Sentence, Keyphrase

# Remember to split tokens that contain whitespaces (len(t.text.split()) > 1)
# when writting output (when reading it is not necessary).


MODELS = {}

dict_questions = {'What' : '¿Qué?',
                  'Where': '¿Dónde?',
                  'Why':   '¿Por qué?',
                  'How':  '¿Cómo?',
                  'Who':   '¿Quién?',
                  'When':  '¿Cuándo?',
                }


def load_training_q_a(collection: Collection):
    element_to_remove = set(['Figure', 'Title' , 'Conclusion', 'Body', 'Key-Expression', 'Subtitle', 'Quote', 'Lead', 'Orthotypography'])
    entity_types = set(keyphrase.label for sentence in collection.sentences for keyphrase in sentence.keyphrases)
    entity_types = entity_types.difference(element_to_remove)
    sentences = [s.text for s in collection.sentences]
    mapping = [['O'] * len(s) for s in sentences]
    sentences_ann = []
    sentences_dict = []
    sentences_dict_1 = []
    for entity_type in sorted(entity_types):
        for i, s in enumerate(collection.sentences):
            for j, p in enumerate(s.keyphrases):
                if p.label == entity_type:
                    if not s.text:
                        print('dd')
                    sentences_ann.append({'question': entity_type, 'text': p.original_text, 'context': s.text})
                    context = s.text
                    question = dict_questions[entity_type]
                    answer_star = context.find(p.original_text)
                    sentences_dict.append({'context': context, 'qas': [{'id': str(i)+ str(j), "is_impossible": False, 'question': question,
                                                                       'answers': [{'text': p.original_text , 'answer_start': answer_star}]}]})
                    sentences_dict_1.append({'answers': {'text': [p.original_text], 'answer_start': [answer_star]},
                                             'context': context, 'id': str(i) + str(j),
                                             'question': question})

    df = pd.DataFrame(sentences_ann)
    df_1 = pd.DataFrame(sentences_dict_1)

    return df, sentences_dict, df_1


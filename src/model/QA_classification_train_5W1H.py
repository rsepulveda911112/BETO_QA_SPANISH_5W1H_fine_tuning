from transformers import *
import os
from src.model.question_answering.question_answering_model import QuestionAnsweringModel
import random
from src.common.metrics import compute_f1_texts
import copy


dict_questions = {'¿Qué?': 'What',
                  '¿Dónde?': 'Where',
                  '¿Por qué?': 'Why',
                  '¿Cómo?': 'How',
                  '¿Quién?': 'Who',
                  '¿Cuándo?': 'When',
                 }


class QA_5W1H_Model():

    def __init__(self, model_dir="", use_cuda=False):
        self.get_model(model_dir, use_cuda)

    def get_model(self, model_dir="", use_cuda=False):
        manual_seed = random.randint(1, 5000)
        manual_seed = 4918
        if not model_dir:
            self.model = QuestionAnsweringModel(
                "auto", "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es", use_cuda=use_cuda,
                args={'num_train_epochs': 3,
                      'manual_seed': manual_seed,
                      'evaluate_during_training': True,
                      'wandb_project': '5W1H'})
        else:
            self.model = QuestionAnsweringModel(
                 "auto", os.getcwd() +  model_dir, use_cuda=use_cuda)

    def fine_tuning(self, train_data, eval_data, wandb_disabled=False):
        # os.environ["WANDB_DISABLED"] = str(wandb_disabled)
        self.model.train_model(train_data, eval_data=eval_data)
        result, _ = self.model.eval_model(eval_data)
        print(result)

    def predict(self, test_data):
        truth = copy.deepcopy(test_data)
        result, texts, all_predictions = self.model.predict(test_data)
        metric, _ = self.model.calculate_results(truth, all_predictions, F1=compute_f1_texts)
        print(metric)
        print('MA:', metric['correct'] / len(test_data))







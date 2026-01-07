import pandas as pd
from icecream import ic
from rouge_score import rouge_scorer, scoring
import numpy as np
import service.chat.chat as ct
import zeroShot.zeroShot as zs
import fewShot.fewShot as fs
from service.processData.processData import split_adr_content
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    return scores


def compute_bleu(predictions, references):
    scores = [sentence_bleu([ref.split()], pred.split()) for pred, ref in zip(predictions, references)]
    return scores


def compute_meteor(predictions, references):
    scores = [meteor_score([ref.split()], pred.split()) for pred, ref in zip(predictions, references)]
    return scores

def compute_bertscore(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en")
    return {"precision": P, "recall": R, "f1": F1}

class evaluation:
    def __init__(self):
        pass

    def print_results(self, predictions, references):
        rouge_scores = compute_rouge(predictions, references)
        ic(rouge_scores)
        bleu_scores = compute_bleu(predictions, references)
        ic(bleu_scores)
        meteor_scores = compute_meteor(predictions, references)
        ic(meteor_scores)
        bert_scores = compute_bertscore(predictions, references)
        ic(bert_scores)
        df = pd.DataFrame(list(zip(rouge_scores, bleu_scores, meteor_scores,)),
                          columns=['rouge_scores', 'bleu_scores', 'meteor_scores'])
        df.to_csv('D:\Code\Final\\final\\result\\qwen-plus\\' + "_scores" + '.csv', index=False)

    def store_output(self, model_name, context, decision, predicted_decision, experiment='0_shot'):
        df = pd.DataFrame(list(zip(context, decision, predicted_decision)),
                          columns=['context', 'decision', 'predicted_decision'])
        model_name = model_name.replace('/', '_')
        df.to_csv('D:\Code\Final\\final\\result\\qwen-plus\\' + model_name + '.csv', index=False)

    def start(self,input_file, model):
        df = pd.read_csv(input_file, encoding='latin1')
        chat = ct.chat("qwen-plus")
        zeroShot =  zs.zeroShot()
        contexts = []
        decisions = []
        predicted_decisions = []
        for i, row in df.head(5).iterrows():
            content = row['md_content']
            context, predicted_decision = split_adr_content(content)
            prompt = zeroShot.get_prompt(context)
            decision = chat.chat(prompt).choices[0].message.content
            contexts.append(context)
            decisions.append(decision)
            predicted_decisions.append(predicted_decision)
            ic(f"Context: {context}")
            ic(f"Predicted Decision: {predicted_decision}")
            ic(f"Actual Decision: {decision}")
            print("--------------------------------------------------")
        self.print_results(predicted_decisions, decisions)
        self.store_output(model, contexts, decisions, predicted_decisions, experiment='0_shot')

if __name__ == "__main__":
    evaluator = evaluation()
    evaluator.start(input_file="D:\Code\Final\\final\\filtered_final_data.csv", model="qwen-plus")


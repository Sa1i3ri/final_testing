import os
import pandas as pd
from icecream import ic
from rouge_score import rouge_scorer, scoring
import service.chat.chat as ct
import zeroShot.zeroShot as zs
import fewShot.fewShot as fs
import service.retriever.retriever as rag
from service.processData.processData import split_adr_content
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from service.PromptProvider import PromptProvider


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    for score in scores:
        rounded_score = {
            'precision': round(score['rouge1'].precision, 3),
            'recall': round(score['rouge1'].recall, 3),
            'fmeasure': round(score['rouge1'].fmeasure, 3)
        }
        score['rouge1'] = rounded_score
    return scores


def compute_bleu(predictions, references):
    scores = [round(sentence_bleu([ref.split()], pred.split()), 3) for pred, ref in zip(predictions, references)]
    return scores


def compute_meteor(predictions, references):
    scores = [round(meteor_score([ref.split()], pred.split()), 3) for pred, ref in zip(predictions, references)]
    return scores

def compute_bertscore(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en")
    return {
        "precision": [round(p.item(), 3) for p in P],
        "recall": [round(r.item(), 3) for r in R],
        "f1": [round(f.item(), 3) for f in F1]
    }

class evaluation:
    def __init__(self):
        pass

    def print_results(self,predictions, references,model_name='qwen-plus' , experience = "0-shot"):
        rouge_scores = compute_rouge(predictions, references)
        ic(rouge_scores)
        bleu_scores = compute_bleu(predictions, references)
        ic(bleu_scores)
        meteor_scores = compute_meteor(predictions, references)
        ic(meteor_scores)
        bert_scores = compute_bertscore(predictions, references)
        ic(bert_scores)
        data = []
        data.append({
            'rouge1_precision': rouge_scores[0]['rouge1']['precision'],
            'rouge1_recall': rouge_scores[0]['rouge1']['recall'],
            'rouge1_fmeasure': rouge_scores[0]['rouge1']['fmeasure'],
            'bleu_score': bleu_scores[0],
            'meteor_score': meteor_scores[0],
            'bert_precision': bert_scores['precision'][0],
            'bert_recall': bert_scores['recall'][0],
            'bert_f1': bert_scores['f1'][0]
        })
        df = pd.DataFrame(data)

        output_file = f'D:\\Code\\Final\\final\\result\\{model_name}\\{experience}\\scores.csv'
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    def store_output(self,context, decision, predicted_decision, model_name,  experience = "0-shot"):
        model_name = model_name.replace('/', '_')
        output_dir = f'D:\\Code\\Final\\final\\result\\{model_name}\\{experience}'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}\\result.csv'
        self.print_results(predicted_decision, decision,model_name, experience)
        for ctx, dec, pred_dec in zip(context, decision, predicted_decision):
            df = pd.DataFrame([[ctx, dec, pred_dec]], columns=['context', 'decision', 'predicted_decision'])
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

    def start(self,input_file, model, experience, promptProvider: PromptProvider):
        df = pd.read_csv(input_file, encoding='latin1')
        chat = ct.chat(model=model)
        for i, row in df.iloc[-2:].iterrows():
            contexts = []
            decisions = []
            predicted_decisions = []
            context = row['context']
            predicted_decision = row['decision']
            prompt = promptProvider.get_prompt(context)
            decision = chat.chat(prompt).choices[0].message.content
            contexts.append(context)
            decisions.append(decision)
            predicted_decisions.append(predicted_decision)
            ic(f"Context: {context}")
            ic(f"Predicted Decision: {predicted_decision}")
            ic(f"Actual Decision: {decision}")
            self.store_output([context], [decision], [predicted_decision], model, experience)
            print("--------------------------------------------------")

if __name__ == "__main__":
    evaluator = evaluation()
    model = "gpt-4"
    # evaluator.start(input_file="D:\Code\Final\\final\\filtered_final_data.csv", model=model, experience="0-shot", promptProvider=zs.zeroShot())
    # evaluator.start(input_file="D:\Code\Final\\final\\filtered_final_data.csv", model=model, experience="few-shot", promptProvider=fs.fewShot())
    evaluator.start(input_file="D:\Code\Final\\final\\filtered_final_data.csv", model=model, experience="rag", promptProvider=rag.retriever())


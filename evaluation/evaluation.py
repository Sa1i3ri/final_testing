from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score


def meteor(reference, hypothesis):
    score = meteor_score([reference], hypothesis)
    return score


def rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores


def bleu(reference, hypothesis):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    score = sentence_bleu([reference_tokens], hypothesis_tokens)
    return score

def bert_score(reference, hypothesis, lang="en"):
    P, R, F1 = score([hypothesis], [reference], lang=lang, verbose=False)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

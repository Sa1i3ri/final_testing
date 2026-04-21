import os
from dotenv import load_dotenv
from evaluation.evaluation import evaluation
import pandas as pd
import zeroShot.zeroShot as zs
import fewShot.fewShot as fs
import service.rag.retriever.retriever as rag
from icecream import ic


if __name__ == "__main__":
    load_dotenv()
    evaluator = evaluation()
    model = "gpt2"
    input_file = os.getenv("IMPORT_FILE")
    # evaluator.start(input_file=input_file, model=model, experience="0-shot",promptProvider=zs.zeroShot())
    # evaluator.start(input_file=input_file, model=model, experience="few-shot", promptProvider=fs.fewShot())
    # evaluator.start(input_file=input_file, model=model, experience="rag", promptProvider=rag.retriever())
    df = pd.read_csv("D:\Code\Final\\final\\result\gpt2\\fine-tune\\result.csv", encoding='latin1')
    for i, row in df.iterrows():
        # context = row['context']
        # ic(f"Context: {context}")
        decision = row['decision']
        ic(f"Actual Decision: {decision}")
        predicted_decision = row['predicted_decision']
        ic(f"Predicted Decision: {predicted_decision}")
        evaluator.print_results([predicted_decision], [decision], model_name=model, experience="fine-tune")

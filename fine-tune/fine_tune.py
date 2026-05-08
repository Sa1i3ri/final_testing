import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# =====================
# 1. 加载数据
# =====================
df = pd.read_csv("filtered_final_data.csv", usecols=["context", "decision"])

# 保存真实标签到 predicted_decision（按你的需求）
df["predicted_decision"] = df["decision"]

# 简单切分
train_df = df.iloc[:-200].copy()
val_df = df.iloc[-200:-100].copy()
test_df = df.iloc[-100:].copy()

# 清理空值，避免训练/推理异常
train_df = train_df.dropna(subset=["context", "decision"])
val_df = val_df.dropna(subset=["context", "decision"])
test_df = test_df.dropna(subset=["context", "predicted_decision"])

# =====================
# 2. 加载模型
# =====================
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# =====================
# 3. 预处理函数
# =====================
def preprocess_function_t5(examples):
    inputs = ["context: " + str(x) for x in examples["context"]]
    targets = [str(x) for x in examples["decision"]]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    # 把 padding token 替换成 -100，避免参与 loss
    label_ids = labels["input_ids"]
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in seq]
        for seq in label_ids
    ]

    model_inputs["labels"] = label_ids
    return model_inputs

# =====================
# 4. 数据集
# =====================
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(preprocess_function_t5, batched=True)
val_dataset = val_dataset.map(preprocess_function_t5, batched=True)

# 删掉 pandas 自动加的索引列
for col in ["__index_level_0__"]:
    if col in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns([col])
    if col in val_dataset.column_names:
        val_dataset = val_dataset.remove_columns([col])

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# =====================
# 5. 训练参数
# =====================
training_args = TrainingArguments(
    output_dir=f"./results_{model_name}",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=5e-5,
    weight_decay=0.01,

    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    save_total_limit=2,
    report_to="none",
)

# =====================
# 6. Trainer
# =====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(f"./fine_tuned_{model_name}")
tokenizer.save_pretrained(f"./fine_tuned_{model_name}")

# =====================
# 7. 绘制 Loss 曲线
# =====================
log_history = trainer.state.log_history

train_epochs = []
train_losses = []

eval_epochs = []
eval_losses = []

for log in log_history:
    if "loss" in log and "epoch" in log:
        train_epochs.append(log["epoch"])
        train_losses.append(log["loss"])
    if "eval_loss" in log and "epoch" in log:
        eval_epochs.append(log["epoch"])
        eval_losses.append(log["eval_loss"])

plt.figure(figsize=(8, 5))

if len(train_losses) > 0:
    plt.plot(train_epochs, train_losses, marker="o", label="Train Loss")
if len(eval_losses) > 0:
    plt.plot(eval_epochs, eval_losses, marker="s", label="Eval Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training & Validation Loss - {model_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"loss_curve_{model_name}.png")
plt.close()

print("训练日志：")
print("train_losses =", train_losses)
print("eval_losses =", eval_losses)

# =====================
# 8. 推理函数
# =====================
def generate_decision(context):
    if pd.isna(context):
        return "[NO_CONTEXT]"

    model.eval()

    inputs = tokenizer(
        "context: " + str(context),
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            min_length=1,
            num_beams=4,
            early_stopping=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return result if result else "[EMPTY]"

# =====================
# 9. 先抽样看几条生成结果
# =====================
print("\n抽样生成结果：")
for i in range(min(5, len(test_df))):
    sample_context = test_df.iloc[i]["context"]
    sample_true = test_df.iloc[i]["predicted_decision"]
    sample_pred = generate_decision(sample_context)

    print(f"\n样本 {i+1}")
    print("context:", str(sample_context)[:120])
    print("true   :", sample_true)
    print("pred   :", sample_pred)

# =====================
# 10. 生成结果 CSV
# =====================
test_df_copy = test_df.copy()
test_df_copy["decision"] = test_df_copy["context"].apply(generate_decision)

test_df_copy[["context", "decision", "predicted_decision"]].to_csv(
    f"result_{model_name}.csv",
    index=False
)

print(f"\n✅ 完成！结果已保存：result_{model_name}.csv")
print(f"📈 Loss 曲线：loss_curve_{model_name}.png")
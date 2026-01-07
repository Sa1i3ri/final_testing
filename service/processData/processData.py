import pandas as pd
import service.chat.chat as ct
import os

def filter_csv_stars():
    # 读取原始 CSV 文件
    input_file = "D:\Code\Final\\final\\data.csv"  # 替换为你的文件名
    output_file = "D:\Code\Final\\final\\filtered_100stars_data.csv"  # 输出文件名

    # 加载 CSV 文件到 DataFrame
    df = pd.read_csv(input_file, encoding='latin1')

    # 筛选出 stars > 1000 的条目
    filtered_df = df[df['stars'] > 100]

    # 将筛选后的数据存储到新的 CSV 文件
    filtered_df.to_csv(output_file, index=False)

    print(f"筛选后的数据已保存到 {output_file}")

# Extract the content for each section
def extract_section(md_content, start, section_name):
    next_section_start = md_content.find("## ", start + len(section_name))
    if next_section_start == -1:  # If no next section header is found, take the rest of the content
        return md_content[start + len(section_name):].strip()
    return md_content[start + len(section_name):next_section_start].strip()


def split_adr_content(md_content):
    # Find the start of the "Context" and "Decision" sections
    context_start = md_content.find("Context")
    decision_start = md_content.find("Decision")

    context = extract_section(md_content, context_start, "Context")
    decision = extract_section(md_content, decision_start, "Decision")

    return context, decision

def str_to_bool(s: str) -> bool:
    return s.lower() in ['true', '1', 'yes', 'y']

def filter_csv_content():
    # 读取原始 CSV 文件
    input_file = "D:\Code\Final\\final\\filtered_100stars_data.csv"  # 替换为你的文件名
    output_file = "D:\Code\Final\\final\\filtered_100stars_final_data.csv"  # 输出文件名
    df = pd.read_csv(input_file, encoding='latin1')
    chat = ct.chat("qwen-plus")
    num = 0
    # 遍历每一条记录
    for i, row in df.iterrows():
        content = row['md_content']
        context, decision = split_adr_content(content)
        custom_prompt = [
            {"role": "system",
             "content": "This is an architecture decision record. Determine if the ADR is of good quality,instead of just stating what tools were used. If the quality is good, return true; otherwise, return false."},
            {"role": "user",
             "content": f"## Context \n{context}\n\n## Decision \n{decision}"}
        ]
        answer = str_to_bool(chat.chat(custom_prompt).choices[0].message.content.strip().lower())
        print(f"Row {i} - Quality Check: {answer}")
        if answer:
            num += 1
            print(f"Accepted rows count: {num}")
            # 将满足条件的行追加到新的 CSV 文件
            row.to_frame().T.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)





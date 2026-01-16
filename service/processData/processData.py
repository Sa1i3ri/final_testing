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
    next_section_start = md_content.find("##", start + 3)
    if next_section_start == -1:  # If no next section header is found, take the rest of the content
        return md_content[start :].strip()
    return md_content[start:next_section_start].strip()


def split_adr_content(md_content):
    # Find the start of the "Context" and "Decision" sections
    context_start = md_content.find("## Context")
    decision_start = md_content.find("## Decision")

    context = extract_section(md_content, context_start, "Context")
    decision = extract_section(md_content, decision_start, "Decision")

    return context, decision

def split_data(file_path, output_file):
    df = pd.read_csv(file_path, encoding='latin1')
    for i, row in df.iterrows():
        content = row['md_content']
        context, decision = split_adr_content(content)
        new_df = pd.DataFrame([[content, context, decision]], columns=['md_content', 'context', 'decision'])
        new_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)



def str_to_bool(s: str) -> bool:
    return s.lower() in ['true', '1', 'yes', 'y']

def filter_csv_content(input_file, output_file):
    df = pd.read_csv(input_file, encoding='latin1')
    chat = ct.chat("qwen-plus")
    num = 0
    # 遍历每一条记录
    for i, row in df.iterrows():
        context = row['context']
        decision = row['decision']
        custom_prompt = [
            {"role": "system",
             "content": "这里有一份architecture decision record。专注于它的context和decision，如果其中有任何一个是空的，或者你认为这个adr的decision质量不高，请返回False，否则返回True。只需要返回True或False，不需要其他解释。"},
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

if __name__ == "__main__":
    input_file = "D:\Code\Final\\final\\split_data.csv"
    output_file = "D:\Code\Final\\final\\filtered_final_data.csv"
    # split_data("D:\Code\Final\\final\\filtered_data.csv", "D:\Code\Final\\final\\split_data.csv")
    filter_csv_content(input_file, output_file)







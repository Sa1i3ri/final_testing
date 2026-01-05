import pandas as pd

def filter_csv():
    # 读取原始 CSV 文件
    input_file = "/data.csv"  # 替换为你的文件名
    output_file = "/filtered_data.csv"  # 输出文件名

    # 加载 CSV 文件到 DataFrame
    df = pd.read_csv(input_file, encoding='latin1')

    # 筛选出 stars > 1000 的条目
    filtered_df = df[df['stars'] > 1000]

    # 将筛选后的数据存储到新的 CSV 文件
    filtered_df.to_csv(output_file, index=False)

    print(f"筛选后的数据已保存到 {output_file}")

filter_csv()
# import json
# import pandas as pd
# from datasets import Dataset

# # ローカルのJSONファイル名
# file_name = "../raw_data/test_data.json"

# # JSONファイルを読み込む
# with open(file_name, "r") as file:
#     data_list = json.load(file)

# # リストをpandas.DataFrame形式に変換
# df = pd.DataFrame(data_list)

# # pandas.DataFrameをdatasets.arrow_dataset.Dataset形式に変換
# dataset = Dataset.from_pandas(df)

# print(f"Number of prompts: {len(dataset)}")
# print(f"Column names are: {dataset.column_names}")
# print(f"head  : {dataset[1]}")
# print(f"type : {type(dataset)}")

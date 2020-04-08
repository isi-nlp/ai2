import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)

df = pd.read_csv('param_search.csv')

for d1 in (df['AI2_Data'].unique()):
    for d2 in (df['CN_Data'].unique()):
        sub_df = (df[(df['AI2_Data']==d1) & (df['CN_Data']==d2)]).sort_values(by=['accuracy'], ascending=False)
        top = sub_df[["AI2_Data", "CN_Data", "bs", "acb", "ws", "dr", "accuracy"]].head(10)
        top.reset_index(drop=True, inplace=True)
        with open("top_10.txt", "a") as out:
            print(top, file=out)
            print('', file=out)
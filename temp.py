import os
import pandas as pd

df = pd.read_csv("AILA_Dataset.csv")
org = df['original']
gen = df['generated']
print(len(org), ', ', len(gen))

os.chdir("dataset//original_dataset")
l = os.listdir()
gen = list(gen)
org = list(org)

cnt  = 0
for i in l:
    if i in org:
        cnt+=1

print("cnt : ", cnt)
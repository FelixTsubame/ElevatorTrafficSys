import pandas as pd 
import numpy as np

df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],
                   'population': [1864, 22000, 80000]},
                   index=['panda', 'polar', 'koala'])
#print(df)
for row in df.itertuples(name='animal'):
    print(row)
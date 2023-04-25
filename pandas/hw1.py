import pandas as pd
import math
df = pd.read_csv('dataframe_practice.csv', index_col = '0')

#df.rename(columns = {'a':'class', 'b':'student_id', 'c':'English', 'd':'Math'}, inplace = True)
df = df.rename(columns = {'a':'class', 'b':'student_id', 'c':'English', 'd':'Math'})

df['English'] = df['English'].fillna(round(df['English'].mean(), 1))
df['Math'] = df['Math'].fillna(round(df['Math'].mean(), 1))
df['student_id'] = df['student_id'].astype('int8')


math_adjusted = []
for i in df['Math']:
    math_adjusted.append(math.sqrt(i) * 10)
math_adjusted = pd.Series(math_adjusted)
df['Math_adjusted'] = math_adjusted


math_diff = []
df = df.sort_values('Math', ascending = False)
for i in df['Math']:
    math_diff.append(i)

for i in range(len(df['Math']) - 1):
    math_diff[i] = math_diff[i] - math_diff[i + 1]

math_diff = pd.Series(math_diff, index = df['Math'].index)
df['Math_diff'] = math_diff


output = []
for i in df['Math'].index:
    output.append("English: " + str(df['English'][i]) + " Math: " + str(df['Math'][i]))

output = pd.Series(output, index = df['Math'].index)
df['output'] = output


Math_get_60 = []
for i in range(10):
    mask1 = df['class'] == i
    mask2 = df['Math'] > 60
    Math_get_60.append(len(df[(mask1 & mask2)]))
data = {'class':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'Math_get_60':Math_get_60}
frame = pd.DataFrame(data, columns = ['class', 'Math_get_60'])
print(frame)

df.to_csv('dataframe_practice1.csv')
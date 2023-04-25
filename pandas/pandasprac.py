import pandas as pd 
import math


cvs_file = 'D:\\新作業用\\電梯排程\\pandas\\dataframe_practice.csv'
df = pd.read_csv(cvs_file)
df.set_index('0',inplace=True)

#1#
df = df.rename(columns={'a':'class','b':'student_id','c':'English','d':'Math'})
#print(df)
df.to_csv('D:\\新作業用\\電梯排程\\pandas\\df_step1.csv')
#print(df.head())


#2#
en_sum = df.sum()['English']
en_count = df.count()['English']
df['English'] = df['English'].fillna(en_sum/en_count)
df['English'] = df['English'].round(1)
en_sum = df.sum()['Math']
en_count = df.count()['Math']
df['Math'] = df['Math'].fillna(en_sum/en_count)
df['Math'] = df['Math'].round(1)
df['student_id'] = df['student_id'].astype('int8')
#print(df.dtypes)
df.to_csv('D:\\新作業用\\電梯排程\\pandas\\df_step2.csv')
#print(df.sum())

#3#
df['Math_adjusted'] = 0
for i in range(df.count()['Math']):
	df.loc[i,'Math_adjusted']=math.sqrt(df.loc[i,'Math'])*10
df['Math_adjusted'] = df['Math_adjusted'].round(1)
#print(df)
df.to_csv('D:\\新作業用\\電梯排程\\pandas\\df_step3.csv')

#4#
df_m_d = pd.DataFrame()
'''
temp = df.loc[df['class']==0].sort_values(by='Math',ascending=False)
print(temp)
df_m_d = df_m_d.append(temp,ignore_index=True)
print(df_m_d)
'''
for i in range(10):
	temp = df.loc[df['class']==i].sort_values(by='Math',ascending=False)
	temp = temp.reset_index()
	temp['math_diff'] = 0
	#print(temp)
	for j in range(temp.count()['Math']):
		if j == (temp.count()['Math']-1):
			temp.loc[j,'math_diff'] = 0.0
		else:
			temp.loc[j,'math_diff'] = temp.loc[j,'Math']-temp.loc[j+1,'Math']
	df_m_d = df_m_d.append(temp,ignore_index=True)
	#print(df_m_d)
df_m_d.set_index('0',inplace=True)
df = df_m_d
df.to_csv('D:\\新作業用\\電梯排程\\pandas\\df_step4.csv')

#5#
df_o = df.reset_index()
output = []
for i in range(df_o.count()['Math']):
	s = "English: {e}, Math: {m}".format(e = df_o.loc[i,'English'],m = df_o.loc[i,'Math'])
	output.append(s)
output = pd.Series(output)
df_o['output'] = output
#print(df_o)
df_o.set_index('0',inplace=True)
df = df_o
df.to_csv('D:\\新作業用\\電梯排程\\pandas\\df_step5.csv')

#6#
df_c = df.groupby(['class'])
class_ = []
Mg6 = []
for i in range(10):
	df_ci = df_c.get_group(i)
	class_.append(i)
	Mg6.append(df_ci.loc[df_ci["Math"]>60].count()['Math'])
Math_pass = pd.DataFrame(columns=["class","Math_gte_60"])
Mg6 = pd.Series(Mg6)
class_ = pd.Series(class_)
Math_pass['class'] = class_
Math_pass['Math_gte_60'] = Mg6
Math_pass.set_index('class',inplace=True)
Math_pass.to_csv('D:\\新作業用\\電梯排程\\pandas\\df_step6.csv')

print(df.dtypes)

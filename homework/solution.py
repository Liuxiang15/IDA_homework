import pandas as pd
import numpy as np
from pandas import Series,DataFrame

import matplotlib.pyplot as plt

def preProcess():
	data = pd.read_csv("diabetic_data.csv")

	#Weight、payer_code和medical_specialty：字段的缺失率较高，且对治疗效果的预测无明显作用，故直接删去
	data = data.drop(columns=["weight", "payer_code", "medical_specialty"])

	#race、diag_3：缺失值用特殊值“default_value”和 -1填补
	data.loc[data.race == "?", "race"] = 'default_value'
	data.loc[data.diag_3 == "?", "diag_3"] = -1

	#根据patient_nbr字段进行去重处理，只保留第一条记录（改进：根据较小的encounter_id）
	data.drop_duplicates(subset=['patient_nbr'],keep='first',inplace=True)


	#移除导致临终关怀或病人死亡的记录
	#将 discharge_disposition_id 字段为11，13，14，19，20，21的记录过滤掉
	data = data[~data["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
	print("此时数据的行数为："+ str(data.index.size))
	graph(data)

def graph(data):
	#以图表形式统计余下病人年龄、种族、性别、入院类型、入院初诊（diag_1）、出院去向等信息分布情况
	gender_data = data.loc[:, ['gender']]
	# gender_num = print(gender_data.apply(pd.value_counts))
	# print(gender_num)
	gender_num = gender_data.ix[:,"gender"]
	#print(gender_num)
	gender_result = pd.value_counts(gender_num.values, sort=False) #索引被重新定义的series数组
	print(gender_result)
	print(gender_result.index)
	print(gender_result.values)


	#绘制gender饼状图
	
	#label = ["Female", "Male", "Unknown"]#定义饼图的标签，标签是列表
	# 调节图形大小，宽，高
	plt.figure(figsize=(9, 6))
	labels = gender_result.index
	explode = [0.01, 0.01, 0.01]  # 设定各项距离圆心n个半径
	#fig = plt.figure(figsize=(6, 6))# 将画布设定为正方形，则绘制的饼图是正圆
	plt.pie(gender_result.values, labels=labels, autopct='%1.4f%%',explode=explode)#画饼图（数据，数据对应的标签，百分数保留两位小数点）
	plt.title("gender pie chart")
	plt.show()
	plt.savefig("GenderPieChart.jpg")


preProcess()



import pandas as pd
import numpy as np
from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import  matplotlib
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

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

	#绘制race饼状图
	race_num = data.ix[:,"race"]
	race_result = pd.value_counts(race_num.values, sort=False) #索引被重新定义的series数组
	print(race_result)
	plt.figure(figsize=(9, 6))  # 将画布设定为长方形
	labels = race_result.index
	plt.pie(race_result.values, labels=labels, autopct='%1.4f%%')  # 画饼图（数据，数据对应的标签，百分数保留4位小数点）
	plt.title("race pie chart")
	plt.show()

	#绘制gender饼状图
	gender_num = data.ix[:, "gender"]
	gender_result = pd.value_counts(gender_num.values, sort=False)  # 索引被重新定义的series数组
	print(gender_result)
	plt.figure(figsize=(9, 6))  # 将画布设定为长方形
	labels = gender_result.index
	plt.pie(gender_result.values, labels=labels, autopct='%1.4f%%')#画饼图（数据，数据对应的标签，百分数保留4位小数点）
	plt.title("gender pie chart")
	plt.show()

	#年龄分布柱状图
	age_num = data.ix[:,"age"]
	age_result = pd.value_counts(age_num.values, sort=True)  # 索引被重新定义的series数组
	print(age_result)
	plt.figure(figsize=(9, 6))  # 将画布设定为长方形
	num_list = age_result.values
	name_list = age_result.index
	plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
	plt.title("age bar chart")
	plt.show()

	#入院类型分布柱状图
	admission_type_num = data.ix[:,"admission_type_id"]
	admission_type_result = pd.value_counts(admission_type_num.values, sort=True)  # 索引被重新定义的series数组
	print(admission_type_result)
	plt.figure(figsize=(9, 6))  # 将画布设定为长方形
	num_list = admission_type_result.values
	name_list = admission_type_result.index
	plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
	plt.title("admission_type bar chart")
	plt.show()

	#入院初诊（diag_1）直方图
	diag_1_num = data.ix[:, "diag_1"]
	diag_1_result = pd.value_counts(diag_1_num.values, sort=True)  # 索引被重新定义的series数组
	print(diag_1_result)
	plt.figure(figsize=(9, 6))  # 将画布设定为长方形
	num_list = diag_1_result.values
	name_list = diag_1_result.index
	plt.hist(num_list, bins=50, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
	plt.title("diag_1 hist chart")
	# 显示横轴标签
	plt.xlabel("入院初诊区间")
	# 显示纵轴标签
	plt.ylabel("频数")
	plt.show()

	#出院去向 discharge_disposition_id
	discharge_disposition_num = data.ix[:, "discharge_disposition_id"]
	discharge_disposition_result = pd.value_counts(discharge_disposition_num.values, sort=True)  # 索引被重新定义的series数组
	print(discharge_disposition_result)
	num_list = discharge_disposition_result.values
	name_list = discharge_disposition_result.index
	plt.figure(figsize=(9, 6))  # 将画布设定为长方形
	plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
	plt.title("discharge_disposition_id bar chart")
	plt.xlabel("出院去向")
	plt.show()


#合理划分训练集和测试集，实现课程介绍的（但不限于）两种分类算法，用于预测患者的治疗效果，并比较分析各算法精度
def divide_into_train_test():
	pass

preProcess()



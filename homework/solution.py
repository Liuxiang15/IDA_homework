import pandas as pd
import numpy as np
from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import  matplotlib
# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

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
	# graph(data)
	
	X_train, labels, test_data, test_labels = divide_into_train_test(data)
	df = X_train.loc[:, ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
					  'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
					  'num_medications',
					  'number_outpatient', 'number_emergency','number_inpatient','readmitted']]
	prior_probability = df['readmitted'].value_counts() / df['readmitted'].size
	print("--------------类别的先验概率--------------")
	print(prior_probability)
	
	# 计算每个特征属性条件概率：
	race_condition_propability = pd.crosstab(df['race'], df['readmitted'], margins=True).apply(lambda x: x / x[-1], axis=1)
	gender_condition_propability = pd.crosstab(df['gender'], df['readmitted'], margins=True).apply(lambda x: x / x[-1], axis=1)
	age_condition_propability = pd.crosstab(df['age'], df['readmitted'], margins=True).apply(lambda x: x / x[-1], axis=1)
	admission_type_id_condition_propability = pd.crosstab(df['admission_type_id'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	discharge_disposition_id_condition_propability = pd.crosstab(df['discharge_disposition_id'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	admission_source_id_condition_propability = pd.crosstab(df['admission_source_id'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	time_in_hospital_condition_propability = pd.crosstab(df['time_in_hospital'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	num_lab_procedures_condition_propability = pd.crosstab(df['num_lab_procedures'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	num_procedures_condition_propability = pd.crosstab(df['num_procedures'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	num_medications_condition_propability = pd.crosstab(df['num_medications'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	number_outpatient_condition_propability = pd.crosstab(df['number_outpatient'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	number_emergency_condition_propability = pd.crosstab(df['number_emergency'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	number_inpatient_condition_propability = pd.crosstab(df['number_inpatient'], df['readmitted'], margins=True).apply(lambda x: x / x[-1],
																								   axis=1)
	
	print(race_condition_propability)
	print(gender_condition_propability)
	# 给出测试样本：
	test_data = pd.Series(['Caucasian', 'Male', '[80-90)', 1, 6, 7, 6, 60, 2, 21, 1, 0, 0],
						  index=['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
							'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
							'num_medications','number_outpatient', 'number_emergency','number_inpatient'])
	print("--------race_condition_propability.ix[:,test_data[0]] = -----")
	px = race_condition_propability.ix[test_data[0],:].mul(gender_condition_propability.ix[test_data[1],:]).mul(age_condition_propability.ix[test_data[2],:])[:-1]
	print(px)
	
	# 计算P(C | x)
	res = prior_probability.mul(px).idxmax()
	print("贝叶斯预测结果是"+res)
	#记住是用占比20%的testData中的每一条进行验证
    # naiveBayesian(data)
	

    #注释部分为测试KNN算法
	# trainDatas = normData(X_train)
	# testDatas = normData(test_data)
	# testKNN(trainDatas, testDatas, labels, test_labels, 10)

def naiveBayesian(data):
	# 1.分解各类先验样本数据中的特征
	# 2、计算各类数据中，各特征的条件概率
	# （比如：特征1出现的情况下，属于A类的概率p(A | 特征1)，属于B类的概率p(B | 特征1)，属于C类的概率p(C | 特征1)......）
	# 3、分解待分类数据中的特征（特征1、特征2、特征3、特征4......）
	# 4、计算各特征的各条件概率的乘积，如下所示：
	# 判断为A类的概率：p(A | 特征1) * p(A | 特征2) * p(A | 特征3) * p(A | 特征4).....
	# 5、结果中的最大值就是该样本所属的类别

    df = data.loc[:, ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
                        'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                        'num_medications',
                        'number_outpatient', 'number_emergency', 'number_inpatient','readmitted']]
    prior_probability = df['readmitted'].value_counts() / df['readmitted'].size
    print("--------------类别的先验概率--------------")
    print(prior_probability)

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


def normData(data):
	# 取部分属性作为标准来预测患者的治疗效果
	data = data.loc[:, ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
						'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
						'num_medications',
						'number_outpatient', 'number_emergency', 'number_inpatient',
						# 'diag_1', 'diag_2','diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult'
						]]

	#将[race]列元素量化：Caucasian 1， AfricanAmerican 2, default_value 3 ,Hispanic 4,Asian 5, other 6
	data.loc[data.race == "Caucasian", "race"] = 1
	data.loc[data.race == "AfricanAmerican", "race"] = 2
	data.loc[data.race == "default_value", "race"] = 3
	data.loc[data.race == "Hispanic", "race"] = 4
	data.loc[data.race == "Asian", "race"] = 5
	data.loc[data.race == "Other", "race"] = 6

	#将[gender]列元素量化
	data.loc[data.gender == "Female", "gender"] = 0
	data.loc[data.gender == "Male", "gender"] = 1
	data.loc[data.gender == "Unknown/Invalid", "gender"] = 2

	#[age]列元素量化
	data.loc[data.age == "[0-10)", "age"] = 5
	data.loc[data.age == "[10-20)", "age"] = 15
	data.loc[data.age == "[20-30)", "age"] = 25
	data.loc[data.age == "[30-40)", "age"] = 35
	data.loc[data.age == "[40-50)", "age"] = 45
	data.loc[data.age == "[50-60)", "age"] = 55
	data.loc[data.age == "[60-70)", "age"] = 65
	data.loc[data.age == "[70-80)", "age"] = 75
	data.loc[data.age == "[80-90)", "age"] = 85
	data.loc[data.age == "[90-100)", "age"] = 95

	print("---------------量化后的数据是：------------------")
	row = data.iloc[0].values  # 返回一个list
	print(row)
	# print(data.iloc[[0,1],[2]])

	#把每一个特征值除以该特征的范围，保证标准化后每一个特征值都在0~1之间。
	maxVals = data.max(axis=0)
	# print('-------------------------输出最大值--------------------')
	# print(maxVals)
	minVals = data.min(axis=0)
	# print('-------------------------输出最小值--------------------')
	# print(minVals)

	ranges = maxVals - minVals
	normedData = (data - minVals) / ranges
	# print('-------------------------标准化数据----------------------')
	# print(normedData)

	print('-------------------------前三条标准化数据是----------------------')
	# print(normedData.loc[[0,1,2],:])
	for i in range(2):
		row = normedData.iloc[i].values  # 返回一个list
		print(row)
	return normedData

#合理划分训练集和测试集，实现课程介绍的（但不限于）两种分类算法，用于预测患者的治疗效果，并比较分析各算法精度
def divide_into_train_test(data):
	# print("--------------进入划分函数-------")
	# print(data)

	y = data.ix[:, "readmitted"].values			#提取标签数组
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = 42)
	print("-----------------以下为X_test的五个示例----------------------------------")
	# for i in range(5):
	# 	row = X_test.iloc[i].values  # 返回一个list
	# 	print(row)
	# print(X_train.iloc[[0,5],:])
	# print("-----------------以下为X_train的五个示例标准化后的数据----------------------------------")
	# print(normData(X_train.iloc[[0,5],:]))
	# print("--------------以下为X_test------------")
	# print(X_test)
	# print('----------------提取某些属性后的新data是-------------')
	# print(data.loc[0,:])
	return X_train, y_train, X_test, y_test


def KNN(trainData, testData, labels, k):
	distSquare = (trainData - testData) ** 2  #计算距离的平方
	distSquareSum = distSquare.sum(axis=1)    # 求每一行的差值平方和
	distances = distSquareSum ** 0.5 		  # 开平方，得出每个样本到测试点的距离
	sortedIndex = distances.argsort()		  # 排序，得到排序后的下标
	# print("---------------------------排序后的下标是：------------------------")
	# print(sortedIndex)
	min_k = sortedIndex[:k]					  # 取最小的k个
	#分别定义三个变量统计三类标签No <30 >30对应的数量
	total = [0,0,0]
	for i in min_k:
		label = labels[i]
		if(label == "NO"):
			total[0] += 1
		elif label == "<30":
			total[1] += 1
		else:
			total[2] += 1
	max_label_num = total.index(max(total))
	if max_label_num == 0:
		print("NO")
		return "NO"
	elif max_label_num == 1:
		print("<30")
		return "<30"
	else:
		print(">30")
		return ">30"

def testKNN(trainDatas, testDatas, labels, test_labels, k):
	correct_rate = 0
	test_datas_list = testDatas.values.tolist()
	test_total_num = len(test_datas_list)		#测试数据个数
	correct_total_num = 0  # 测试成功个数
	for index, item in enumerate(test_datas_list):
		# 用来测试KNN算法的数据
		single_testdata = np.array(item)
		prediction_label = KNN(trainDatas.values, single_testdata, labels, k)
		if(prediction_label == test_labels[index]):
			correct_total_num += 1
			print("预测成功")
		else:
			print("预测失败")
	correct_rate = correct_total_num / test_total_num
	print(str(test_total_num)+"条测试数据中成功"+str(correct_total_num)+",成功占比为"+str(correct_rate))



preProcess()



import pandas as pd
import numpy as np
from pandas import Series,DataFrame

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

preProcess()



import pandas as pd

data = [['打喷嚏', '护士', '感冒'],
        ['打喷嚏', '农夫', '过敏'],
        ['头痛', '建筑工人', '脑震荡'],
        ['头痛', '建筑工人', '感冒'],
        ['打喷嚏', '教师', '感冒'],
        ['头痛', '教师', '脑震荡']]

df = pd.DataFrame(data, columns=['症状', '职业', '疾病'])

# 计算类别的先验概率
pr = df['疾病'].value_counts() / df['疾病'].size

# 计算每个特征属性条件概率：
pzz = pd.crosstab(df['疾病'], df['症状'], margins=True).apply(lambda x: x / x[-1], axis=1)
pzy = pd.crosstab(df['疾病'], df['职业'], margins=True).apply(lambda x: x / x[-1], axis=1)

dict = {}
dict["pzz"] = pzz
dict["pzy"] = pzy
print(dict)

# # 给出测试样本：
# x = pd.Series(['打喷嚏', '建筑工人'], index=['症状', '职业'])
# print(pzz)
# print("--------------------------------")
# print(pzz.loc[:, x[0]])
# print("--------x[0]------------------------")
# print(x[0])
#
# px = pzz.loc[:, x[0]].mul(pzy.ix[:, x[1]])[:-1]
#
# # 计算P(C | x)
# res = pr.mul(px).idxmax()
# print(res)
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd

# Change Excel file name and usecols to change compared factors

df = pd.read_excel('../Excel Files/Bank Data.xlsx', usecols=['IT/Cost', 'Growth in Rev'], sheet_name=0)
df.dropna(inplace = True)
data_sheet=df.to_numpy()

it_spend = data_sheet[:, 0].reshape(-1, 1)
dep_factor=data_sheet[:, 1]

lm = linear_model.LinearRegression()
lm.fit(it_spend, dep_factor)
y_pred = lm.predict(it_spend)

print("r2_score", r2_score(dep_factor, y_pred))

plt.scatter(it_spend,dep_factor, color ='blue')
plt.plot(it_spend, y_pred, color='red', linewidth=3)
plt.xlabel("IT Expenditure as a percentage of Total Costs")
plt.ylabel("Revenue growth two years out")

plt.show()

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd

# Change Excel file name and usecols to change compared factors

df = pd.read_excel('../Excel Files/Bank Data.xlsx', usecols=['Growth in IT', 'Future Profit'], sheet_name=0)
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
plt.xlabel("Percentage increase in IT expenditure over a period of two years")
plt.ylabel("Profit per Unit Cost in two years")

plt.show()

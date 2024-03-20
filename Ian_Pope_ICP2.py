#Ian Pope 700717419

import numpy as np
import pandas as pa
import pandas as pd

#Part A
#a)
vec = np.random.randint(1,21,15)
vec = vec.reshape((3,5),)
print(vec.shape)
print(vec)
print()

for i in range(3):
    j = np.argmax(vec[i])
    vec[i][j] = 0
print(vec)
print()

vec1 = np.random.randint(1,21,(4,3),dtype='int16')
print(vec1)
print(f"Shape: {vec1.shape}, Type: {type(vec1)}, Data Types: {vec1.dtype}\n")

#b)
ar1 = np.array([[3,-2],[1,0]])
print(ar1)
print(np.linalg.eig(ar1))
print()

#c)
ar2 = np.array([[0,1,2],[3,4,5]])
print(ar2)
print("The sum of the diagonal elements is", np.trace(ar2))
print()

#d)
ar3 = np.array([1,2,3,4,5,6])
print(ar3)
ar3 = np.reshape(ar3,(3,2))
print(ar3)
ar3 = np.reshape(ar3,(2,3))
print(ar3)



#Part B
print("\nPart B Pandas\n")

#1)
data = pa.read_csv("data.csv")
print(data)

#2)
print(f"\n{data.dtypes}\n")

#3)
pa.set_option('display.max_rows', 500)
print(data.isnull())
print()
#3a)
data = data.fillna(data.mean())
print(data.isnull())
print()

#4)
pa.set_option('display.max_rows', 10)
print(data[["Duration", "Calories"]].describe())

#5)
print(data[(data["Calories"] > 500) & (data["Calories"] < 1000)])
print()

#6)
print(data[(data["Calories"] > 500) & (data["Pulse"] < 100)])
print()

#7)
print("df_modified")
df_modified = pd.DataFrame([data.Duration, data.Pulse, data.Calories]).transpose()
print(df_modified)
print()

#8)
print("old_df")
data = data.drop("Maxpulse",axis=1)
print(data)
print()

#9)
data = data.astype({'Calories': 'int32'})
print(data)

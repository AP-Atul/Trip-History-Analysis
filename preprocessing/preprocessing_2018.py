import pandas as pd
from sklearn.preprocessing import LabelEncoder

# reading the csv file
dataFrame = pd.read_csv("./dataset/2018_01_02_capital_share.csv")
names = ['Duration', 'Start Station', 'End Station', 'Class']
data = dataFrame.values

# encode the last column to 1 and 0
le = LabelEncoder()
dataFrame['Member type'] = le.fit_transform(dataFrame['Member type'].values)
dataFrame.dropna(inplace=True)

# select only 4 cols
newDataFrame = pd.DataFrame({names[0]: dataFrame['Duration'], names[1]: dataFrame['Start station number'],
                             names[2]: dataFrame['End station number'], names[3]: dataFrame['Member type']})

# save the final processed file
newDataFrame.to_csv("2018_01_02_preprocessed.csv",
                    index=False,
                    header=names)
print("File processed")

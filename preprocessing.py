from datetime import datetime

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

dataFile = read_csv("./dataset/202007-capitalbikeshare-tripdata.csv")
names = dataFile.columns
data = dataFile.values

# timeData = dataFile[['started_at', 'ended_at']]
# dataFile['started_at'] = pd.to_datetime(dataFile['started_at']).total_seconds()
# dataFile['ended_at'] = pd.to_datetime(dataFile['ended_at']).total_seconds()


# print(dataFile.ended_at - dataFile.started_at)

duration = []
for i in tqdm(range(len(data))):
    duration.append(str((datetime.strptime(data[i][3], "%Y-%m-%d %H:%M:%S") -
                         datetime.strptime(data[i][2], "%Y-%m-%d %H:%M:%S")).total_seconds()))

names = ['duration', 'start_id', 'end_id', 'class']

le = LabelEncoder()
classes = le.fit_transform(dataFile['member_casual'].values.flatten())

df = pd.DataFrame({"duration": np.array(duration), "start_id": dataFile['start_station_id'],
                   "end_id": dataFile['end_station_id'], "class": classes})
# df.fillna(value=0, inplace=True)
df.dropna(inplace=True)
df.to_csv("duration_cal_preprocessed.csv", index=False, header=names)

# dataFile['started_at'] = pd.DataFrame(data=data[:][2].flatten())
# usefulCol = ['start_station_id', 'end_station_id', 'started_at', 'member_casual']
# dataFile.loc[:, usefulCol].to_csv('distance_added.csv')
# np.savetxt("distance_added.csv", X, delimiter=",")

# dataFile['Duration'] = dataFile['ended_at'] - dataFile['started_at']
# print(dataFile['Duration'])

import matplotlib.pyplot as plt
from pandas import read_csv

dataFrame = read_csv("processed_dataset/2018_01_02_preprocessed.csv")
dataFrame.hist()
plt.show()

dataFrame.plot.scatter(x='Start Station',
                       y='Duration',
                       c='Blue')

plt.title('Start Station and Duration')
plt.show()

dataFrame.plot.scatter(x='End Station',
                       y='Duration',
                       c='Red')

plt.title('End Station and Duration')
plt.show()

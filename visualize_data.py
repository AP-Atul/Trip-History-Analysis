from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

df = pd.read_csv("./dataset/2018_01_02_capital_share.csv")
df = df.iloc[0:100, [0, 1, 2, 3, 4, 5]]

"""
Class to visualize the data set of the csv file
Usage create an object pass the data set object

`dataFrame = read_csv("processed_dataset/2018_01_02_preprocessed.csv")`
"""


class Visualize:
    def __init__(self, dataFrame):
        """
        class to visualize the data with plots

        Parameters
        ----------
        dataFrame : pd.dataFrame
            pandas data frame object from reading the csv file
        """
        self.dataFrame = dataFrame

    def visualize(self):
        """
        Plot the data convert dates to datetime object to print occurrence
        """

        df["Start date"] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in df["Start date"]]
        df["End date"] = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in df["End date"]]
        # y = df['Start station number']
        y = df['Start station number']
        x = matplotlib.dates.date2num(df['Start date'])

        matplotlib.pyplot.plot_date(x, y, c='r', label='Start Journey')

        y = df['End station number']
        x = matplotlib.dates.date2num(df['End date'])

        matplotlib.pyplot.plot_date(x, y, c='b', label='End Journey')

        plt.legend(loc="upper left")
        plt.show()

        self.dataFrame.hist()
        plt.show()

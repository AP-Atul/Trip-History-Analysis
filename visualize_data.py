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
# dataFrame = read_csv("processed_dataset/2018_01_02_preprocessed.csv")
"""


class Visualize:
    def __init__(self, dataFrame):
        """
        pass the pandas data frame read from csv file
        :param dataFrame: dataset
        """
        self.dataFrame = dataFrame

    def visualize(self):
        """
        plot the data set initialized
        :return: None
        """

        # self.dataFrame.plot.scatter(x='Start Station',
        #                             y='Duration',
        #                             c='Blue')
        #
        # plt.title('Start Station and Duration')
        # plt.show()
        #
        # self.dataFrame.plot.scatter(x='End Station',
        #                             y='Duration',
        #                             c='Red')
        #
        # plt.title('End Station and Duration')
        # plt.show()

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

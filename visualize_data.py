import matplotlib.pyplot as plt

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
        self.dataFrame.hist()
        plt.show()

        self.dataFrame.plot.scatter(x='Start Station',
                                    y='Duration',
                                    c='Blue')

        plt.title('Start Station and Duration')
        plt.show()

        self.dataFrame.plot.scatter(x='End Station',
                                    y='Duration',
                                    c='Red')

        plt.title('End Station and Duration')
        plt.show()

# Trip-History-Analysis
Data Analysis on 'Captial Bikeshare : A bike sharing service in the United States'

## Instructions to run

1. Install all libraries using(requirements file) : pip3 install -r requirements.txt
2. Run the ui.py file

## Screen

![screen](https://github.com/AP-Atul/Trip-History-Analysis/blob/master/charts/window.png)

## Few Notes
The data includes:

* Duration – Duration of trip
* Start Date – Includes start date and time
* End Date – Includes end date and time
* Start Station – Includes starting station name and number
* End Station – Includes ending station name and number
* Bike Number – Includes ID number of bike used for the trip
* Member Type – Indicates whether user was a "registered" member (Annual Member, 30-Day Member or Day Key Member) or a "casual" rider (Single Trip, 24-Hour Pass, 3-Day Pass or 5-Day Pass)


This data has been processed to remove trips that are taken by staff as they service and inspect the system, trips that are taken to/from any of our “test” stations at our warehouses and any trips lasting less than 60 seconds (potentially false starts or users trying to re-dock a bike to ensure it's secure).

NOTE: The 3-Day Membership replaced the 5-Day Membership in Fall 2011


## Classification Algo
Classes 
1. Registered (member)
2. Casual (casual)

 * Logistic Regression
 * Naive Bayes
 * KNN

## Preprocessing
* Selecting duration, station (start and end ids), class (member, casual)
(These are the saved models accuracies)
1. KNN accuracy: 
    * Custom : 95.15
    * SKLearn : 91.30

2. NB accuracy:
    * Custom : 89.90
    * SKLearn : 91.45

3. LR accuracy:
    * Custom : 88.71
    * SKLearn : 90.17

## Directory details
1. processed_dataset/ : processed csv file
2. charts : plots to visualize data
3. lib : custom implementations of all the algos
4. model : saved pre-trained model 

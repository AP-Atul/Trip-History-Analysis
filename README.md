# Trip-History-Analysis
Data Analysis on 'Captial Bikeshare : A bike sharing service in the United States'

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

# Project II
In this paper, we describe the entire process we used to predict one week of sales. The dataset is real and was provided by a Product Owner (PO), along with some initial information about its contents. Throughout the process, we had several conversations with the PO to better understand what some of the fields represent.

The process described here includes a data analysis phase to understand the data and its relationships among all tables. Following this, we present our findings. Subsequently, we detail the ETL process used to handle the data, including populating and loading missing values, data casting, and other necessary steps. Next, we apply models such as ARIMA and Linear Models (LM) to make predictions. Finally, we explain the models and conclude with our findings.

For all phases, we opted to use Python for the various data treatments and model applications, including ARIMA and others. All code can be found in a GitHub repository located here https://github.com/yurimfreitas/ProjectII

Dataset
As mentioned before, this dataset was provided by our Product Owner, and the main goal was to predict one week of sales. We received a few CSV files:

Sales: This file contains 8,886,058 records of sales, including information about price, product, store, and other details about promotions and discounts.
Product: This file contains information about the products, with a total of 699 records.
Cities: This file contains information about the stores and their locations, with a total of 63 entries.
Forecast Revenue: This file contains 1,943 records, representing the forecasted revenue for one month.
1. Data Understanding
Data understanding will help us to know all about our data:

· Understanding our data

· Exploring structure of data

· Recognizing relationship between variables

Briefly, Data Profiling tell us almost everything about data.

Here are some point we want to investigate in our data

1.1. Understanding our data
In the sales information, we can check the following fields:

![alt text](image.png)
Figure 1 — Sales information
By looking at these data, it is clear that we will need to treat many columns in the next phase.
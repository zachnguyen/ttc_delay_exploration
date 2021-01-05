# Exploration & Modelling of TTC Bus Delays
Everyday, more than a million people in Toronto and the GTA make use of the TTC public transit system to get around to where they need to be. However, it is not always easy to be on time, as TTC buses may experience delays which are not only inconvenient but at times detrimental to Torontonian's work and personal lives.

The advent of predictive technology, particularly Machine Learning, has contributed to making public transit system more effective and enjoyable for riders. As someone who prefers public over private for transportation anyday, I feel motivated to learn more about TTC and its delays and perhaps try to provide some solutions with Machine Learning.

![TTC Bus](https://cdn.mobilesyrup.com/wp-content/uploads/2018/03/ttc-bus.jpg)

This repository house my project to explore the TTC Bus Delay dataset and find an ML Use Case for such data. 

## Data
The data used by this project are [TTC Delay datasets](https://open.toronto.ca/dataset/ttc-bus-delay-data/) from the City of Toronto Open Data Portal.
Future incorporation of data such as Bus Stop, GPS coordinates, Weather Data is being explored.

## Setup
The 'setup_script.ipynb' script download the data with an API call and set up the structure of the directory for the project. If you're planning to use it. Make sure to set your own path for your own repository. The project was completed on Google Colab so some of the code may need to be refactored to work on Jupyter Notebook.

## Preprocessing
The 'preprocessing.ipynb' script compiles the data into one dataframe and pre-process the raw data to an intermediate form ready for Exploration.
![dataframe](https://github.com/zachnguyen/ttc_delay_exploration/blob/main/images/dataframe.PNG)
## EDA
The 'eda.ipynb' script performs rigorous exploration of the data to develop intuition about it and formulate a good use case for Machine Learning. The use case is to predict whether a delay will be severe enough and warrant the search for an alternative route (other public transport, uber, friend-calling). Any TTC rider will have experienced stressful contemplation of whether to abandon a route in the face of tardiness, why not have an app that make this decision accurately for you?
![Features](https://github.com/zachnguyen/ttc_delay_exploration/blob/main/images/eda.PNG)
## ML Modelling
The 'modelling.ipynb' script execute various Machine Learning algorithms to assess the plausibility of the supervised classification task. It also outline future directions for the project.
![ROC](https://github.com/zachnguyen/ttc_delay_exploration/blob/main/images/roc.PNG)

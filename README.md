# Disaster_Response_Pipelines
Udacity project for Data Engineering Lesson


## About this Repository

This is a Udacity project in Data-Science Nano-Degree. Tackling skills I learned in Data Engineering, Software Engineering and Machine Learning
The fronend could be seen in this link: https://disaster-response-zeanrovin.herokuapp.com. In order to access classification feature, the app needs to be installed locally[Currently working on how to deploy both charts and classification feature online].

# Instructions for installation

Run the following commands in the root directory of each file to set up your database and model

  * To run ETL pipeline that cleans data and stores in db file: 
  
               python data/process_data.py
    
  * To run ML pipeline that trains classifier and saves it into a pickle file: 
  
               python models/train_classifier
               
  * Run the following command in the app's directory to run your web app: 
  
               python app/run.py
             go to http://0.0.0.0:3001/
             
   * Go to http://0.0.0.0:3001/
	  

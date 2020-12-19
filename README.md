# Disaster_Response_Pipelines
Udacity project for Data Engineering Lesson


## About this Repository

This is a Udacity project in Data-Science Nano-Degree. Tackling skills I learned in Data Engineering, Software Engineering and Machine Learning
The fronend could be seen in this link: https://disaster-response-zeanrovin.herokuapp.com. 
* In order to access classification feature, the app needs to be installed locally[Currently working on how to deploy both charts and classification feature online].

This web app will help the organzation classifying the emergency messages on what category they fall in. As they will receive thousands of messages, this will improve where they need to put most attention to

# File Structure
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py #merge two cv, clean and outputs a db file
|- messages_categories.db # database to save clean data to
models
|- train_classifier.py #train the data and outputs a pkl file
|- classifier.pkl # saved model
web_app #this folder is for heroku deployment
| - disasterapp
| | -app
| | - template
| | | - master.html # main page of web app
| | | - go.html # classification result page of web app
|- run.py # Flask file that runs app
| | -data
| | | - disaster_categories.csv # data to process
| | | - disaster_messages.csv # data to process
| | | - process_data.py #merge two cv, clean and outputs a db file
| | | - messages_categories.db # database to save clean data to
| | -models
| | | - train_classifier.py #train the data and outputs a pkl file
| | | - classifier.pkl # saved model
| | - Procfile #Procfile needed for heroki
| | - requirements.txt #Installation needed to deploy online
etl_pipeline.py #used as preparation for data/process_data.py (personal notes)
ml_pipeline.py #used as preparation for models/train_classifier.py (personal notes)
README.md

# Instructions for installation

Run the following commands in the root directory of each file to set up your database and model

  * Clone this repository
  
               git:clone -a https://github.com/zeanrovin/Disaster_pipeline_project
	       
  * Change directory
  
               cd Disaster_pipeline_project

  * To run ETL pipeline that cleans data and stores in db file: 
  
               python data/process_data.py
    
  * To run ML pipeline that trains classifier and saves it into a pickle file: 
  
               python models/train_classifier.py
               
  * Run the following command in the app's directory to run your web app: 
  
               python app/run.py
             go to http://0.0.0.0:3001/
             
   * Go to http://0.0.0.0:3001/
	  
etl_pipeline.py and ml_pipeline.py were used as preparation for ETL and training program.

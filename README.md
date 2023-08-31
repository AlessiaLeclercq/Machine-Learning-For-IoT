# Machine-Learning-For-IoT
Machine Learning for IoT course at Politecnico di Torino; year 2022/2023

# VOCAL DETECTION ACTIVITY 
Development of a Deep Learning model to be used in VAD.
## Audio preprocessing and model training
The [following script](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/preprocessing.py) preprocesses the audios. 

Model training and inference on the test-set can be found [here](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/training.ipynb). The model structure, the preprocessing arguments as well as the optimization hyperparameters have been found using a parameter grid search in order to respect the following constraints [notebook](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/testing.ipynb). 

|  | TEST ACCURACY |  MEDIAN LATENCY OVER TEST | MEMORY OCCUPATION |
|:-------|:-------|:--------|:--------|
|  CONSTRAINT| > .97    |  < 8 ms     | < 25KB |
|  RESULT | .99 | 5 ms | 13.2 KB (zip) |

Model optimization has been performed using a combination of width scaling and weight pruning. 

## Vocal detection and battery status monitoring 
The trained model is deployed in the following [following VAD script](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/VAD.py). 

When "Go" is detected with probability higher than .95, the battery monitoring should start; wheras, when "Stop" is detected with probability higher than .95, the battery monitoring should stop. In all other cases, any action on the battery monitoring must be performed (either probability less than .95, silence or other words detected). 

Battery monitoring: data about the battery status (in percentage) and the plugged power (boolean) are recorded in Redis timeseries.


## MQTT PUBLISHER AND SUBSCRIBER 
An [MQTT publisher](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/publisher.py) has been developed with the aim of publishing the battery status and percentage of the laptop on the message broker 'mqtt.eclipseprojects.io' at port 1883. 
The [MQTT subscriber](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/subscriber.ipynb) instead stores on Redis the obtained time series. 

## REST SERVER AND CLIENT
A [REST server](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/rest_server.ipynb) has been developed according to the specifications present in [this file](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/ML4IoT-HW3.pdf).
A [REST client](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/Homeworks/rest_client.ipynb) has been developed to test the server functionalities. 


# EMERGENCY VEHICLE SIREN DETECTION 
The pipeline has ben adapted and applied to the [Emergency Vehicle Sirens Detection task](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/tree/main/EMERGENCYSIRENDET)

# Machine-Learning-For-IoT
Machine Learning for IoT course at Politecnico di Torino; year 2022/2023

#VOCAL DETECTION ACTIVITY WITH DEEP LEARNING MODEL
Development of a Deep Learning model to be used in VAD.
## Audio preprocessing and model training
The [following script](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/preprocessing.py) preprocesses the audio and converts them into MFCCs on which to perfrom model training and inference. 

Model training and inference on the test-set can be found [here](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/training.ipynb). The model structure, the preprocessing arguments as well the optimization ones have been found using a parameter grid search in order to respect the following constraints (when trained on the standard version od Deepnote [notebook]()). 

|  | TEST ACCURACY |  MEDIAN LATENCY OVER TEST | MEMORY OCCUPATION |
|:-------|:-------|:--------|:--------|
|  CONSTRAINT| > .97    |  < 8 ms     | < 25KB |
|  RESULT | .99 | 5 ms | 13.2 KB (zip) |

Model optimization has beenperformed using a combination of width scaling and weight pruning. 

## Vocal detection and monitoring 
The trained model is deployed in the [following VAD script](https://github.com/AlessiaLeclercq/Machine-Learning-For-IoT/blob/main/VAD.py). 

When "Go" is detected with probability higher than .95, the battery monitoring should start; wheras, when "Stop" is detected with probability higher than .95, the battery monitoring should stop. In all other cases, any action on the battery monitoring must be performed (either probability less than .95, silence or other words detected). 

Battery monitoring: data about the battery status (in percentage) and the plugged power (boolean) are recorded in Redis timeseries.

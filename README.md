# Cryptocurrency_Prediction
Recurrent Neural Network model to predict the prices of 4 major [crytocurrencies](https://en.wikipedia.org/wiki/Cryptocurrency) 3 secs in the future.  
The RNN learns from the time-series dataset.  
The four cryptocurrencies are:

1. [Bitcoin](https://bitcoin.org/en/)
2. [Litecoin](https://litecoin.com/)
3. [Bitcoin Cash](https://www.bitcoincash.org/)
4. [Ethereum](https://www.ethereum.org/)  

The [data](data/) I have used are:  
- Open
- High
- Low
- Close
- Volume  

The Close column measures the final price at the end of each interval.   
The Volume column is how much of the asset was traded per each interval, In this project, these are 1 minute intervals. So, at the end of each minute, what was the price of the asset.  

## Required Installations  

The code is written in python 3.6 version.  

 [Tensorflow](https://www.tensorflow.org/) 
 
     pip3 install tensorflow=1.10 
     
 [Pandas](https://pandas.pydata.org/) 
 
     pip3 install pandas     
 
 [numpy](http://www.numpy.org/) 
 
     pip3 install numpy 
     
 [sklearn](https://scikit-learn.org/stable/) 
 
     pip install -U scikit-learn
     
Number of EPOCHS = 10  
Batch size = 64  
Future period of prediction = 3 sec  
Sequence length = 60  
Number of nodes in input layer = 128  
Number of nodes in output layer = 2  
Model type = Sequential  

## How to train the model

    python3 Cryptocurrency.py
    



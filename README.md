# Cryptocurrency_Prediction
Recurrent Neural Network model to predict the prices of 4 major [crytocurrencies](https://en.wikipedia.org/wiki/Cryptocurrency) 3 secs in the future.  
The RNN learns from the time-series dataset.  
The four cryptocurrencies are:

1. [Bitcoin](https://bitcoin.org/en/)
2. [Litecoin](https://litecoin.com/)
3. [Bitcoin Cash](https://www.bitcoincash.org/)
4. [Ethereum](https://www.ethereum.org/)  

The [data](data/) we will be using are 
- Open
- High
- Low
- Close
- Volume  

The Close column measures the final price at the end of each interval.   
The Volume column is how much of the asset was traded per each interval, In this project, these are 1 minute intervals. So, at the end of each minute, what was the price of the asset.  


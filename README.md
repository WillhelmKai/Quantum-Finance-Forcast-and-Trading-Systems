# RNN-QFFTS:

## Background & Abstrct

With average global trading transaction amounts exceeding 5 tril-lion dollars per day, foreign exchange (FOREX) is one of the largest financial markets in the world. With so much competition, finding a competitive ad-vantage, both through an intelligent financial forecasting systems and trad-ing strategy, can be highly effective and extremely profitable. With the adop-tion of the latest R&D on Quantum Finance Theory (QFT), we propose to build a more effective prediction and trading algorithm to better handle the highly chaotic and complex foreign exchange market. It is for this reason we put forward a novel Recurrent Neural Network based Quantum Finance Forecast and Trading System (RNN-QFFTS) for neural network prediction in tandem with the employment of a new kind of financial indicator called Quantum Price Level (QPL). From the experimental perspective, we com-pare the performance of 3 prediction models: FFBP, RNN, and RNN-QFFTS. Using the Meta Trader (MT) platform, we analyzed the previous 2048 days of daily trading data for each forex product to predict the following day’s open, high, low, and close. Utilizing our RNN-QFFTS, we then compare 3 trading algorithms: Moving Average with RSI (Relative Strength Index), Moving Average with RSI and QPL, and Moving Average with RSI and QPL integrated with our prediction. We find that QPL helps to accelerate gradient substantially which enabled us to handle a greater number of products in a smaller time period. With the application of QPL, in cooperation with major financial indicators, RNN-QFFTS achieves promising success rate in terms of trades and profitability. With the implementation of RNN-QFFTS, we were able to gauge prime times for investing occurring, at most, once a day or every other day. This prime time lowered the risk we had when engaging in trades and has the potential to increase the profitability substantially as compared to the traders not using QPL indicators or financial prediction re-sults for trading.

Keywords: Recurrent Neural Network, Quantum Finance, Quantum Price Level, Financial Forecasting, Intelligent Trading.

## Check the Performance
a. Prediction Low VS. Actual Low Price on a 200 days' real testing set
>  Testing Set in /Project_Data/.
>  Loss Steps and Prediction VS. Actual Graph in /Preciction_Results/.


b. A trading algorithm with return rate of about 15%/M is developed, according to prediction
>  An account statement for a half month is listed in ./Trading_Experiment_Result.htm 


## To reproduce prediction result:

0. Python version >= 3.6 (depend on your hardware requirement), tensorflow (CPU version only)>= 1.9.0 (GPU version to be defined), </br>
Install MateTrader4

1. To Collect financial Products' data
> Run /data_retrive/QF_peoject.mq4 on MT4

2. Mordify data storage setting with your own configuration
> Change 3 directories, data_add, prediction_add and model_add, in FFBP_L2, RNN_L2 or QPL_L2 to directories store .csv data, prediction results and network parameter

3. To train the model
> Run .py programs e.g.normal_distribution_QFFTS.py is the best performed ones

4. To get prediction for today
> Enable "continue" in codes, e.g. line 179 FFBP_l2 , and run again

5. Selecting most valuable features with genetic algorithm
> Run feature_selection.py


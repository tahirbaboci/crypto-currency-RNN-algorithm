# -*- coding: utf-8 -*-

import pandas as pd
from src.modules.Classify import classify
from src.modules.Read import read
from src.modules.Preprocessing import preprocess_df
from src.modules.MyModel import Train_model

#Close : is the price of the end of 60 seconds interval


def main():
    
    FUTURE_PERIOD_PREDICT = 3 # next minutes
    RATIO_TO_PREDICT = "LTC-USD" # which one we are going to predict
    
    #pd.show_versions()
    
    main_df = pd.DataFrame() # merging data frames # this is just an empty data frame
    main_df = read()
    print(main_df.head(10))
    
    main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT) # it will check 3 periods of the data in the future
    main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"])) #map close and future parameters to classify function
    print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(15))
    
    #time           close           future     target
    #124124124   96.38999999     96.47000111      1    # Here!!!!
    #234234234   96.51232323     96.40000211      0
    #452342342   96.44000233     96.44000233      0
    #623423454   96.47000111     96.40000002      0    # we can see 96.47000111
    
    # we have to build the sequences
    # we have to balace the data
    # we have to normalize the data
    # we have to scale the data
    
    # from now we are ready to make some sequenceses and train the model
    
    times = sorted(main_df.index.values)
    last_10percent = times[-int(0.10*len(times))] # get the time of last 10 percent === -int(0.10*len(times))
    print(last_10percent)
    
    test_main_df = main_df[(main_df.index >= last_10percent)] # it takes last 10% of the data
    main_df = main_df[(main_df.index < last_10percent)]  # it takes the rest (first 90%)
    
    train_x, train_y = preprocess_df(main_df)
    test_x, test_y = preprocess_df(test_main_df)
    
    
    print(f"train data: {len(train_x)} validation: {len(test_x)}")
    print(f"Dont buys: {train_y.count(0)}. buys: {train_y.count(1)}")
    print(f"TEST Dont buys: {test_y.count(0)}, buys: {test_y.count(1)}")
    
    Train_model(train_x, train_y, test_x, test_y)



if __name__ == '__main__':
    main()




# -*- coding: utf-8 -*-

import pandas as pd
from Classify import classify
from Read import read
from Preprocessing import preprocess_df
from MyModel import Train_model

#Close : is the price of the end of 60 seconds interval

SEQ_LEN = 60 # use last 60 minute of pricing data
FUTURE_PERIOD_PREDICT = 3 # next minutes
RATIO_TO_PREDICT = "LTC-USD" # which one we are going to predict

#pd.show_versions()

main_df = pd.DataFrame() # merging data frames # this is just an empty data frame
main_df = read()


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
last_5percent = times[-int(0.05*len(times))] # get the time of last 5 percent === -int(0.05*len(times))
print(last_5percent)

test_main_df = main_df[(main_df.index >= last_5percent)] # it takes last 5% of the data
main_df = main_df[(main_df.index < last_5percent)]  # it takes the rest (first 95%)

train_x, train_y = preprocess_df(main_df)
test_x, test_y = preprocess_df(test_main_df)

Train_model(train_x, train_y, test_x, test_y)

print(f"train data: {len(train_x)} validation: {len(test_x)}")
print(f"Dont buys: {train_y.count(0)}. buys: {train_y.count(1)}")
print(f"TEST Dont buys: {test_y.count(0)}, buys: {test_y.count(1)}")


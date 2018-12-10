# -*- coding: utf-8 -*-
import pandas as pd


ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD'] # files that are going to be used
#pd.show_versions()


def read():
    main_df = pd.DataFrame() # merging data frames # this is just an empty data frame

    for ratio in ratios:
        dataset = f"crypto_data/{ratio}.csv"
    
        df = pd.read_csv(dataset, names=['time','low','high','open','close','volume'])
        df.rename(columns={"close" : f"{ratio}_close", "volume" : f"{ratio}_volume"}, inplace=True) #inplace is in case so we dont need to redifine dataframe
    
        df.set_index("time", inplace=True)
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume
        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)
            
    return main_df


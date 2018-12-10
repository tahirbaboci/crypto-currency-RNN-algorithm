# -*- coding: utf-8 -*-

def classify(current, future):
    if float(future) > float(current):
        return 1 # means is going up and you should buy this
    else:
        return 0 # means is going down and you should sell it this



# -*- coding: utf-8 -*-

def classify(current, future):
    if float(future) > float(current):
        return 1 # means is good thing and you should buy this
    else:
        return 0 # means is not good thing and you shouldn't buy this



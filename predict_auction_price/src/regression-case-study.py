import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/home/noahprowell/galvanize/case-studies/regression-case-study/predict_auction_price/data/Train.csv')

"""
Columns to keep and drop
Keep: 
SalesPrice
YearMade: Outlier Years(look to remove)
MachineHoursCurrentMeter
state

Drop:
ProductGroupDesc - use column ProductGroup instead
Enclosure_Type - use column Enclosure instead
"""

def drop_column(df, column_name):
    df = df.drop(columns = 'column_name')
    return df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/noahprowell/galvanize/case-studies/regression-case-study/predict_auction_price/data/Train.csv')


"""
Columns to keep and drop

Keep: 
SalesPrice
YearMade: Outlier Years(look to remove)
MachineHoursCurrentMeter
state
datasource(if we keep, needs to be regularized)
Drive_System
Turbocharged
Transmission
Rise_Control
'Drive_System',
'Enclosure',
'Forks',
'Pad_Type',
'Ride_Control',
'Stick',
'Transmission',
'Turbocharged',
'Blade_Extension',
'Blade_Width',
'Hydraulics',
'Pushblock',
'Ripper',
'Scarifier',
'Tip_Control',
'Tire_Size',
'Coupler',
'Grouser_Tracks',
'Hydraulics_Flow',
'Track_Type',
'Undercarriage_Pad_Width',
'Stick_Length',
'Thumb',
'Pattern_Changer',
'Grouser_Type',
'Backhoe_Mounting',
'Blade_Type',
'Travel_Controls',
'Differential_Type',
'Steering_Controls',
'datasource',
'UsageBand',
'saledate',
'ProductSize',
'state',
'ProductGroup',]

Drop:
ProductGroupDesc - use column ProductGroup instead
Enclosure_Type - use column Enclosure instead
Coupler_System - use column Coupler instead (both have high amounts of nulls)
SalesID
auctioneerID
Engine_Horsepower
'fiModelDesc',
'fiBaseModel',
'fiSecondaryDesc',
'fiModelSeries',
'fiModelDescriptor',
'fiProductClassDesc',
['MachineID',
'ModelID',
"""

# pd.get_dummies(df, columns = [''], drop_first = True)

def drop_column(df, column_name):
    df = df.drop(columns = column_name)
    return df
    

def formatting(df):
    '''Format Dataframe for Price Prediction Model
    
    
    '''
    df_usage = df[df.UsageBand.notnull()]
    drop_list = ['ProductGroupDesc', 'Enclosure_Type', 'Coupler_System', 'SalesID', 'auctioneerID', 'Engine_Horsepower', 'MachineID', 'ModelID',
                'fiModelDesc', 'fiBaseModel','fiSecondaryDesc','fiModelSeries','fiModelDescriptor', 'fiProductClassDesc']
    dummies = ['UsageBand',
       'ProductSize',
       'state', 'ProductGroup', 
       'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Grouser_Tracks', 'Hydraulics_Flow',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Travel_Controls', 'Differential_Type', 'Steering_Controls']
    
    df_usage = pd.get_dummies(df_usage, columns = dummies)
    
    for col in drop_list:
        df_usage = drop_column(df_usage, col)

    df_usage['saledate'] = pd.to_datetime(df_usage['saledate'])
    df_usage['saledate'] = df_usage['saledate'].dt.year

    df_usage['age_at_sale'] = df_usage['saledate']- df_usage['YearMade']
    df_usage = df_usage[df_usage['age_at_sale'] <200]

    y = df_usage['SalePrice']
    X = df_usage.loc[:, df_usage.columns != 'SalePrice']

    return X,y


X,y = formatting(df)
X_train, X_test, y_train, y_test = train_test_split(X,y)




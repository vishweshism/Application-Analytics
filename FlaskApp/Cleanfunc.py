#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

#preprocessing



def prodClean(year, manufacturer, condition, fuel, odometer):
    #loading model
    filename = 'all_star.sav'
    load_model = pickle.load(open(filename,'rb'))
    filled = pd.read_csv('AllFilled.csv')
    filled = filled.drop(['price'], axis = 'columns')
    region = filled['region'].mode()[0]
    model = filled['model'].mode()[0]
    cylinders = filled['cylinders'].mode()[0]
    title_status = filled['title_status'].mode()[0]
    transmission = filled['transmission'].mode()[0]
    drive = filled['drive'].mode()[0]
    size = filled['size'].mode()[0]
    type_ = filled['type'].mode()[0]
    paint_color = filled['paint_color'].mode()[0]
    state = filled['state'].mode()[0]
    lat = filled['lat'].median()
    long = filled['long'].median()
    valuesin = []
    valuesin = {'region' : [region], 'year' : [year],'manufacturer' : [manufacturer], 'model' : [model], 'condition' : [condition],
               'cylinders': [cylinders], 'fuel' : [fuel], 'odometer' :[odometer], 'title_status' :[title_status], 'transmission' :[transmission],
               'drive': [drive], 'size': [size], 'type': [type_], 'paint_color':[paint_color], 'state': [state], 'lat':[lat],
               'long':long}
    dfpro = pd.DataFrame(valuesin, columns = ['region', 'year', 'manufacturer', 'model', 'condition','cylinders', 'fuel', 'odometer',
                                              'title_status', 'transmission', 'drive', 'size', 'type',
                                              'paint_color', 'state', 'lat', 'long'])
    df = pd.concat([filled, dfpro], ignore_index=True)
    num_cols=['year','odometer','lat','long']
    cat_cols=['region','manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','size','type','paint_color','state']
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(list(df[col].astype(str).values))
            df[col] = le.transform(list(df[col].astype(str).values))
    #scaling numerical data
    norm = StandardScaler()
    df['odometer'] = norm.fit_transform(np.array(df['odometer']).reshape(-1,1))
    df['year'] = norm.fit_transform(np.array(df['year']).reshape(-1,1))
    df['manufacturer'] = norm.fit_transform(np.array(df['manufacturer']).reshape(-1,1))
    df['model'] = norm.fit_transform(np.array(df['model']).reshape(-1,1))
    df['state'] = norm.fit_transform(np.array(df['state']).reshape(-1,1))
    df['region'] = norm.fit_transform(np.array(df['region']).reshape(-1,1))
    df['lat'] = norm.fit_transform(np.array(df['lat']).reshape(-1,1))
    df['long'] = norm.fit_transform(np.array(df['long']).reshape(-1,1))
    df = df.drop(df.columns[0], axis=1)
    predictn = load_model.predict(df)
    predictn = np.exp(predictn)
    output = predictn[-1]
    output = output.astype(int)
    return output

op = prodClean('2017', 'bmw', 'new', 'diesel', '12000')
print(op)


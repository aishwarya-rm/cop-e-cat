import sys
sys.path.append('../../')
from cop_e_cat.copecat import CopECat, CopECatParams
import os
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import mstats
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from functools import reduce
import pickle


    # deal with bug in xgboost 1.0.2 that results in 'KeyError: 'weight'
class NewXGBClassifier(XGBClassifier):
    @property
    def coef_(self):
        return None


if __name__ == '__main__':
    params = CopECatParams('params.json')
    print("Generating state spaces")
    copecat = CopECat(params)
    copecat.generate_state_spaces()
    allFrames = pd.read_csv(params.output_dir + 'allFrames.csv')

    print('Total number of processed adms =', len(allFrames.hadm_id.unique()),'; number of transitions =', len(allFrames))
    allFrames = allFrames.sort_values(by=['hadm_id', 'timestamp'])



# load xgb-vitals from file
xgb_vitals = pickle.load(open("xgb-vitals.pickle.dat", "rb"))

### subset relevant data categories (ones used in RL pipeline)

# adms - retains original ethnicity info - want to use more categories than white vs. non-white
# from adms_orig, create an ethnicity column which will replace that of adms

data_dir = '/tigress/BEE/mimic/usr/ecyoo/covidEHR/mimic/ecyoo/mimic3/data/admission/'
p_data_dir = '/tigress/BEE/mimic/usr/ecyoo/covidEHR/mimic/ecyoo/mimic3/data-processed/'

adms_orig = pd.read_pickle(data_dir + 'admissions.pkl')
# preprocessed by Niran
adms = pd.read_csv(p_data_dir + 'adms.csv')
adms.sort_values(by=['hadm'], inplace=True)

# vitals 
charts = pd.read_csv(p_data_dir + 'charts.csv')
charts.sort_values(by=['hadm'], inplace=True)

# inputs 
inputs = pd.read_csv(p_data_dir + 'inputs.csv')
inputs.sort_values(by=['hadm'], inplace=True)

identifier_cols = ['icustay','hadm']
# 'icustay','hadm' are patient identifiers 
static_cols = ['icustay','hadm','Age', 'Gender', 'Weight', 'Ethnicity', 'AdmitUnit', 'AdmitType']
drugs_cols = ['Anticoagulants', 'BetaBlockers', 'CaBlockers', 'Fluids', 'Insulin', 
         'Sedation', 'Paralytics', 'Vasoactive']
vitals_cols = ['icustay','hadm']
labs_cols = ['icustay','hadm']
static = adms[static_cols]
drugs = inputs[inputs['inputkey'].isin(drugs_cols)]


WHITE = ['White', 'WHITE', 'WHITE - RUSSIAN','WHITE - BRAZILIAN', 'WHITE - EASTERN EUROPEAN', 'WHITE - OTHER EUROPEAN', 'PORTUGUESE']
BLACK = ['Black', 'BLACK/AFRICAN AMERICAN', 'BLACK/AFRICAN', 'BLACK/CAPE VERDEAN','BLACK/HAITIAN' ]
ASIAN = ['Asian', 'ASIAN',  'ASIAN - ASIAN INDIAN', 'ASIAN - CHINESE','ASIAN - VIETNAMESE', 'ASIAN - CAMBODIAN', 
         'ASIAN - OTHER' , 'ASIAN - FILIPINO', 'ASIAN - KOREAN', 'ASIAN - JAPANESE', 
         'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']
MIDDLE = ['MiddleEastern', 'MIDDLE EASTERN']
LATINO = ['Latino','HISPANIC OR LATINO','HISPANIC/LATINO - PUERTO RICAN',  'HISPANIC/LATINO - DOMINICAN', 
          'HISPANIC/LATINO - COLOMBIAN', 'HISPANIC/LATINO - MEXICAN', 'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)',
          'HISPANIC/LATINO - GUATEMALAN', 'HISPANIC/LATINO - SALVADORAN', 'SOUTH AMERICAN']
NATIVE = ['Native', 'AMERICAN INDIAN/ALASKA NATIVE', 'AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE'] 
OTHER = ['Other', 'OTHER', 'MULTI RACE ETHNICITY']
UNKNOWN = ['Unknown','UNKNOWN/NOT SPECIFIED', 'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN']
                                                                                                   
# load xgb-vitals from file                                                                                 
xgb_vitals = pickle.load(open("xgb-vitals.pickle.dat", "rb"))                                               
                                                                                                            
### subset relevant data categories (ones used in RL pipeline)                                              
                                                                                                            
# adms - retains original ethnicity info - want to use more categories than white vs. non-white             
# from adms_orig, create an ethnicity column which will replace that of adms                                
                                                                                                            
data_dir = '/tigress/BEE/mimic/usr/ecyoo/covidEHR/mimic/ecyoo/mimic3/data/admission/'                       
p_data_dir = '/tigress/BEE/mimic/usr/ecyoo/covidEHR/mimic/ecyoo/mimic3/data-processed/'                     
                                                                                                            
adms_orig = pd.read_pickle(data_dir + 'admissions.pkl')                                                     
# preprocessed by Niran                                                                                     
adms = pd.read_csv(p_data_dir + 'adms.csv')                                                                 
adms.sort_values(by=['hadm'], inplace=True)                                                                 
                                                                                                            
# vitals                                                                                                    
charts = pd.read_csv(p_data_dir + 'charts.csv')                                                             
charts.sort_values(by=['hadm'], inplace=True)                                                               
                                                                                                            
# inputs                                                                                                    
inputs = pd.read_csv(p_data_dir + 'inputs.csv')                                                             
inputs.sort_values(by=['hadm'], inplace=True)                                                               
                                                                                                            
identifier_cols = ['icustay','hadm']                                                                        
# 'icustay','hadm' are patient identifiers                                                                  
static_cols = ['icustay','hadm','Age', 'Gender', 'Weight', 'Ethnicity', 'AdmitUnit', 'AdmitType']           
drugs_cols = ['Anticoagulants', 'BetaBlockers', 'CaBlockers', 'Fluids', 'Insulin',                          
         'Sedation', 'Paralytics', 'Vasoactive']                                                            
# TO DO: Heart rate, Respiratory rate, Temperature, Arterial pH,Non-invasive blood pressure (systolic, dias\
-UU-:----F1  script.py      12% L71   Git:main  (Python) ---------------------------------------------------
                                                                                             ethnicity_cats = [WHITE, BLACK, ASIAN, MIDDLE, LATINO, NATIVE, OTHER, UNKNOWN]

for category in ethnicity_cats: 
    adms_orig.loc[(adms_orig['ethnicity']).isin(category), 'ethnicity'] = str(category[0])
multi_ethnicities = adms_orig[['icustay', 'hadm', 'ethnicity']]

adms_col_order = adms.columns.tolist()
# merge with original adms df
adms = adms.merge(multi_ethnicities, on =['icustay', 'hadm'], how = 'inner')
adms.drop('Ethnicity', axis = 1, inplace=True)
adms.rename(columns = {'ethnicity': 'Ethnicity'}, inplace=True) 
adms_cols = adms.columns.tolist()
adms = adms[adms_col_order]

# one-hot-encode: want to convert categorical data into numeric form
X = adms['Ethnicity']
one_hot_enc = OneHotEncoder()
x_ohenc = one_hot_enc.fit_transform(X.values.reshape(len(X),1))

ethnicity_ohe_df = pd.DataFrame(data = x_ohenc.toarray(), 
                     columns=one_hot_enc.categories_)
# replace original Ethnicity column with one-hot-encoded ethnicity matrix
adms = adms.merge(ethnicity_ohe_df,left_index=True, right_index=True)

adms.columns.values # returns ('Asian',), ('Black',),... as a result of merging
# reformat ethnicity columns
adms.rename(columns={('Asian',): 'Race_Asian', ('Black',): 'Race_Black',
                    ('Latino',): 'Race_Latino', ('MiddleEastern',): 'Race_MiddleEastern',
                    ('Native',): 'Race_Native', ('Other',): 'Race_Other', 
                    ('Unknown',): 'Race_Unknown',  ('White',): 'Race_White'}, inplace=True)

# select features for training set 
adms_cols_to_drop = ["Ethnicity", "icu_admit","icu_discharge","icu_los", "exp"]     
adms_vals_only = adms.drop(adms_cols_to_drop, axis = 1)

# make a separate column for each unqiue 'chartkey' element
drugs.groupby(['icustay','hadm', 'inputkey']).sum().reset_index()
charts = pd.pivot_table(charts, index = ['icustay', 'hadm'], columns = ['chartkey'], values = ['value_num'],aggfunc= 'first', fill_value=0).reset_index() 
drugs = pd.pivot_table(drugs, index = ['icustay', 'hadm'], columns = ['inputkey'], values = ['amount'], aggfunc = 'first', fill_value=0).reset_index()

# join two column levels 'chartkey' and 'value_num'
drugs.columns = [''.join((col[1], col[0])) for col in drugs.columns]
drugs.columns = [i.replace('amount', '') for i in drugs.columns]
charts.columns = [''.join((col[1], col[0])) for col in charts.columns]
charts.columns = [i.replace('value_num', '') for i in charts.columns]

# merge all dataframes - adms, drugs, charts
dataset = adms_vals_only.merge(drugs,on=['icustay','hadm']).merge(charts,on=['icustay','hadm'])

# baseline - no removing outliers 
raw_data = adms_vals_only.merge(drugs,on=['icustay','hadm']).merge(charts,on=['icustay','hadm'])
# remove outliers
def WinsorizeStats(data):
    out = mstats.winsorize(data, limits=[0.05, 0.05])
    return out

# columns excluding patient identifiers 
val_cols_only = np.setdiff1d(dataset.columns, identifier_cols)
output_cols = ['h_exp']
input_cols = np.setdiff1d(val_cols_only, output_cols)

outliers_removed = pd.DataFrame(data = WinsorizeStats(dataset[input_cols]), columns = input_cols)
dataset[input_cols] = outliers_removed

# preprocessed data 
# rescale inputs
scale = MinMaxScaler(feature_range = (0,1))

output_cols = ['h_exp']
input_cols = np.setdiff1d(val_cols_only, output_cols)

# designate final inputs
X = pd.DataFrame(data = scale.fit_transform(dataset[input_cols]), columns = input_cols)

# designate final output
Y = dataset[output_cols]

# split data into train and test sets
seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y.values.reshape(X.shape[0],), test_size=test_size, random_state=seed)
# test/train split removes columns from dataframe
# put feature names back into test data -> need later for important feature labels
X_test = pd.DataFrame(data = X_test, columns = X.columns) 

# fit model to training data
model = NewXGBClassifier()
eval_set = [(X_test, y_test)]
model_fit = model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=['auc', 'error', 'logloss'], eval_set=eval_set, verbose=True)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# plot feature importance
#fig, axes = plt.subplots(nrows=1, ncols=2)
fig, (ax1, ax2) = plt.subplots(1, 2)
plot_importance(model, ax=ax1, max_num_features = 10, color = '#029E73')
plt.title('Top contributing features comparison')

ax1.set_title('COP-E-CAT')
#plt.savefig(save_dir+'xgboost_all_importance.png')

plot_importance(model_r, ax=ax2, max_num_features = 10, color = '#029E73')
ax2.set_title('Raw data')


# plot feature importance
fig, ax = plt.subplots(figsize=(8,6))
plot_importance(model_r, ax=ax, max_num_features = 20, color = '#029E73')
plt.title('XGBoost top contributing features')
plt.savefig(save_dir+'xgboost_all_importance.png')



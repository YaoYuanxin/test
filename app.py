"""
The main body of the app
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt, mpld3
from mpld3 import plugins
import streamlit.components.v1 as components

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from keras.models import load_model
from pre_process import *

st.write("""
# This app predicts the birth weight and the diagnosis of possible LGA and Macrosomia.
""")

##### Mother's information, Non-Sequential Input #####
######################################################
st.sidebar.header("Mother's Information")

def non_sequential_input():
    wt_before_preg = st.sidebar.slider("Mother's Weight Before Pregnancy (kg)", 37.00,112.00,59.18)
    height = st.sidebar.slider("Mother's Height (cm)", 147.00,186.00,165.95)
    NoPrevPreg = st.sidebar.slider("Number of Previous Pregnancies", 1,2,1)
    
    hpb = st.sidebar.selectbox('Does the mother have **High Blood Pressure**?',('Yes','No'))
    cardiac = st.sidebar.selectbox('Does the mother have **Cardiac Diseases**?',('Yes','No'))
    baseline_diabetes = st.sidebar.selectbox('Does the mother have **Diabetes**?',('Yes','No'))
    renal = st.sidebar.selectbox('Does the mother have **Renal Disorder**?',('Yes','No'))
    reg_smoke = st.sidebar.selectbox('Is the mother a **Regular Smoker**?',('Yes','No'))



    non_sequential_input_data = {'wt_before_preg':wt_before_preg,
                'height':height,
                'NoPrevPreg':NoPrevPreg,
                'hpb': hpb,
                'cardiac': cardiac,
                'baseline_diabetes': baseline_diabetes,
                'renal': renal,
                'reg_smoke': reg_smoke
}

    non_sequential_input_df = pd.DataFrame(non_sequential_input_data, index=[0])
    non_sequential_input_df = non_sequential_input_df.replace({"Yes":1, "No":0})
    return non_sequential_input_df

input_df_mom = non_sequential_input()


# Display Non-Sequential Input
st.subheader("***Mother's Information***")
st.write("1 indicates 'yes' and 0 indicates 'no'.")
st.write(input_df_mom)



##### Fetus's information, Sequential Input #####
######################################################
st.sidebar.header("Ultrasound Measurements")
#Information at the 17th Week
st.sidebar.subheader("Measurements Around the **17th Week** of Gestational Age")
def sequential_input_17():
    gadays = st.sidebar.slider("**Gestational Age Days** Around the 17th Week",84,157,122)
    bpd_mm = st.sidebar.slider("**Bipateral Diameter(BPD)** in milimeters",20.00,62.00,39.86)
    mad_mm = st.sidebar.slider("**Middle Abdominal Diameter(MAD)** in milimeters",20.00,64.00,37.91)
    fl_mm = st.sidebar.slider("**Femur Length(FL)** in milimeters",10.00,47.00,24.57)

    sequential_input_data_17 = {
        "gadays":gadays,
        "bpd_mm":bpd_mm,
        "mad_mm":mad_mm,
        "fl_mm":fl_mm
    }
    sequential_input_df_17 = pd.DataFrame(sequential_input_data_17, index=[0])
    return sequential_input_df_17

sequential_input_17_df = sequential_input_17()

#Information at the 25th Week
st.sidebar.subheader("Measurements Around the **25th Week** of Gestational Age")
def sequential_input_25():
    gadays = st.sidebar.slider("**Gestational Age Days** Around the 25th Week",158,201,175)
    bpd_mm = st.sidebar.slider("**Bipateral Diameter(BPD)** in milimeters",47.00,96.00,64.51)
    mad_mm = st.sidebar.slider("**Middle Abdominal Diameter(MAD)** in milimeters",50.00,109.00,64.84)
    fl_mm = st.sidebar.slider("**Femur Length(FL)** in milimeters",36.00,71.00,46.64)

    sequential_input_data_25 = {
        "gadays":gadays,
        "bpd_mm":bpd_mm,
        "mad_mm":mad_mm,
        "fl_mm":fl_mm
    }
    sequential_input_df_25 = pd.DataFrame(sequential_input_data_25, index=[0])
    return sequential_input_df_25

sequential_input_25_df = sequential_input_25()

#Information at the 33th Week
st.sidebar.subheader("Measurements Around the **33rd Week** of Gestational Age")
def sequential_input_33():
    gadays = st.sidebar.slider("**Gestational Age Days** Around the 33rd Week",202,246,230)
    bpd_mm = st.sidebar.slider("**Bipateral Diameter(BPD)** in milimeters",66.00,98.00,85.95)
    mad_mm = st.sidebar.slider("**Middle Abdominal Diameter(MAD)** in milimeters",66.00,111.00,92.40)
    fl_mm = st.sidebar.slider("**Femur Length(FL)** in milimeters",47.00,74.00,64.14)

    sequential_input_data_33 = {
        "gadays":gadays,
        "bpd_mm":bpd_mm,
        "mad_mm":mad_mm,
        "fl_mm":fl_mm
    }
    sequential_input_df_33 = pd.DataFrame(sequential_input_data_33, index=[0])
    return sequential_input_df_33

sequential_input_33_df = sequential_input_33()

#Information at the 37th Week
st.sidebar.subheader("Measurements Around the **37th Week** of Gestational Age")
def sequential_input_37():
    gadays = st.sidebar.slider("**Gestational Age Days** Around the 37th Week",247,276,259)
    bpd_mm = st.sidebar.slider("**Bipateral Diameter(BPD)** in milimeters",78.00,104.00,92.97)
    mad_mm = st.sidebar.slider("**Middle Abdominal Diameter(MAD)** in milimeters",85.00,124.00,104.97)
    fl_mm = st.sidebar.slider("**Femur Length(FL)** in milimeters",52.00,82.00,71.51)

    sequential_input_data_37 = {
        "gadays":gadays,
        "bpd_mm":bpd_mm,
        "mad_mm":mad_mm,
        "fl_mm":fl_mm
    }
    sequential_input_df_37 = pd.DataFrame(sequential_input_data_37, index=[0])
    return sequential_input_df_37

sequential_input_37_df = sequential_input_37()



sequential_input_all = pd.concat([sequential_input_17_df,sequential_input_25_df,
                                sequential_input_33_df,sequential_input_37_df],axis = 0)

sequential_input_all['efw'] = efw19(sequential_input_all.bpd_mm,
                                    sequential_input_all.mad_mm,sequential_input_all.fl_mm)
gaweeks = ["17th Week", "25th Week", "33rd Week", "37th Week"]
sequential_input_all.insert(0, column = 'Gestational Week', value = gaweeks)
sequential_input_all.set_index("Gestational Week", inplace = True)

st.subheader("***Ultrasound Measurement Information***")
st.write("The fetus' ultrasound measurements over 4 gestational age time steps.")
st.write(sequential_input_all)

 
 

## Data Augmentation Step ###
 ## Strategy 1: Find the closest existing id from the ORIGINAL DATA and use the augmented data from that id
 ##        Pros: Could be faster
 ##        Cons: Might not be as accurate
 ##              Matching is quite difficult (match gadays and efw)






 ## Strategy 2: Fit the augmentation model from scratch 
 ##        Pros: As accurate as possible
 ##        Cons: Can be slow


 ## Strategy 3: Find the closest existing id from the Augmented DATA and use the augmented data from that id

 ### pre-process input data
sequential_input = sequential_input_all[['gadays','efw']]
sequential_input = sequential_input.transpose()
sequential_input = sequential_input.astype("float")
sequential_input = sequential_input.rename(columns=sequential_input.iloc[0])
sequential_input = sequential_input.drop(sequential_input.index[0])

st.write("pre-processed input sequential data")
st.write(sequential_input)


# find the closest existing id in the data pool
augmented_data_quadratic_df_ready = pd.read_csv("/workspaces/test/augmented_data_quadratic_df_ready.csv",index_col=0)
augmented_data_quadratic_df_ready = augmented_data_quadratic_df_ready.loc[:,:'301.0']

matching_df = pd.DataFrame()
for gaday in sequential_input_all.gadays:
    matching_df = pd.concat([matching_df,augmented_data_quadratic_df_ready.loc[:,augmented_data_quadratic_df_ready.columns == 
                    str(float(gaday))]],axis =1)

matching_id = (matching_df - sequential_input.values).abs().sum(axis = 1).idxmin()
st.write("The closes match in the data pool is id", matching_id,".")

training_seq = augmented_data_quadratic_df_ready[augmented_data_quadratic_df_ready.index == matching_id]

# training_seq.loc[:,training_seq.columns.astype('float').isin(sequential_input.columns)] = sequential_input.values

augmented_data_quadratic_df_ready = pd.concat([augmented_data_quadratic_df_ready,training_seq],axis = 0)

## Preprocess the input data ####
#################################
y_days = 53
n_features = 1
y = augmented_data_quadratic_df_ready.iloc[:,-y_days:]
X = augmented_data_quadratic_df_ready.drop(columns = y.columns)

  
scaler_1 = StandardScaler()
X_scaled = scaler_1.fit_transform(X)
y_scaled = scaler_1.fit_transform(y)

X_scaled_pred = X_scaled[-1,:]
y_scaled_pred = y_scaled[-1,:]

X_scaled_pred = X_scaled_pred.reshape(X_scaled_pred.shape[0],n_features)
X_scaled_pred = X_scaled_pred.reshape(1,X_scaled_pred.shape[0],1)

covariates_5_record = pd.read_csv("/workspaces/test/covariates_5_record.csv",index_col=0)
covariates_5_record = covariates_5_record.iloc[:,1:]
covariates_5_record = pd.concat([covariates_5_record,input_df_mom],axis = 0)
scalar_2 = StandardScaler()
covariates_5_record.wt_before_preg = scalar_2.fit_transform(np.array(covariates_5_record.wt_before_preg).reshape(-1,1))
covariates_5_record.height = scalar_2.fit_transform(np.array(covariates_5_record.height).reshape(-1,1))
input_df_mom_standardized = covariates_5_record.iloc[-1]
input_df_mom_standardized = pd.DataFrame(input_df_mom_standardized).transpose()

# st.write("standardized covariates")
# st.write(input_df_mom_standardized)

# st.write("traning sequence along with the pool of data")
# st.write(augmented_data_quadratic_df_ready)
# print(augmented_data_quadratic_df_ready.shape)

# st.write("scaled traning sequence")
# st.write(X_scaled_pred)
# print(X_scaled_pred.shape)
# print(input_df_mom_standardized.shape)

# pd.DataFrame(X_scaled).to_csv("X_scaled.csv")
# input_df_mom_standardized.to_csv("input_df_mom_standardized.csv")

## Load the Pre-Trained Models and generate predictions ##
##########################################################
# model_qua_lstm_std = load_model("/workspaces/test/model_qua_lstm_std_75.h5")
model_nl_rnn_std_5 = load_model("/workspaces/test/model_nl_rnn_std_5.h5")

pred_y = model_nl_rnn_std_5.predict([X_scaled_pred,input_df_mom_standardized])
true_pred = scaler_1.inverse_transform(pred_y)
true_pred_df = pd.DataFrame(true_pred)
true_pred_df.columns = y.columns

true_pred_df.to_csv("true_pred_df.csv")


df_90th_10th = pd.read_csv("/workspaces/test/df_90th_10th.csv",index_col=0)
raw_copy_5_record = pd.read_csv("/workspaces/test/raw_copy_5_record.csv",index_col=0)


lga_true = is_lga(true_pred_df,df_90th_10th)
macro_true = is_macro(true_pred_df)
lga_true.loc[0] = lga_true.loc[0].map({True: "Yes", False: "No"})
macro_true.loc[0] = macro_true.loc[0].map({True: "Yes", False: "No"})

st.write("Prediction Result")
result = pd.concat([true_pred_df,lga_true,macro_true],axis = 0)
result.insert(0, column = "Result", value = ["Predicted Birthweight", "LGA Diagnosis", "Macrosomia Diagnosis"])
result.set_index("Result",inplace=True)
st.write(result)

# result.to_csv("result.csv")


## Interactive Plot ################
#################################3
lower_bound = float(result.columns[0])
upper_bound = float(result.columns[-1])
lga_limit_df = df_90th_10th.loc[((df_90th_10th["gadays"]>=lower_bound) & (df_90th_10th["gadays"]<=upper_bound))]

lga_fig = plt.figure(figsize=(5,5))
plt.plot(lga_limit_df["gadays"], lga_limit_df["90th percentile BW"],color='r', marker='.')
plt.plot(result.iloc[0].astype(float).index.astype('float'),result.iloc[0].values.astype('float'), color = 'b', marker = ',')


# Define some CSS to control our custom labels
css = '''
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
'''

for axes in lga_fig.axes:
  for line in axes.get_lines():
    xy_data = line.get_xydata()
    labels = []
    for x,y in xy_data:
      html_label = f'<table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> </thead> <tbody> <tr> <th>x</th> <td>{x}</td> </tr> <tr> <th>y</th> <td>{y}</td> </tr> </tbody> </table>'
      labels.append(html_label)
    tooltip = plugins.PointHTMLTooltip(line, labels, css=css)
    plugins.connect(lga_fig, tooltip)

fig_html = mpld3.fig_to_html(lga_fig)
components.html(fig_html, height=500, width=500)


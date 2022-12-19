"""
Test file to test interactive plots
"""
import matplotlib.pyplot as plt
from mpld3 import plugins
import streamlit as st
import pandas as pd
import numpy as np
import mpld3
import streamlit.components.v1 as components

result = pd.read_csv("/workspaces/test/result.csv",index_col=0)
df_90th_10th = pd.read_csv("/workspaces/test/df_90th_10th.csv",index_col=0)

lower_bound = float(result.columns[0])
upper_bound = float(result.columns[-1])
lga_limit_df = df_90th_10th.loc[((df_90th_10th["gadays"]>=lower_bound) & (df_90th_10th["gadays"]<=upper_bound))]

lga_fig = plt.figure(figsize=(10,10))
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
components.html(fig_html, height=1000, width=850)
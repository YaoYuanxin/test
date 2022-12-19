import matplotlib.pyplot as plt, mpld3
import numpy as np
from mpld3 import plugins
import streamlit.components.v1 as components
import pandas as pd

result = pd.read_csv("/workspaces/test/result.csv",index_col=0)
df_90th_10th = pd.read_csv("/workspaces/test/df_90th_10th.csv",index_col=0)



lga_fig = plt.figure(figsize=(8,8))
plt.plot(np.array(df_90th_10th.gadays),np.array(df_90th_10th['90th percentile BW']),color='r', marker='.')


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
components.html(fig_html, height=850, width=850)
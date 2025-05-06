## The file train.py contains several time series in the form (nb_ts, seq_len, nb_features). I want to visualize them using an advanced visualization library like plotly.


import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data = np.load('train.npy')[:10]

## Min and Max in the data
raw = np.load('test.npy')
print('Min:', np.min(raw))
print('Max:', np.max(raw))


# Assuming data is a 3D numpy array with shape (nb_ts, seq_len, nb_features)
# Convert the data to a pandas DataFrame for easier manipulation
df = pd.DataFrame(data.reshape(data.shape[0], -1))
# Create a list of colors for the lines
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Create a Plotly figure
fig = go.Figure()
# Add traces for each time series
for i in range(data.shape[0]):
    fig.add_trace(go.Scatter(
        x=np.arange(data.shape[1]),
        y=data[i, :, 0],  # Assuming you want to plot the first feature
        mode='lines',
        name=f'Time Series {i+1}',
        line=dict(color=colors[i % len(colors)], width=2)
    ))
# Update layout
fig.update_layout(
    title='Time Series Visualization',
    xaxis_title='Time Step',
    yaxis_title='Value',
    legend_title='Time Series',
    template='plotly_white'
)
# Show the figure
fig.show()
# Save the figure as an HTML file
fig.write_html('plots/time_series_visualization.html')

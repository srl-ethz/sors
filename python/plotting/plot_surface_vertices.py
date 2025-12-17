import numpy as np
import plotly.graph_objects as go

# Load the CSV file into a numpy array
# Replace 'your_file.csv' with the path to your CSV file
vertex_positions = np.loadtxt('surface_vertices.csv', delimiter=',')

# Create a scatter plot using plotly
fig = go.Figure(data=[go.Scatter3d(
    x=vertex_positions[:, 0],
    y=vertex_positions[:, 1],
    z=vertex_positions[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',  # Change color as needed
        opacity=0.8
    )
)])

# Update layout for better visuals
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title='Interactive 3D Scatter Plot'
)

# Save the plot as an HTML file
fig.write_html('interactive_3d_scatter_plot.html')
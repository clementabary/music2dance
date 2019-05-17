from utils import load_dataset, stickwise
import plotly.graph_objs as go
from plotly.offline import iplot
import networkx as nx


def to_graph_data(frame):

    # Create networkx graph
    G = nx.Graph()

    # Add all named nodes
    for n in range(frame.shape[0]):
        G.add_node(n, pos=frame[n])

    # Add all relevant edges
    # TODO: Fix two last added nodes
    e = [(0, 1), (2, 7), (2, 16), (16, 17), (17, 18),
         (19, 20), (7, 8), (8, 9), (10, 11), (3, 4),
         (4, 5), (5, 6), (12, 13), (13, 14), (14, 15), (3, 12)]
    G.add_edges_from(e)

    # Create edge scatter plot
    edge_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        line=dict(width=3.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0, z0 = G.node[edge[0]]['pos']
        x1, y1, z1 = G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_trace['z'] += tuple([z0, z1, None])

    # Create node scatter plot
    node_trace = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode='markers',
        hoverinfo='text',
        text=list(range(frame.shape[0])),
        marker=dict(symbol='circle', size=3.5))

    for node in G.nodes():
        x, y, z = G.node[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['z'] += tuple([z])

    # Add special nodes and edges
    x0, y0, z0 = (G.node[0]['pos'] + G.node[1]['pos'])/2
    x1, y1, z1 = (G.node[3]['pos'] + G.node[12]['pos'])/2
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    x0, y0, z0 = (G.node[3]['pos'] + G.node[12]['pos'])/2
    x1, y1, z1 = G.node[2]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    x0, y0, z0 = (G.node[10]['pos'] + G.node[11]['pos'])/2
    x1, y1, z1 = G.node[9]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    x0, y0, z0 = (G.node[19]['pos'] + G.node[20]['pos'])/2
    x1, y1, z1 = G.node[18]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    return node_trace, edge_trace


def visualize_graph(data):

    # Create layout for graph
    # TODO: set custom view & fix hyperparameters
    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=True,
                showticklabels=True, title='')
    graph_layout = go.Layout(title='Stick figure', width=450, height=550,
                             scene=dict(xaxis=axis, yaxis=axis, zaxis=axis))
    graph_fig = go.Figure(data=data, layout=graph_layout)
    iplot(graph_fig)


if __name__ == '__main__':
    # Load dataset
    datasetf = 'Music-to-Dance-Motion-Synthesis-master'
    dataset = stickwise(load_dataset(datasetf))

    # Take sample frame
    example = dataset[0]

    # Build network nodes & edges
    node_trace, edge_trace = to_graph_data(example)

    # Visualize 3D graph
    visualize_graph([node_trace, edge_trace])

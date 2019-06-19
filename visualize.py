from utils import load_sticks, stickwise
from utils import SequenceDataset
import plotly.graph_objs as go
from plotly.offline import iplot
import networkx as nx
import plotly.io as pio
import numpy as np
import cv2
# import librosa


def to_2d_graph_data(frame):

    # For visualization purposes
    frame = frame[:, [2, 1]]

    # Create networkx graph
    G = nx.Graph()

    # Add all named nodes
    N = frame.shape[0]
    for n in range(N):
        G.add_node(n, pos=frame[n])

    # Add all relevant edges
    # TODO: Fix two last added nodes
    e = [(0, 1), (2, 7), (2, 16), (16, 17), (17, 18),
         (19, 20), (7, 8), (8, 9), (10, 11), (3, 4),
         (4, 5), (5, 6), (12, 13), (13, 14), (14, 15), (3, 12)]
    G.add_edges_from(e)

    # Create edge scatter plot
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=3.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create node scatter plot
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        text=list(range(frame.shape[0])),
        marker=dict(symbol='circle', size=6, color=list(range(N)), colorscale='Viridis'))

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    # Add special nodes and edges
    x0, y0 = (G.node[0]['pos'] + G.node[1]['pos']) / 2
    x1, y1 = (G.node[3]['pos'] + G.node[12]['pos']) / 2
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

    x0, y0 = (G.node[3]['pos'] + G.node[12]['pos']) / 2
    x1, y1 = G.node[2]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

    x0, y0 = (G.node[10]['pos'] + G.node[11]['pos']) / 2
    x1, y1 = G.node[9]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

    x0, y0 = (G.node[19]['pos'] + G.node[20]['pos']) / 2
    x1, y1 = G.node[18]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

    return [edge_trace, node_trace]


def to_3d_graph_data(frame):

    # For visualization purposes
    frame[:, [2, 1]] = frame[:, [1, 2]]

    # Create networkx graph
    G = nx.Graph()

    # Add all named nodes
    N = frame.shape[0]
    for n in range(N):
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
        text=list(range(N)),
        marker=dict(symbol='circle', size=5))

    for node in G.nodes():
        x, y, z = G.node[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['z'] += tuple([z])

    # Add special nodes and edges
    x0, y0, z0 = (G.node[0]['pos'] + G.node[1]['pos']) / 2
    x1, y1, z1 = (G.node[3]['pos'] + G.node[12]['pos']) / 2
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    x0, y0, z0 = (G.node[3]['pos'] + G.node[12]['pos']) / 2
    x1, y1, z1 = G.node[2]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    x0, y0, z0 = (G.node[10]['pos'] + G.node[11]['pos']) / 2
    x1, y1, z1 = G.node[9]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    x0, y0, z0 = (G.node[19]['pos'] + G.node[20]['pos']) / 2
    x1, y1, z1 = G.node[18]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    edge_trace['z'] += tuple([z0, z1, None])

    return [edge_trace, node_trace]


def visualize_2d_graph(data, show=False, save=''):

    # Create layout for graph
    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=True,
                showticklabels=False, title='')
    graph_layout = go.Layout(width=450, height=550, showlegend=False,
                             scene=dict(xaxis=axis, yaxis=axis))
    graph_fig = go.Figure(data=data, layout=graph_layout)
    if save.endswith('.png'):
        pio.write_image(graph_fig, save)
    if show:
        iplot(graph_fig)
    return graph_fig


def visualize_3d_graph(data, show=False):

    # Create layout for graph
    axis = dict(showbackground=False, showline=False, zeroline=False, showgrid=True,
                showticklabels=False, title='')
    cam = dict(up=dict(x=0, y=0, z=1),
               center=dict(x=0, y=0, z=0),
               eye=dict(x=-2.5, y=-0.1, z=0.1))
    graph_layout = go.Layout(width=450, height=550, showlegend=False,
                             scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, camera=cam))
    graph_fig = go.Figure(data=data, layout=graph_layout)
    if show:
        iplot(graph_fig)


def draw(frame, size):
    frame[:, 0] += size[0] // 2
    frame[:, 1] += size[1] // 2
    cvs = np.ones(size)
    cvs *= (255, 255, 255)
    color = (255, 0, 0)
    edge_list = [(0, 1), (3, 4), (4, 5), (5, 6),
                 (12, 13), (13, 14), (14, 15), (2, 7),
                 (7, 8), (8, 9), (10, 11), (2, 16),
                 (16, 17), (17, 18), (19, 20)]
    for node in frame:
        cv2.circle(cvs, (int(node[0]), int(node[1])), 4, color, -1)
    for (s, f) in edge_list:
        cv2.line(cvs,
                 (int(frame[s][0]), int(frame[s][1])),
                 (int(frame[f][0]), int(frame[f][1])),
                 color, 2)
    cv2.line(cvs,
             (int((frame[0][0] + frame[1][0]) / 2),
              int((frame[0][1] + frame[1][1]) / 2)),
             (int((frame[3][0] + frame[12][0]) / 2),
              int((frame[3][1] + frame[12][1]) / 2)),
             color, 2)
    cv2.line(cvs,
             (int(frame[3][0]), int(frame[3][1])),
             (int((frame[3][0] + frame[12][0]) / 2),
              int((frame[3][1] + frame[12][1]) / 2)),
             color, 2)
    cv2.line(cvs,
             (int(frame[12][0]), int(frame[12][1])),
             (int((frame[3][0] + frame[12][0]) / 2),
              int((frame[3][1] + frame[12][1]) / 2)),
             color, 2)
    cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])),
             (int((frame[3][0] + frame[12][0]) / 2),
              int((frame[3][1] + frame[12][1]) / 2)),
             color, 2)
    cv2.line(cvs,
             (int(frame[9][0]), int(frame[9][1])),
             (int((frame[10][0] + frame[11][0]) / 2),
              int((frame[10][1] + frame[11][1]) / 2)),
             color, 2)
    cv2.line(cvs,
             (int(frame[18][0]), int(frame[18][1])),
             (int((frame[19][0] + frame[20][0]) / 2),
              int((frame[19][1] + frame[20][1]) / 2)),
             color, 2)
    frame[:, 0] -= size[0] // 2
    frame[:, 1] -= size[1] // 2
    return np.flip(cvs, 0)


def frame_to_vid(frames, name, fps):
    w, h = 300, 300
    size = (w, h, 3)
    video = cv2.VideoWriter(
        name, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (h, w))
    for _ in range(len(frames)):
        video.write(draw(frames[_], size).astype('uint8'))
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load dataset
    datasetf = 'Music-to-Dance-Motion-Synthesis-master'
    stick_dataset = stickwise(load_sticks(datasetf), 'skeletons')
    seq_dataset = SequenceDataset(datasetf)

    # Select sample frame
    idx = 0
    example = stick_dataset[idx]

    # Build network nodes & edges
    trace_2d = to_2d_graph_data(example)
    trace_3d = to_3d_graph_data(example)

    # Visualize graphs
    visualize_2d_graph(trace_2d, show=True)
    visualize_3d_graph(trace_3d, show=True)

    # Save 2D graph instead
    visualize_2d_graph(trace_2d, save='foo.png')

    # Save 2D animation
    seq, _, dir = seq_dataset[0]
    # librosa.output.write_wav(dir+'/audio_extract.wav', np.asarray(audio), sr=44100)
    frame_to_vid(seq, dir + '/animation_extract.avi', fps=25)

import networkx as nx
import plotly.graph_objects as go


def plot_airports(G, labels):
    """
    Plot and show a given graph in the airport dataset.
    Colours the graph according to airport popularity.

    Args:
        G (nx.Graph) : The airport graph, as a networkx graph
        labels: a dictionary of node ids to labels

    Returns:
        a Plotly figure
    """

    # adapted from https://plotly.com/python/network-graphs/

    # give the nodes positions
    positions = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Airport popularity",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    node_popularity = []
    for node in G.nodes():
        node_popularity.append(labels.get(node, 0))

    node_trace.marker.color = node_popularity

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="The airports network",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig

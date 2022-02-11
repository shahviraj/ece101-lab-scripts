import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

circle_opts = {'alpha': 0.2, 'radius': 0.3,
               'linewidth': 2, 'linestyle': 'solid'}
header_opts = {'ha': 'center', 'va': 'center', 'fontsize': 10}
data_opts = {'ha': 'center', 'va': 'center',
             'fontsize': 20, 'fontweight': 'bold'}
fig_opts = {'figsize': (4, 4), 'dpi': 100}


def draw_hamming_code(data):
    data = str(data)
    p = [None, None, None]
    if len(data) == 4:
        # check that all characters are 0 or 1
        okay = True
        for c in data:
            if c not in ['0', '1']:
                okay = False
                break
        if not okay:
            raise ValueError('Invalid message, only 0 and 1 can be used')
        p[0] = int(data[0]) + int(data[3]) + int(data[1])
        p[0] = str(int(p[0] % 2))
        p[1] = int(data[0]) + int(data[3]) + int(data[2])
        p[1] = str(int(p[1] % 2))
        p[2] = int(data[1]) + int(data[2]) + int(data[3])
        p[2] = str(int(p[2] % 2))
    else:
        # check that all characters are 0, 1 or ?
        okay = True
        for c in data:
            if c not in ['0', '1', '?']:
                okay = False
                break
        if not okay:
            raise ValueError('Invalid message, only 0 and 1 can be used')
        if len(data) < 7:
            raise ValueError('Invalid message, exactly 4 characters are needed')
        p[0] = data[4]
        p[1] = data[5]
        p[2] = data[6]

    fig, ax = plt.subplots(**fig_opts)
    ax.set_axis_off()
    ax.grid(False)

    circle1 = plt.Circle((0.66, 0.66), color='r', **circle_opts)
    circle2 = plt.Circle((0.33, 0.66), color='y', **circle_opts)
    circle3 = plt.Circle((0.5, 0.33), color='b', **circle_opts)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    plt.text(0.5, 0.8, 'd1', color="#50483B", **header_opts)
    plt.text(0.5, 0.7, data[0], color="#50483B", **data_opts)
    plt.text(0.29, 0.42, 'd2', color="#304663", **header_opts)
    plt.text(0.36, 0.46, data[1], color="#304663", **data_opts)
    plt.text(1-0.29, 0.42, 'd3', color="#69326F", **header_opts)
    plt.text(1-0.36, 0.46, data[2], color="#69326F", **data_opts)
    plt.text(0.5, 0.58, 'd4', color="#200D36", **header_opts)
    plt.text(0.5, 0.50, data[3], color="#200D36", **data_opts)

    plt.text(0.17, 0.83, 'p1', color="#606121", **header_opts)
    plt.text(0.25, 0.7, p[0], color="#606121", **data_opts)
    plt.text(1-0.17, 0.83, 'p2', color="#4B1819", **header_opts)
    plt.text(1-0.25, 0.7, p[1], color="#4B1819", **data_opts)
    plt.text(0.5, 0.13, 'p3', color="#25245C", **header_opts)
    plt.text(0.5, 0.25, p[2], color="#25245C", **data_opts)

    plt.show()


class Graph():
    def __init__(self):
        self.graph = nx.Graph()
        self.pos = None

    def add_node(self, name, color=None) -> None:
        if name is not str:
            name = str(name)
        if color:
            self.graph.add_node(name, color=color)
        else:
            if name[0] == 'S':
                self.graph.add_node(name, color='#E66100')
            elif name[0] == 'D' or name[0] == 'I':
                self.graph.add_node(name, color='#5D3A9B')
            else:
                self.graph.add_node(name, color='black')

    def add_edge(self, u, v, capacity=1) -> None:
        if u is not str:
            u = str(u)
        if v is not str:
            v = str(v)
        self.graph.add_edge(u, v, cap=capacity,
                            cap_str=(str(capacity) + ""))

    def draw_graph(self, figsize=(6, 4)) -> None:
        plt.figure(figsize=figsize)
        self.pos = nx.spring_layout(self.graph, weight=None, iterations=100)

        def get_color(node):
            try:
                return self.graph.nodes[node]['color']
            except KeyError:
                return 'black'

        nx.draw(self.graph, self.pos, node_color=[get_color(
            node) for node in self.graph.nodes], with_labels=True, font_weight='bold', font_color='w')
        labels = nx.get_edge_attributes(self.graph, 'cap_str')
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=labels)

    # def draw_shortest_path(self) -> None:
    #     try:
    #         if self.weighted:
    #             self.path = nx.shortest_path(
    #                 self.graph, source='S', target='D', weight='weight')
    #         else:
    #             self.path = nx.shortest_path(
    #                 self.graph, source='S', target='D')
    #         path_edges = list(zip(self.path, self.path[1:]))
    #         nx.draw_networkx_edges(
    #             self.graph, self.pos, edgelist=path_edges, edge_color='purple', width=10, alpha=0.2)
    #     except nx.NetworkXNoPath:
    #         display(HTML('<h2 style="color:#B00020;">No path from S to D</h2>'))

    # def show_all_paths(self, figsize=(10, 10)) -> None:
    #     paths = [path for path in nx.all_simple_paths(
    #         self.graph, source='S', target='D')]
    #     paths = sorted(paths, key=lambda x: len(x))
    #     num_plots = len(paths)
    #     print(f'Found {num_plots} paths')
    #     plt.figure(figsize=figsize)
    #     for i, path in enumerate(paths):
    #         plt.subplot((num_plots // 4) + 1, 4, i + 1)
    #         path_edges = list(zip(path, path[1:]))
    #         nx.draw(self.graph, self.pos, node_color=[self.graph.nodes[node]['color']
    #                 for node in self.graph.nodes], with_labels=True, font_weight='bold', font_color='w')
    #         nx.draw_networkx_edges(
    #             self.graph, self.pos, edgelist=path_edges, edge_color='purple', width=10, alpha=0.2)
    #     plt.show()

    # def remove_edge(self, u, v) -> None:
    #     if u is not str:
    #         u = str(u)
    #     if v is not str:
    #         v = str(v)
    #     try:
    #         self.graph.remove_edge(u, v)
    #     except nx.NetworkXError:
    #         print("There was no edge between {} and {}".format(u, v))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

circle_opts = {'alpha': 0.2, 'radius': 0.3,
               'linewidth': 2, 'linestyle': 'solid'}
header_opts = {'ha': 'center', 'va': 'center', 'fontsize': 10}
data_opts = {'ha': 'center', 'va': 'center',
             'fontsize': 20, 'fontweight': 'bold'}

fig_opts = {'figsize': (9, 5), 'dpi': 100}
small_graph_opts = {'node_color': '#66023C',
                    'node_size': 50,
                    'edge_color': (0, 0, 0, 0.4),
                    'width': 2}
large_graph_opts = {'node_color': '#66023C',
                    'node_size': 10,
                    'edge_color': (0, 0, 0, 0.2),
                    'width': 1}
dia_path_opts = {'edge_color': '#1B3B72',
                 'width': 3}


def get_graph_opts(n):
    node_size = 50/np.log2(n)
    edge_color = (0, 0, 0, 0.8-np.log(n)/10)
    return {'node_color': '#66023C',
            'node_size': node_size,
            'edge_color': edge_color,
            'width': 2}


class Graph:
    def __init__(self, G):
        self.G = G
        self.n = G.number_of_nodes()
        self.pos = None
        self.dia_start = None
        self.dia_end = None
        self.dia = None
        self.all_pairs_shortest_path_length = None

    def draw_graph(self):
        fig, ax = plt.subplots(**fig_opts)
        self.pos = nx.spring_layout(self.G)
        nx.draw(self.G, self.pos, **get_graph_opts(self.n))

    def get_average_friends(self):
        neighbors = nx.average_neighbor_degree(self.G)
        return np.round(np.mean(list(neighbors.values())), 2)

    def get_diameter(self):
        if self.all_pairs_shortest_path_length is None:
            self.all_pairs_shortest_path_length = dict(
                nx.all_pairs_shortest_path_length(self.G))
        self.dia = 0
        self.dia_i = 0
        self.dia_j = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.all_pairs_shortest_path_length[i][j] > self.dia:
                    self.dia = self.all_pairs_shortest_path_length[i][j]
                    self.dia_i = i
                    self.dia_j = j
        return self.dia

    def highlight_diameter(self):
        dia_path = nx.shortest_path(self.G, self.dia_i, self.dia_j)
        path_edges = list(zip(dia_path, dia_path[1:]))
        nx.draw_networkx_edges(
            self.G, self.pos, edgelist=path_edges, **dia_path_opts)

    def show_path_length_distribution(self):
        if self.all_pairs_shortest_path_length is None:
            self.all_pairs_shortest_path_length = dict(
                nx.all_pairs_shortest_path_length(self.G))
        edge_lens = [list(d.values()) for d in self.all_pairs_shortest_path_length.values()]
        # flatten list of lists
        edge_lens = [item for sublist in edge_lens for item in sublist]
        edge_lens = [x for x in edge_lens if x > 0]
        plt.figure()
        plt.hist(edge_lens, bins=np.arange(0, self.dia+1, 1)-0.5)
        plt.xticks(np.arange(0, self.dia+1, 1))
        plt.title('Path Length Distribution')
        plt.xlabel('Path Length')
        plt.ylabel('Frequency')
        plt.show()
    
    def show_degree_distribution(self):
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        plt.figure()
        plt.title('Distribution of Number of Friends')
        plt.hist(degrees, bins=np.arange(0, max(degrees)+1, 1)-0.5)
        plt.xticks(np.arange(0, max(degrees)+1, 1))
        plt.xlabel('Number of Friends')
        plt.ylabel('Frequency')
        plt.show()


def small_world_graph(num_people, num_friends, chance):
    G = nx.connected_watts_strogatz_graph(num_people, num_friends, chance/100)
    return Graph(G)


def caveman_graph(n, p_in, p_out):
    if p_out <= 0 or p_in <= 0:
        raise ValueError('Chance must be positive')
    connected = False
    while not connected:
        a = np.log2(n)
        a = np.arange(1, a + 1)
        a = np.log2(1/a)
        a = a * n / a.sum()
        a = np.round(a, 0).astype(int)
        G = nx.random_partition_graph(a, p_in/100, p_out/100)
        connected = nx.is_connected(G)
    return Graph(G)

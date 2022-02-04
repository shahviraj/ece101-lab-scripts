from nis import cat
import networkx as nx
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML


class Graph():
    def __init__(self):
        self.graph = nx.Graph()
        self.pos = None
        self.weighted = False
        self.path = None

    def add_node(self, name, color=None) -> None:
        if name is not str:
            name = str(name)
        if color:
            self.graph.add_node(name, color=color)
        else:
            if name == 'S':
                self.graph.add_node(name, color='green')
            elif name == 'D':
                self.graph.add_node(name, color='purple')
            else:
                self.graph.add_node(name, color='black')

    def add_edge(self, u, v, weight=None) -> None:
        if u is not str:
            u = str(u)
        if v is not str:
            v = str(v)
        if weight:
            self.graph.add_edge(u, v, weight=weight, antiweight=1/weight)
            self.weighted = True
        else:
            self.graph.add_edge(u, v)

    def draw_graph(self, figsize=(6, 4)) -> None:
        plt.figure(figsize=figsize)
        if self.weighted:
            self.pos = nx.spring_layout(
                self.graph, weight=None, iterations=100)
        else:
            self.pos = nx.spring_layout(
                self.graph, weight='antiweight', iterations=100)

        def get_color(node):
            try:
                return self.graph.nodes[node]['color']
            except KeyError:
                return 'black'

        nx.draw(self.graph, self.pos, node_color=[get_color(
            node) for node in self.graph.nodes], with_labels=True, font_weight='bold', font_color='w')
        if self.weighted:
            labels = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw_networkx_edge_labels(
                self.graph, self.pos, edge_labels=labels)

    def draw_shortest_path(self) -> None:
        try:
            if self.weighted:
                self.path = nx.shortest_path(
                    self.graph, source='S', target='D', weight='weight')
            else:
                self.path = nx.shortest_path(
                    self.graph, source='S', target='D')
            path_edges = list(zip(self.path, self.path[1:]))
            nx.draw_networkx_edges(
                self.graph, self.pos, edgelist=path_edges, edge_color='purple', width=10, alpha=0.2)
        except nx.NetworkXNoPath:
            display(HTML('<h2 style="color:#B00020;">No path from S to D</h2>'))
    
    def show_all_paths(self) -> None:
        paths = [path for path in nx.all_simple_paths(self.graph, source='S', target='D')]
        paths = sorted(paths, key=lambda x: len(x))
        for i, path in enumerate(paths):
            plt.figure(figsize=(3, 2))
            path_edges = list(zip(path, path[1:]))
            nx.draw(self.graph, self.pos, node_color=[self.graph.nodes[node]['color'] for node in self.graph.nodes], with_labels=True, font_weight='bold', font_color='w')
            nx.draw_networkx_edges(
                self.graph, self.pos, edgelist=path_edges, edge_color='purple', width=10, alpha=0.2)
            plt.show()
    
    def remove_edge(self, u, v) -> None:
        if u is not str:
            u = str(u)
        if v is not str:
            v = str(v)
        try:
            self.graph.remove_edge(u, v)
        except nx.NetworkXError:
            print("There was no edge between {} and {}".format(u, v))


def get_hierarchial_graph():
    G = Graph()
    G.add_node('S', color='orange')
    G.add_node('D', color='purple')
    for i in range(1,10):
        G.add_node(i, color='orange')
    for i in range(10, 42):
        G.add_node(i)
    G.add_edge('S', 2)
    G.add_edge(1, 2)
    G.add_edge(2, 5)
    G.add_edge(3, 4)
    G.add_edge(6, 4)
    G.add_edge(4, 5)
    G.add_edge(11, 5)
    G.add_edge(5, 9)
    G.add_edge(9, 8)
    G.add_edge(9, 7)
    G.add_edge(9, 10)
    G.add_edge(10, 26)
    G.add_edge(10, 11)
    G.add_edge(12, 11)
    G.add_edge(13, 11)
    G.add_edge(14, 11)
    G.add_edge(14, 17)
    G.add_edge(16, 17)
    G.add_edge(15, 17)
    G.add_edge(14, 19)
    G.add_edge(18, 19)
    G.add_edge(20, 19)
    G.add_edge(21, 19)
    G.add_edge(21, 22)
    G.add_edge(23, 22)
    G.add_edge(23, 24)
    G.add_edge(23, 25)
    G.add_edge(23, 26)
    G.add_edge(26, 27)
    G.add_edge(28, 27)
    G.add_edge(29, 27)
    G.add_edge(26, 30)
    G.add_edge(31, 30)
    G.add_edge(32, 30)
    G.add_edge(30, 'D')
    G.add_edge(21, 33)
    G.add_edge(34, 33)
    G.add_edge(34, 35)
    G.add_edge(34, 36)
    G.add_edge(33, 37)
    G.add_edge(38, 37)
    G.add_edge(39, 37)
    G.add_edge(33, 40)
    G.add_edge(41, 40)
    G.add_edge(42, 40)

    return G
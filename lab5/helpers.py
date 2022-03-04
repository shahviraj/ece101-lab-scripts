
from IPython.display import clear_output
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import lxml
import requests
import networkx as nx
import holoviews as hv
from holoviews.operation.datashader import bundle_graph
import pandas as pd
import pickle as pkl
hv.extension('bokeh')


class Spider():
    def fix_url(self, url):
        if not url.startswith('http'):
            url = 'https://' + url
        # return None if mailto: or tel:
        if url.startswith('mailto:') or url.startswith('tel:'):
            return None
        return url

    def get_domain(self, url):
        if url is None:
            return None
        return urlparse(url).netloc

    def crawl_and_get_links(self, url):
        url = self.fix_url(url)
        try:
            f = requests.get(url, headers=self.headers, timeout=1)
        except:
            return []
        soup = BeautifulSoup(f.text, 'lxml')
        links = []
        for link in soup.find_all('a'):
            links.append(link.get('href'))
        # print(links)
        # remove all relative links
        links = [
            link for link in links if link is not None and not link.startswith('/')]
        # remove all links that are in the same domain
        links = [link for link in links if self.get_domain(
            self.fix_url(link)) != self.get_domain(url)]
        # print(links)
        return links

    def is_social(self, url):
        social_domains = [
            'facebook.com',
            'twitter.com',
            'instagram.com',
            'linkedin.com',
            'pinterest.com',
            'tumblr.com',
            'youtube.com',
            'reddit.com',
            'flickr.com',
            'oculus.com',
            'messenger.com',
            'google.com',
        ]

        for social_domain in social_domains:
            if social_domain in url:
                return True
        return False

    def clean_graph(self, web):
        node_list = list(web.nodes)
        prefix_blocklist = [
            'javascript:',
            'mailto:',
            'tel:',
            'whatsapp:',
            'skype:',
            'sms:',
            '#',
        ]
        for prefix in prefix_blocklist:
            for node in node_list:
                if node.startswith(prefix):
                    web.remove_node(node)

        for node in node_list:
            if node == "":
                web.remove_node(node)
            else:
                if self.is_social(node):
                    web.remove_node(node)

        return web

    def __init__(self, start_url: str, max_dist: int):
        self.start_url = start_url
        self.max_depth = max_dist - 1
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'}
        self.web = nx.DiGraph()
        self.crawled = set()
        self.to_crawl = [(self.fix_url(start_url), 0)]
        self.web_hv = None

    def crawl(self):
        while self.to_crawl:
            url, depth = self.to_crawl.pop(0)
            node = self.get_domain(self.fix_url(url))
            if node is None:
                continue
            self.web.add_node(node)
            if node in self.crawled:
                continue
            clear_output(wait=True)
            print("{}/{}: Currently crawling {} at distance {}".format(len(self.crawled),
                  len(self.crawled) + len(self.to_crawl), url, depth+1))
            links = self.crawl_and_get_links(url)
            self.crawled = self.crawled.union({node})
            for link in links:
                domain = self.get_domain(self.fix_url(link))
                if domain is not None:
                    self.web.add_node(domain)
                    self.web.add_edge(node, domain)
                    if domain not in self.crawled and depth < self.max_depth:
                        if not self.is_social(link):
                            self.to_crawl.append((link, depth+1))
        clear_output()
        print("Done crawling!")
        print("Found {} nodes & {} edges".format(
            len(self.web), len(self.web.edges)))
        self.web = self.clean_graph(self.web)
        # return self.web

    def show_graph(self):
        pos = nx.spring_layout(self.web, k=0.8, iterations=20)
        self.web_hv = hv.Graph.from_networkx(self.web, pos).opts(
            directed=True,
            node_size=10,
            arrowhead_length=0.015,
            width=800, height=600,
            aspect='equal',
            xaxis=None,
            yaxis=None,
        )
        return self.web_hv

    def show_organic_graph(self):
        if self.web_hv is None:
            _ = self.show_graph()
        organic_graph = bundle_graph(self.web_hv)
        return organic_graph

    def show_pagerank_list(self):
        pagerank_list = nx.pagerank(self.web)
        print("Top 10 pagerank nodes:")
        for node, pagerank in sorted(pagerank_list.items(), key=lambda x: x[1], reverse=True)[:10]:
            print("  {}".format(node))
        print()
        print("Bottom 10 pagerank nodes:")
        for node, pagerank in sorted(pagerank_list.items(), key=lambda x: x[1], reverse=False)[:10]:
            print("  {}".format(node))

def spotify_feature_space(x_axis, y_axis):
    x_axis = x_axis.lower()
    y_axis = y_axis.lower()
    spotify_data_df = pkl.load(open('spotify_data.pkl', 'rb'))
    if x_axis not in spotify_data_df.columns:
        raise ValueError("x_axis is not valid")
    if y_axis not in spotify_data_df.columns:
        raise ValueError("y_axis is not valid")
    
    return hv.Scatter(spotify_data_df, kdims=[x_axis], vdims=[y_axis, "name"]).opts(
        title="{} vs {} for Classical, Electronic, and Rap music".format(x_axis, y_axis),
        width=800,
        height=500,
        xlabel=x_axis,
        ylabel=y_axis,
        tools=['hover'],
        # increase dot size and add edge color
        size=10,
        alpha=0.5,
    )
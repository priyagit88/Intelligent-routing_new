import networkx as nx

class OSPFRouting:
    """
    Simulates OSPF (Open Shortest Path First) routing.
    - Link-state protocol
    - Uses Dijkstra's algorithm with link costs
    - Considers link weights (latency/cost)
    - Faster convergence than RIP
    """
    def __init__(self, graph):
        self.graph = graph
        self.link_state_db = {}  # Link State Database
        self._build_link_state_db()
    
    def _build_link_state_db(self):
        """Build link-state database from graph."""
        for node in self.graph.nodes():
            self.link_state_db[node] = {}
            for neighbor in self.graph.neighbors(node):
                # Store link cost (weight)
                cost = self.graph[node][neighbor].get('weight', 1)
                self.link_state_db[node][neighbor] = cost
    
    def update_link_state(self, u, v, cost):
        """Update link state when topology changes."""
        if u in self.link_state_db:
            self.link_state_db[u][v] = cost
    
    def find_path(self, source, target):
        """
        Find shortest path using Dijkstra's algorithm (OSPF).
        Uses link costs from link-state database.
        """
        try:
            # OSPF uses Dijkstra with link costs (weights)
            return nx.shortest_path(self.graph, source=source, target=target, weight='weight')
        except nx.NetworkXNoPath:
            return None

import simpy
import networkx as nx
import random
from utils import setup_logger

logger = setup_logger()

class NetworkSimulation:
    def __init__(self, env):
        self.env = env
        self.graph = nx.DiGraph()
        self.nodes = []

    def create_topology(self, num_nodes=10, connectivity=0.3):
        """Randomly generates a network topology"""
        self.graph = nx.gnp_random_graph(num_nodes, connectivity, directed=True)
        # Ensure weights (latency) exist
        for (u, v) in self.graph.edges():
            self.graph.edges[u, v]['weight'] = random.randint(1, 10) # ms latency
            self.graph.edges[u, v]['capacity'] = random.randint(10, 100) # Mbps
        
        # Default "ground-truth" reliability for nodes (used by simulate_packet).
        # Algorithms should NOT influence this value; only the trust model should learn it.
        for n in self.graph.nodes():
            if 'reliability' not in self.graph.nodes[n]:
                self.graph.nodes[n]['reliability'] = 0.99
        
        self.nodes = list(self.graph.nodes())
        logger.info(f"Topology created with {num_nodes} nodes and {len(self.graph.edges())} edges")

    def degrade_node(self, node_id, duration=50):
        """Simulates a node having performance issues or being compromised"""
        logger.warning(f"Node {node_id} is degrading...")

    def update_congestion(self):
        """
        Periodically updates edge weights to simulate congestion.
        Run this as a SimPy process.
        """
        while True:
            yield self.env.timeout(5) # Every 5 ticks
            
            # Select random edges to congest
            if self.graph.edges:
                u, v = random.choice(list(self.graph.edges()))
                current_weight = self.graph[u][v]['weight']
                
                # Randomly spike latency or recover
                if random.random() < 0.3:
                    # Congestion Spike (Cap at 200ms)
                    new_weight = min(200, current_weight + random.randint(10, 50))
                    logger.debug(f"Congestion on link {u}->{v}: Weight {current_weight} -> {new_weight}")
                else:
                    # Recovery (decay back to baseline 1-10)
                    new_weight = max(random.randint(1, 10), current_weight - 10)
                
                self.graph[u][v]['weight'] = new_weight

    def add_node(self):
        """Adds a new node to the graph with a unique ID"""
        if not self.nodes:
            new_id = 0
        else:
            new_id = max(self.nodes) + 1
        
        self.graph.add_node(new_id, reliability=1.0)
        self.nodes.append(new_id)
        logger.info(f"Added new node: {new_id}")
        return new_id

    def add_edge(self, u, v, latency=10, capacity=50):
        """Adds a directed edge between two nodes"""
        if u not in self.nodes or v not in self.nodes:
            return False
        
        self.graph.add_edge(u, v)
        self.graph.edges[u, v]['weight'] = latency
        self.graph.edges[u, v]['capacity'] = capacity
        logger.info(f"Added edge {u}->{v} (Lat: {latency}, Cap: {capacity})")
        return True

    def simulate_packet(self, path, trust_model=None, priority=0):
        """
        Simulates a packet traversing a path.
        priority: 0 (Normal/Data), 1 (High/Voice)
        """
        if not path:
            return False
            
        success = True
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            
            # 1. Congestion Check
            # If link is highly congested (weight > 50), Low Priority packets might get dropped
            weight = self.graph[u][v].get('weight', 1)
            if weight > 50 and priority == 0:
                # 30% drop chance for Data during congestion
                if random.random() < 0.3:
                    logger.debug(f"Packet (Low Prio) dropped due to congestion on {u}->{v}")
                    return False

            # 2. Ground-truth forwarding behavior (independent of the trust model)
            # IMPORTANT: Trust is a *belief* used for routing decisions; it must not
            # influence physical packet drops (otherwise we create a circular feedback loop).
            reliability = self.graph.nodes[v].get('reliability', 1.0)
            dropped = random.random() > reliability
            success = not dropped

            # 3. Update trust model with observation + optional metrics
            if trust_model is not None:
                bandwidth = self.graph[u][v].get('capacity', None)
                trust_model.update_trust(
                    v,
                    success,
                    priority=priority,
                    delay=weight,
                    bandwidth=bandwidth,
                )

            if dropped:
                logger.debug(f"Packet dropped at node {v} (Reliability: {reliability:.2f})")
                break
                
        return success

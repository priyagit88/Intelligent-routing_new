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

    def simulate_packet(self, path, trust_model, priority=0):
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

            # ... Rest of logic usually handled by "realistic_simulate_packet" patch
            # But the base method also needs to be robust if called directly

            u, v = path[i], path[i+1]
            
            # Simple stochastic model for packet loss
            # Lower trust = higher chance of drop
            trust = trust_model.get_trust(v)
            drop_prob = (1 - trust) * 0.5 # Max 50% drop rate for 0 trust
            
            if random.random() < drop_prob:
                logger.info(f"Packet dropped at node {v} (Trust: {trust:.2f})")
                success = False
                trust_model.update_trust(v, False) # Penalize
                break
            else:
                trust_model.update_trust(v, True) # Reward
                
        return success

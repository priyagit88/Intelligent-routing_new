
import simpy
import random
from network_sim import NetworkSimulation
from routing import ShortestPathRouting

def test_standard():
    env = simpy.Environment()
    net_sim = NetworkSimulation(env)
    
    # Deterministic Topology
    random.seed(42) 
    net_sim.create_topology(num_nodes=20, connectivity=0.3)
    
    print("Graph nodes:", net_sim.nodes)
    print("Graph edges:", len(net_sim.graph.edges))
    
    routing = ShortestPathRouting(net_sim.graph)
    
    src, dst = net_sim.nodes[0], net_sim.nodes[5]
    print(f"Finding path from {src} to {dst}")
    
    path = routing.find_path(src, dst)
    print("Path found:", path)
    
    if path:
        print("Test Passed")
    else:
        print("No path found (might be disconnected if connectivity is low)")

if __name__ == "__main__":
    test_standard()

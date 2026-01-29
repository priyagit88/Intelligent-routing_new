import simpy
from network_sim import NetworkSimulation
from rl_agent import QLearningAgent
from advanced_agents import GNNRLAgent

def test_dynamic_topology():
    print("Initializing Network Simulation...")
    env = simpy.Environment()
    net_sim = NetworkSimulation(env)
    net_sim.create_topology(num_nodes=5, connectivity=0.5)
    
    initial_nodes = len(net_sim.graph.nodes)
    print(f"Initial nodes: {initial_nodes}")
    
    # 1. Test Add Node
    print("\nTesting Add Node...")
    new_id = net_sim.add_node()
    assert new_id == initial_nodes, f"Expected node ID {initial_nodes}, got {new_id}"
    assert len(net_sim.graph.nodes) == initial_nodes + 1, "Node count did not increase"
    print("Add Node: SUCCESS")
    
    # 2. Test Add Edge
    print("\nTesting Add Edge...")
    success = net_sim.add_edge(new_id, 0)
    assert success, "Failed to add edge"
    assert net_sim.graph.has_edge(new_id, 0), "Edge not found in graph"
    print("Add Edge: SUCCESS")
    
    # 3. Test Q-Learning Update
    print("\nTesting Q-Learning Agent Update...")
    ql_agent = QLearningAgent(list(net_sim.graph.nodes))
    # Add another node to test update
    newer_id = net_sim.add_node()
    ql_agent.add_node(newer_id)
    assert newer_id in ql_agent.q_table, "New node not found in Q-table"
    print("Q-Learning Update: SUCCESS")
    
    # 4. Test GNN Agent Update
    print("\nTesting GNN Agent Update...")
    # Initialize GNN with current graph
    gnn_agent = GNNRLAgent(net_sim.graph, list(net_sim.graph.nodes))
    # Add another node
    even_newer_id = net_sim.add_node()
    net_sim.add_edge(even_newer_id, 0)
    
    # Update GNN
    gnn_agent.update_graph(net_sim.graph, list(net_sim.graph.nodes))
    
    # Check dimensions
    assert gnn_agent.num_nodes == len(net_sim.graph.nodes), "GNN node count mismatch"
    # Check if forward pass works (requires torch if available)
    try:
        import torch
        emb = gnn_agent._forward(0)
        assert emb.shape[0] == len(net_sim.graph.nodes), "Embedding size mismatch"
        print("GNN Update: SUCCESS")
    except ImportError:
        print("PyTorch not available, skipping GNN forward pass test.")
    
    print("\nAll Tests Passed!")

if __name__ == "__main__":
    test_dynamic_topology()

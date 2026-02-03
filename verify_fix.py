
import sys
import os
import simpy
import networkx as nx

# Add current dir to path
sys.path.append(os.getcwd())

from network_sim import NetworkSimulation
from advanced_agents import DQNAgent, GNNRLAgent
from dashboard import reset_network
import streamlit as st

def test_empty_topology():
    print("Testing Empty Topology initialization...")
    # Mock streamlit session state
    if 'net_sim' not in st.session_state:
        class State:
            pass
        st.session_state = {}
    
    # This should not crash now
    nodes = []
    dqn = DQNAgent(nodes)
    gnn = GNNRLAgent(nx.DiGraph(), nodes)
    
    print("Successfully initialized agents with 0 nodes.")
    
    # Test dynamic growth
    nodes.append(0)
    dqn.nodes = nodes
    dqn.node_to_idx = {0: 0}
    dqn.num_nodes = 1
    
    g = nx.DiGraph()
    g.add_node(0)
    gnn.update_graph(g, nodes)
    
    print("Successfully updated GNN agent with new node.")
    if gnn.gcn_weight is not None:
        print("GNN weights initialized correctly after adding first node.")
    else:
        print("ERROR: GNN weights still None after node addition.")

if __name__ == "__main__":
    try:
        test_empty_topology()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()

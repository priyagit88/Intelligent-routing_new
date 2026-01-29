import streamlit as st
import simpy
import networkx as nx
import random
import matplotlib.pyplot as plt
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import IntelligentRouting, ShortestPathRouting, RLRouting, RIPRouting, TrustAwareRLRouting
from rl_agent import QLearningAgent, TrustQLearningAgent
from visualization import visualize_network
import pandas as pd

from advanced_agents import DQNAgent, GNNRLAgent

# Initialize Session State
if 'env' not in st.session_state:
    st.session_state.env = simpy.Environment()
    st.session_state.net_sim = NetworkSimulation(st.session_state.env)
    st.session_state.net_sim.create_topology(num_nodes=15, connectivity=0.2)
    
    # Pre-seed some bad nodes for demo purposes
    st.session_state.net_sim.graph.nodes[3]['reliability'] = 0.5
    st.session_state.net_sim.graph.nodes[7]['reliability'] = 0.6
    
    st.session_state.trust_model = TrustModel()
    
    # Initialize Agents
    nodes = list(st.session_state.net_sim.graph.nodes)
    st.session_state.rl_agent = QLearningAgent(nodes)
    st.session_state.dqn_agent = DQNAgent(nodes)
    st.session_state.trust_rl_agent = TrustQLearningAgent(nodes, trust_impact=2.0)
    st.session_state.gnn_agent = GNNRLAgent(st.session_state.net_sim.graph, nodes)
    
    # Default Routing
    st.session_state.routing_algo_name = "Intelligent (Trust)"
    st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    
    st.session_state.packet_stats = []
    st.session_state.time = 0

# Sidebar Controls
st.sidebar.header("Simulation Controls")

# Algorithm Selection
algo_option = st.sidebar.selectbox(
    "Routing Protocol", 
    ["Standard OSPF (Latency)", "RIP (Hop Count)", "Intelligent (Trust)", "Q-Learning (AI)", "Trust-Aware Q-Routing (New)", "DQN (Deep RL)", "GNN-RL (Graph AI)"]
)

# Handle Algorithm Change
if algo_option != st.session_state.routing_algo_name:
    st.session_state.routing_algo_name = algo_option
    if algo_option == "Standard OSPF (Latency)":
        st.session_state.routing = ShortestPathRouting(st.session_state.net_sim.graph)
    elif algo_option == "RIP (Hop Count)":
        st.session_state.routing = RIPRouting(st.session_state.net_sim.graph)
    elif algo_option == "Intelligent (Trust)":
        st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    elif algo_option == "Q-Learning (AI)":
        st.session_state.routing = RLRouting(st.session_state.net_sim.graph, st.session_state.rl_agent)
    elif algo_option == "Trust-Aware Q-Routing (New)":
        st.session_state.routing = TrustAwareRLRouting(st.session_state.net_sim.graph, st.session_state.trust_rl_agent, st.session_state.trust_model)
    elif algo_option == "DQN (Deep RL)":
        st.session_state.routing = RLRouting(st.session_state.net_sim.graph, st.session_state.dqn_agent)
    elif algo_option == "GNN-RL (Graph AI)":
        st.session_state.routing = RLRouting(st.session_state.net_sim.graph, st.session_state.gnn_agent)
    
    st.toast(f"Switched to {algo_option}")

# Node Reliability Controls
st.sidebar.subheader("Node Reliability")
nodes = list(st.session_state.net_sim.graph.nodes)
selected_node = st.sidebar.selectbox("Select Node to Modify", nodes)
reliability = st.sidebar.slider(f"Reliability of Node {selected_node}", 0.0, 1.0, 
                                st.session_state.net_sim.graph.nodes[selected_node].get('reliability', 1.0))

if st.sidebar.button("Update Node"):
    st.session_state.net_sim.graph.nodes[selected_node]['reliability'] = reliability
    st.success(f"Node {selected_node} reliability set to {reliability}")

# Topology Management
with st.sidebar.expander("Topology Management"):
    # Add Node
    if st.button("➕ Add New Node"):
        new_id = st.session_state.net_sim.add_node()
        st.success(f"Node {new_id} added!")
        
        # Update Agents
        # Update persistent agents dict
        if "Q-Learning" in st.session_state.agents:
            st.session_state.agents["Q-Learning"].add_node(new_id)
        if "Trust-Aware Q-Routing" in st.session_state.agents:
            st.session_state.agents["Trust-Aware Q-Routing"].add_node(new_id)

        # Update active standalone agents if they exist
        if 'rl_agent' in st.session_state:
            st.session_state.rl_agent.add_node(new_id)
        if 'trust_rl_agent' in st.session_state:
            st.session_state.trust_rl_agent.add_node(new_id)
        
        # GNN Agent needs graph rebuild
        if "GNN-RL (Graph AI)" in st.session_state.agents:
            st.session_state.agents["GNN-RL (Graph AI)"].update_graph(
                st.session_state.net_sim.graph, 
                list(st.session_state.net_sim.graph.nodes)
            )
            
        # DQN Agent needs Reset
        st.warning("DQN Agent RESET due to topology change.")
        st.session_state.agents["DQN (Deep RL)"] = DQNAgent(list(st.session_state.net_sim.graph.nodes))
        st.rerun()

    st.write("---")
    
    # Add Connection
    st.write("**Add Connection**")
    node_options = list(st.session_state.net_sim.graph.nodes)
    u_node = st.selectbox("Source", node_options, key="src_node")
    v_node = st.selectbox("Destination", node_options, key="dst_node")
    
    new_latency = st.number_input("Latency (ms)", 1, 100, 10)
    new_capacity = st.number_input("Capacity (Mbps)", 10, 1000, 100)
    
    if st.button("🔗 Connect"):
        if u_node == v_node:
            st.error("Source and Destination cannot be the same.")
        elif st.session_state.net_sim.graph.has_edge(u_node, v_node):
            st.error("Edge already exists.")
        else:
            st.session_state.net_sim.add_edge(u_node, v_node, new_latency, new_capacity)
            st.success(f"Connected {u_node} -> {v_node}")
            
            # GNN needs update for new edges too (Adjacency Matrix)
            if "GNN-RL (Graph AI)" in st.session_state.agents:
                 st.session_state.agents["GNN-RL (Graph AI)"].update_graph(
                    st.session_state.net_sim.graph, 
                    list(st.session_state.net_sim.graph.nodes)
                )

# Helper to instantiate algo (cached or new)
def get_routing_algo(name, graph, trust_model, existing_agents=None):
    if name == "Shortest Path":
        return ShortestPathRouting(graph)
    elif name == "Intelligent Routing":
        return IntelligentRouting(graph, trust_model)
    elif name == "Q-Learning":
        if existing_agents and "Q-Learning" in existing_agents:
            return RLRouting(graph, existing_agents["Q-Learning"])
        return RLRouting(graph, QLearningAgent(list(graph.nodes)))
    elif name == "Trust-Aware Q-Routing":
        if existing_agents and "Trust-Aware Q-Routing" in existing_agents:
             return TrustAwareRLRouting(graph, existing_agents["Trust-Aware Q-Routing"], trust_model)
        return TrustAwareRLRouting(graph, TrustQLearningAgent(list(graph.nodes)), trust_model)
    elif name == "DQN (Deep RL)":
        if existing_agents and "DQN (Deep RL)" in existing_agents:
             return RLRouting(graph, existing_agents["DQN (Deep RL)"])
        return RLRouting(graph, DQNAgent(list(graph.nodes)))
    elif name == "GNN-RL (Graph AI)":
        if existing_agents and "GNN-RL (Graph AI)" in existing_agents:
            return RLRouting(graph, existing_agents["GNN-RL (Graph AI)"])
        return RLRouting(graph, GNNRLAgent(graph, list(graph.nodes)))
    return ShortestPathRouting(graph)

# Store agents in session state to persist training
# Store agents in session state to persist training
if 'agents' not in st.session_state:
    st.session_state.agents = {
        "Q-Learning": QLearningAgent(list(st.session_state.net_sim.graph.nodes)),
        "Trust-Aware Q-Routing": TrustQLearningAgent(list(st.session_state.net_sim.graph.nodes)),
        "DQN (Deep RL)": DQNAgent(list(st.session_state.net_sim.graph.nodes)),
        "GNN-RL (Graph AI)": GNNRLAgent(st.session_state.net_sim.graph, list(st.session_state.net_sim.graph.nodes))
    }

# Sync current selection with session agents
if st.session_state.routing_algo_name in st.session_state.agents:
    st.session_state.routing = RLRouting(st.session_state.net_sim.graph, st.session_state.agents[st.session_state.routing_algo_name])

# Traffic Scheduling for Consistency
if 'traffic_schedule' not in st.session_state:
    st.session_state.traffic_schedule = []

def get_traffic_pair(index, nodes, graph):
    """
    Returns the (src, dst) pair for the given index.
    Generates and caches it if it doesn't exist.
    """
    # Extend schedule if needed
    while len(st.session_state.traffic_schedule) <= index:
        # Generate valid pair
        attempts = 0
        while True:
            src, dst = random.sample(nodes, 2)
            if nx.has_path(graph, src, dst):
                st.session_state.traffic_schedule.append((src, dst))
                break
            attempts += 1
            if attempts > 20: 
                largest_cc = max(nx.weakly_connected_components(graph), key=len)
                src, dst = random.sample(list(largest_cc), 2)
                st.session_state.traffic_schedule.append((src, dst))
                break
    
    return st.session_state.traffic_schedule[index]

# Simulation Step
st.sidebar.subheader("Run Simulation")
cols = st.sidebar.columns(2)
train_mode = cols[0].button("🚀 Train Agent")
run_mode = cols[1].button("▶️ Run Step")
steps = st.sidebar.number_input("Packets", min_value=1, value=50)

if st.sidebar.button("🔄 Regenerate Traffic"):
    st.session_state.traffic_schedule = []
    st.success("Traffic schedule cleared. New random packets will be generated.")

def run_simulation_batch(num_packets, training=False):
    env = st.session_state.env
    net_sim = st.session_state.net_sim
    routing = st.session_state.routing
    trust_model = st.session_state.trust_model
    
    # Validation: Training only makes sense for RL agents
    if training and not hasattr(routing, 'agent'):
        st.warning("Training is only available for AI/RL algorithms (Q-Learning, DQN, GNN).")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    rewards = []
    
    # Epsilon Decay Setup
    start_epsilon = routing.agent.epsilon if hasattr(routing, 'agent') else 0.1
    end_epsilon = 0.01
    decay_rate = 0.995
    
    try:
        # Determine start index for traffic
        # If training, we might roam freely? 
        # But user asked for "same set of source and destination".
        # This implies standard Testing. 
        # For Training, we usually want infinite random new samples.
        # But for "Run Step" (Evaluation), we want consistency.
        
        # Let's use persistent schedule for Evaluation (Run Step), 
        # and maybe random/extend for Training to avoid overfitting to just 50 packets?
        # A compromise: All traffic comes from the infinite schedule stream.
        # But we need to know WHERE to start. 
        # If we always start at 0, the agent overfits to packet #1 being A->B.
        # Ideally, we shuffle? 
        # "Use the same set" usually means "Test Set".
        
        # Strategy:
        # If "Run Step" (Evaluation): Use indices 0 to num_packets-1 (Fixed Test Set)
        # If "Train Agent": Use random indices or extend indefinitely?
        # User goal: "Fair Comparison". This implies Evaluation.
        
        start_idx = 0 if not training else random.randint(0, 10000) # Randomize for training
        
        for i in range(num_packets):
            # Get consistent pair
            # For evaluation, we always start from 0 to steps
            idx_to_use = i if not training else (start_idx + i)
            src, dst = get_traffic_pair(idx_to_use, nodes, net_sim.graph)

            # Epsilon Decay
            if training and hasattr(routing, 'agent'):
                routing.agent.epsilon = max(end_epsilon, routing.agent.epsilon * decay_rate)
            
            path = routing.find_path(src, dst)
            status = "No Path"
            success = True
            
            if path:
                # Simulate Traversal
                for j in range(len(path) - 1):
                     u, v = path[j], path[j+1]
                     rel = net_sim.graph.nodes[v].get('reliability', 1.0)
                     if random.random() > rel:
                         success = False
                         trust_model.update_trust(v, False)
                         break
                     else:
                         trust_model.update_trust(v, True)
                
                # Feedback / Learning
                if hasattr(routing, 'agent'): # Check if it's an RL routing
                    # Reward Shaping: 
                    # Success: +10 - (0.1 * hops) to encourage shortest valid path
                    # Failure: -10
                    reward = (10 - 0.1 * len(path)) if success else -10
                    rewards.append(reward)
                    
                    # Learn
                    for j in range(len(path) - 1):
                        u_node = path[j]
                        v_node = path[j+1]
                        next_neighbors = list(net_sim.graph.neighbors(v_node)) if v_node in net_sim.graph else []
                        
                        done = (v_node == dst) or (not success and j == len(path)-2)
                        
                        # Compatible Learn Call
                        routing.agent.learn(u_node, v_node, reward, v_node, next_neighbors, target_node=dst)
            else:
                 success = False
                 rewards.append(-20) # Big penalty for no path

            if success: success_count += 1
            
            # Update Statistics for Visualization (only if not training or every 10 steps in training)
            if not training or i % 10 == 0:
                 st.session_state.packet_stats.append({
                    "Time": st.session_state.time,
                    "Algorithm": st.session_state.routing_algo_name,
                    "Source": src, 
                    "Dest": dst,
                    "Status": "Success" if success else "Dropped" if path else "No Path"
                })
                 st.session_state.time += 1
            
            # Update Progress
            if i % 5 == 0:
                progress_bar.progress((i + 1) / num_packets)
                if training:
                    status_text.text(f"Training: {i+1}/{num_packets} | Eps: {routing.agent.epsilon:.3f} | Last Reward: {rewards[-1]:.1f}")
                
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    progress_bar.empty()
    status_text.empty()
    
    if training:
        st.success(f"Training Complete! Valid PDR: {(success_count/num_packets)*100:.1f}% | Final Epsilon: {routing.agent.epsilon:.3f}")
        # Optionally restore epsilon to low value for testing instead of lowest 
        # But usually we keep it low.

if train_mode:
    run_simulation_batch(steps, training=True)
elif run_mode:
    run_simulation_batch(steps, training=False)

# Layout
col1, col2 = st.columns([2, 1])

# Main Interface Tabs
tab1, tab2 = st.tabs(["🔴 Live Simulation", "📊 Benchmark Comparison"])

with tab1:
    # Visualize Controls
    view_mode = st.radio("Visualization Mode", ["Ground Truth (Reliability)", "Agent Perception (Trust Score)"])

    with col1:
        st.subheader(f"Network Topology ({view_mode})")
        
        # If Ground Truth, pass trust_model=None so it uses 'reliability' attribute
        tm_to_use = st.session_state.trust_model if view_mode == "Agent Perception (Trust Score)" else None
        
        # Visualize
        fig = visualize_network(st.session_state.net_sim.graph, 
                                trust_model=tm_to_use, 
                                return_fig=True)
        st.pyplot(fig)

    with col2:
        st.subheader("Recent Traffic")
        if st.session_state.packet_stats:
            df = pd.DataFrame(st.session_state.packet_stats[-10:]) # Last 10
            st.dataframe(df)
            
            # Stats
            all_df = pd.DataFrame(st.session_state.packet_stats)
            pdr = (all_df[all_df['Status'] == 'Success'].shape[0] / all_df.shape[0]) * 100
            st.metric("Packet Delivery Ratio", f"{pdr:.2f}%")
        else:
            st.info("Run simulation to see stats.")

    # Trust Table
    st.subheader("Node Trust Scores")
    trust_data = {n: st.session_state.trust_model.get_trust(n) for n in nodes}
    st.bar_chart(trust_data)

# Benchmark Tab
with tab2:
    st.subheader("🏆 Protocol Benchmark")
    st.markdown("Compare different algorithms on the **exact same traffic** to measure true performance.")
    
    comp_algos = st.multiselect("Select Algorithms to Compare", 
                                ["Shortest Path", "Intelligent Routing", "Q-Learning", "Trust-Aware Q-Routing", "DQN (Deep RL)", "GNN-RL (Graph AI)"],
                                default=["Shortest Path", "Trust-Aware Q-Routing"])
    
    comp_steps = st.number_input("Comparison Packets", min_value=10, value=100)
    
    if st.button("Run Benchmark"):
        results = []
        my_bar = st.progress(0)
        
        chart_data = {'Packet': [], 'Algorithm': [], 'Cumulative PDR': []}
        total_ops = len(comp_algos) * comp_steps
        current_op = 0
        
        for algo_name in comp_algos:
            bench_routing = get_routing_algo(algo_name, st.session_state.net_sim.graph, st.session_state.trust_model, st.session_state.agents)
            successes = 0
            
            for i in range(comp_steps):
                src, dst = get_traffic_pair(i, list(st.session_state.net_sim.graph.nodes), st.session_state.net_sim.graph)
                path = bench_routing.find_path(src, dst)
                
                success = False
                if path:
                    route_success = True
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j+1]
                        rel = st.session_state.net_sim.graph.nodes[v].get('reliability', 1.0)
                        if random.random() > rel:
                            route_success = False
                            break
                    if route_success: success = True
                
                if success: successes += 1
                
                chart_data['Packet'].append(i + 1)
                chart_data['Algorithm'].append(algo_name)
                current_pdr = (successes / (i + 1)) * 100
                chart_data['Cumulative PDR'].append(current_pdr)
                
                current_op += 1
                if current_op % 5 == 0:
                    my_bar.progress(current_op / total_ops)
            
            final_pdr = (successes / comp_steps) * 100
            results.append({"Algorithm": algo_name, "Final PDR": final_pdr})
        
        my_bar.empty()
        
        st.subheader("📈 Performance Evolution")
        chart_df = pd.DataFrame(chart_data)
        st.line_chart(chart_df, x="Packet", y="Cumulative PDR", color="Algorithm")
        
        st.subheader("📊 Final Results")
        res_df = pd.DataFrame(results).set_index("Algorithm")
        st.bar_chart(res_df)
        st.dataframe(res_df)

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
from security import Adversary
import pandas as pd
import time as time_lib

# Try to import advanced agents (requires PyTorch)
try:
    from advanced_agents import DQNAgent, GNNRLAgent
    ADVANCED_AGENTS_AVAILABLE = True
except (ImportError, OSError) as e:
    ADVANCED_AGENTS_AVAILABLE = False
    print(f"Warning: Advanced agents not available: {e}")
    DQNAgent = None
    GNNRLAgent = None

# Helper to reset simulation state
def reset_network(num_nodes=15, connectivity=0.2, random_init=True):
    st.session_state.env = simpy.Environment()
    st.session_state.net_sim = NetworkSimulation(st.session_state.env)
    
    if random_init:
        st.session_state.net_sim.create_topology(num_nodes=num_nodes, connectivity=connectivity)
        # Pre-seed some bad nodes for demo purposes if default
        if num_nodes == 15 and connectivity == 0.2:
            if 3 in st.session_state.net_sim.graph.nodes: st.session_state.net_sim.graph.nodes[3]['reliability'] = 0.5
            if 7 in st.session_state.net_sim.graph.nodes: st.session_state.net_sim.graph.nodes[7]['reliability'] = 0.6
    else:
        # Start completely empty or as requested
        for _ in range(num_nodes):
            st.session_state.net_sim.add_node()
            
    st.session_state.trust_model = TrustModel(
        initial_trust=1.0,
        decay_factor=0.3,  # Very aggressive decay
        bonus_factor=0.02,
        trust_threshold=0.4
    )
    
    # Initialize Agents
    nodes = list(st.session_state.net_sim.graph.nodes)
    st.session_state.rl_agent = QLearningAgent(nodes, alpha=0.6, gamma=0.8, epsilon=0.9)
    st.session_state.trust_rl_agent = TrustQLearningAgent(
        nodes, 
        alpha=0.6, 
        gamma=0.8, 
        epsilon=0.9,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        trust_impact=3.0  # Stronger trust influence
    )
    
    if ADVANCED_AGENTS_AVAILABLE:
        st.session_state.dqn_agent = DQNAgent(nodes)
        st.session_state.gnn_agent = GNNRLAgent(st.session_state.net_sim.graph, nodes)
    else:
        st.session_state.dqn_agent = None
        st.session_state.gnn_agent = None

    st.session_state.agents = {
        "Q-Learning": st.session_state.rl_agent,
        "Trust-Aware Q-Routing": st.session_state.trust_rl_agent,
    }
    if ADVANCED_AGENTS_AVAILABLE:
        st.session_state.agents["DQN (Deep RL)"] = st.session_state.dqn_agent
        st.session_state.agents["GNN-RL (Graph AI)"] = st.session_state.gnn_agent
    
    # Default Routing
    st.session_state.routing_algo_name = "Intelligent (Trust)"
    st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    st.session_state.packet_stats = []
    st.session_state.time = 0
    st.session_state.traffic_schedule = []
    
    # New: Adversary State
    st.session_state.adversaries = {} # node_id -> Adversary object
    
    # New: Throughput tracking
    st.session_state.total_bits_delivered = 0
    st.session_state.total_latency = 0
    st.session_state.total_hops = 0
    st.session_state.delivered_packets = 0
    st.session_state.start_time = time_lib.time()

# Initialize Session State if not present or outdated
if 'env' not in st.session_state or \
   not hasattr(st.session_state.trust_model, 'diagnose_node') or \
   'delivered_packets' not in st.session_state:
    reset_network()

# Sidebar Setup
st.sidebar.header("🛠️ Network Setup")

# Initialization Section
with st.sidebar.expander("Initialize Topology", expanded=False):
    st.write("Reset the simulation with custom parameters.")
    init_mode = st.radio("Creation Mode", ["Random (Preset)", "Empty (Manual)", "Random (Custom)"])
    
    if init_mode == "Random (Preset)":
        if st.button("Generate Default (15 nodes)"):
            reset_network(num_nodes=15, connectivity=0.2, random_init=True)
            st.rerun()
            
    elif init_mode == "Empty (Manual)":
        if st.button("Start Blank Network"):
            reset_network(num_nodes=0, random_init=False)
            st.rerun()
            
    elif init_mode == "Random (Custom)":
        c_nodes = st.number_input("Number of Nodes", 2, 50, 10)
        c_conn = st.slider("Connectivity Probability", 0.05, 0.5, 0.2)
        if st.button("Generate Custom Network"):
            reset_network(num_nodes=c_nodes, connectivity=c_conn, random_init=True)
            st.rerun()

st.sidebar.write("---")

# Existing Sidebar Controls
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
        if ADVANCED_AGENTS_AVAILABLE:
            st.session_state.routing = RLRouting(st.session_state.net_sim.graph, st.session_state.dqn_agent)
        else:
            st.error("DQN Agent is not available (PyTorch issue). Falling back to Intelligent routing.")
            st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    elif algo_option == "GNN-RL (Graph AI)":
        if ADVANCED_AGENTS_AVAILABLE:
            st.session_state.routing = RLRouting(st.session_state.net_sim.graph, st.session_state.gnn_agent)
        else:
            st.error("GNN Agent is not available (PyTorch issue). Falling back to Intelligent routing.")
            st.session_state.routing = IntelligentRouting(st.session_state.net_sim.graph, st.session_state.trust_model)
    
    st.toast(f"Switched to {algo_option}")

# Node Reliability Controls
nodes = list(st.session_state.net_sim.graph.nodes)
if nodes:
    st.sidebar.subheader("Node Reliability")
    selected_node = st.sidebar.selectbox("Select Node to Modify", nodes)
    if selected_node is not None:
        reliability = st.sidebar.slider(f"Reliability of Node {selected_node}", 0.0, 1.0, 
                                        st.session_state.net_sim.graph.nodes[selected_node].get('reliability', 1.0))

        if st.sidebar.button("Update Node"):
            st.session_state.net_sim.graph.nodes[selected_node]['reliability'] = reliability
            st.success(f"Node {selected_node} reliability set to {reliability}")
else:
    st.sidebar.info("Add nodes to the network to configure reliability.")

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
        if ADVANCED_AGENTS_AVAILABLE:
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
            if ADVANCED_AGENTS_AVAILABLE and "GNN-RL (Graph AI)" in st.session_state.agents:
                 st.session_state.agents["GNN-RL (Graph AI)"].update_graph(
                    st.session_state.net_sim.graph, 
                    list(st.session_state.net_sim.graph.nodes)
                )

    st.write("---")
    
    # Modify Existing Edge
    st.write("**Modify Existing Edge**")
    edges = list(st.session_state.net_sim.graph.edges())
    if edges:
        selected_edge = st.selectbox("Select Edge to Edit", edges, format_func=lambda x: f"{x[0]} -> {x[1]}")
        e_u, e_v = selected_edge
        
        curr_lat = st.session_state.net_sim.graph[e_u][e_v].get('weight', 10)
        curr_cap = st.session_state.net_sim.graph[e_u][e_v].get('capacity', 100)
        
        edit_lat = st.number_input("Edit Latency (ms)", 1, 500, int(curr_lat), key="edit_lat")
        edit_cap = st.number_input("Edit Capacity (Mbps)", 1, 10000, int(curr_cap), key="edit_cap")
        
        if st.button("💾 Update Edge"):
            st.session_state.net_sim.graph[e_u][e_v]['weight'] = edit_lat
            st.session_state.net_sim.graph[e_u][e_v]['capacity'] = edit_cap
            st.success(f"Updated {e_u}->{e_v}")
    else:
        st.info("No connections exist yet.")

# Adversary Configuration
with st.sidebar.expander("🛡️ Adversary Settings"):
    if nodes:
        st.write("Configure malicious nodes in the network.")
        target_adv_node = st.selectbox("Select Node for Adversary", nodes, key="adv_node_sel")
        attack_type = st.selectbox("Attack Type", ["None", "blackhole", "grayhole", "on-off"])
        
        if st.button("Apply Attack Configuration") and target_adv_node is not None:
            if attack_type == "None":
                if target_adv_node in st.session_state.adversaries:
                    del st.session_state.adversaries[target_adv_node]
                st.success(f"Node {target_adv_node} is now healthy.")
            else:
                adv = Adversary(target_adv_node, attack_type)
                # Start behavior process for On-Off
                if attack_type == "on-off":
                    st.session_state.env.process(adv.update_behavior(st.session_state.env))
                st.session_state.adversaries[target_adv_node] = adv
                st.success(f"Node {target_adv_node} is now a {attack_type}!")
    else:
        st.info("No nodes available to configure as adversaries.")

# Helper to instantiate algo (cached or new)
def get_routing_algo(name, graph, trust_model, existing_agents=None):
    # Backward compatible names + explicit protocol labels
    if name in ["Shortest Path", "OSPF (Link State)"]:
        return ShortestPathRouting(graph)
    elif name == "RIP (Hop Count)":
        return RIPRouting(graph)
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
        if not ADVANCED_AGENTS_AVAILABLE:
             return IntelligentRouting(graph, trust_model)
        if existing_agents and "DQN (Deep RL)" in existing_agents:
             return RLRouting(graph, existing_agents["DQN (Deep RL)"])
        return RLRouting(graph, DQNAgent(list(graph.nodes)))
    elif name == "GNN-RL (Graph AI)":
        if not ADVANCED_AGENTS_AVAILABLE:
            return IntelligentRouting(graph, trust_model)
        if existing_agents and "GNN-RL (Graph AI)" in existing_agents:
            return RLRouting(graph, existing_agents["GNN-RL (Graph AI)"])
        return RLRouting(graph, GNNRLAgent(graph, list(graph.nodes)))
    return ShortestPathRouting(graph)

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
        
        # Save original epsilon for restoration
        original_eps = None
        if not training and hasattr(routing, 'agent'):
            original_eps = routing.agent.epsilon
            routing.agent.epsilon = 0.0 # Force pure exploitation for evaluation
        
        rewards = []
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
            
            # Randomly assign priority (10% Voice, 90% Data)
            priority = 1 if random.random() < 0.1 else 0
            packet_type = "voice" if priority == 1 else "data"
            
            if path:
                # Simulate Traversal
                actual_path = [src]
                for j in range(len(path) - 1):
                    u_hop, v_hop = path[j], path[j+1]
                    actual_path.append(v_hop)
                    
                    hop_success = True
                    # Check for Adversary first
                    if v_hop in st.session_state.adversaries:
                        verdict = st.session_state.adversaries[v_hop].process_packet(packet_type)
                        if not verdict:
                            hop_success = False
                    
                    # Normal reliability check
                    if hop_success:
                        rel = net_sim.graph.nodes[v_hop].get('reliability', 1.0)
                        if random.random() > rel:
                            hop_success = False
                    
                    # Update Trust Model
                    trust_model.update_trust(v_hop, hop_success, priority=priority)

                    # Reward Shaping:
                    # +1 for safe hop, -20 for failing at this node
                    reward = 1 if hop_success else -20
                    
                    if training and hasattr(routing, 'agent'):
                        nxt_neighbors = list(net_sim.graph.neighbors(v_hop)) if v_hop in net_sim.graph else []
                        done = (v_hop == dst) or not hop_success
                        if done and hop_success: reward += 10 # Bonus for reaching target
                        
                        routing.agent.learn(u_hop, v_hop, reward, v_hop, nxt_neighbors, target_node=dst, done=done)
                        rewards.append(reward)

                    if not hop_success:
                        success = False
                        if not training: rewards.append(-20) # For stats display
                        break
                
                # Update Throughput on final success
                if success:
                    st.session_state.total_bits_delivered += 8000 
            else:
                 success = False
                 # Penalty for no path
                 if training and hasattr(routing, 'agent'):
                     routing.agent.learn(src, src, -30, src, list(net_sim.graph.neighbors(src)), target_node=dst, done=True)

            if success: 
                success_count += 1
                st.session_state.delivered_packets += 1
                # Track hops
                st.session_state.total_hops += len(path) - 1
                # Calculate latency (sum of edge weights)
                path_latency = sum(net_sim.graph[path[j]][path[j+1]]['weight'] for j in range(len(path)-1))
                st.session_state.total_latency += path_latency
            
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
                    last_rew = rewards[-1] if rewards else 0
                    status_text.text(f"Training: {i+1}/{num_packets} | Eps: {routing.agent.epsilon:.3f} | Last Reward: {last_rew:.1f}")
                
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        # Restore epsilon if it was changed
        if not training and original_eps is not None:
             routing.agent.epsilon = original_eps
    
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
        if len(st.session_state.net_sim.graph.nodes) > 0:
            fig = visualize_network(st.session_state.net_sim.graph, 
                                    trust_model=tm_to_use, 
                                    return_fig=True)
            st.pyplot(fig)
        else:
            st.info("The network is currently empty. Add nodes via the sidebar or generate a random topology.")

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
    if nodes:
        st.subheader("Node Trust Scores")
        trust_data = {n: st.session_state.trust_model.get_trust(n) for n in nodes}
        st.bar_chart(trust_data)

# Security Analysis Tab
with st.tabs(["🛡️ Security Analysis"])[0]:
    st.subheader("Adversarial Diagnosis Engine")
    st.markdown("""
    This engine analyzes node behavior (Packet Loss Patterns, Priority Handling, and Temporal Variance) 
     to identify the type of attack.
    """)
    
    if nodes:
        diag_data = []
        for node in nodes:
            status = st.session_state.trust_model.diagnose_node(node)
            trust = st.session_state.trust_model.get_trust(node)
            s = st.session_state.trust_model.stats.get(node, {})
            
            diag_data.append({
                "Node": node,
                "Trust": round(trust, 2),
                "Data (S/F)": f"{s.get('data_success', 0)}/{s.get('data_fail', 0)}",
                "Voice (S/F)": f"{s.get('voice_success', 0)}/{s.get('voice_fail', 0)}",
                "Diagnosis": status
            })
        
        st.table(pd.DataFrame(diag_data))
    else:
        st.info("No nodes in network to diagnose.")
    
    with st.expander("ℹ️ How does diagnosis work?"):
        st.write("**Blackhole**: Constant dropping across all priorities.")
        st.write("**Grayhole**: Selectively drops Data packets while forwarding Voice (Priority) packets.")
        st.write("**On-Off**: Periodic behavior changes detected through trust history variance.")

# Metrics Analysis Section
st.write("---")
st.subheader("📊 Advanced Performance Metrics")
m_col1, m_col2, m_col3 = st.columns(3)

# 1. Throughput
elapsed = time_lib.time() - st.session_state.start_time
throughput_kbps = (st.session_state.total_bits_delivered / 1000) / elapsed if elapsed > 0 else 0
m_col1.metric("Current Throughput", f"{throughput_kbps:.2f} Kbps", help="Bits successfully delivered per second.")

# 3. Avg Latency
avg_lat = st.session_state.total_latency / st.session_state.delivered_packets if st.session_state.delivered_packets > 0 else 0
m_col2.metric("Avg Latency", f"{avg_lat:.1f} ms", help="Average end-to-end latency for delivered packets.")

# 4. Avg Hop Count
avg_hops = st.session_state.total_hops / st.session_state.delivered_packets if st.session_state.delivered_packets > 0 else 0
m_col3.metric("Avg Hops", f"{avg_hops:.2f}", help="Average number of hops per successful delivery.")

# 5. Malicious Nodes Detected (Moved to separate row or handled below)
detected_count = sum(1 for n in nodes if "Potential" in st.session_state.trust_model.diagnose_node(n)) if nodes else 0
st.sidebar.metric("Attacks Tagged", detected_count)

# Benchmark Tab
with tab2:
    st.subheader("🏆 Protocol Benchmark")
    st.markdown("Compare different algorithms on the **exact same traffic** to measure true performance.")
    
    comp_algos = st.multiselect("Select Algorithms to Compare", 
                                ["OSPF (Link State)", "RIP (Hop Count)", "Intelligent Routing", "Q-Learning", "Trust-Aware Q-Routing", "DQN (Deep RL)", "GNN-RL (Graph AI)"],
                                default=["OSPF (Link State)", "RIP (Hop Count)", "Trust-Aware Q-Routing"])
    
    col_b1, col_b2 = st.columns(2)
    comp_steps = col_b1.number_input("Comparison Packets", min_value=10, value=100)
    auto_train = col_b2.checkbox("Auto-Train AI Algos", value=True, help="Run 500 packets of background training before benchmarking RL algorithms.")
    
    if st.button("Run Benchmark"):
        # Auto-Training Phase
        if auto_train:
            with st.spinner("Pre-training RL algorithms for fair comparison..."):
                for algo_name in comp_algos:
                    if "Q-Learning" in algo_name or "DQN" in algo_name or "GNN" in algo_name:
                        temp_routing = get_routing_algo(algo_name, st.session_state.net_sim.graph, st.session_state.trust_model, st.session_state.agents)
                        # Switch to session routing temporarily to use run_simulation_batch
                        old_routing = st.session_state.routing
                        st.session_state.routing = temp_routing
                        run_simulation_batch(200, training=True)
                        st.session_state.routing = old_routing
        results = []
        my_bar = st.progress(0)
        
        chart_data = {'Packet': [], 'Algorithm': [], 'Cumulative PDR': []}
        total_ops = len(comp_algos) * comp_steps
        current_op = 0
        
        for algo_name in comp_algos:
            bench_routing = get_routing_algo(algo_name, st.session_state.net_sim.graph, st.session_state.trust_model, st.session_state.agents)
            
            # Disable exploration for benchmark
            bench_original_eps = None
            if hasattr(bench_routing, 'agent'):
                bench_original_eps = bench_routing.agent.epsilon
                bench_routing.agent.epsilon = 0.0
            
            successes = 0
            total_lat = 0
            
            for i in range(comp_steps):
                src, dst = get_traffic_pair(i, list(st.session_state.net_sim.graph.nodes), st.session_state.net_sim.graph)
                path = bench_routing.find_path(src, dst)
                
                success = False
                if path:
                    route_success = True
                    path_lat = 0
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j+1]
                        rel = st.session_state.net_sim.graph.nodes[v].get('reliability', 1.0)
                        path_lat += st.session_state.net_sim.graph[u][v].get('weight', 10)
                        if random.random() > rel:
                            route_success = False
                            break
                    if route_success: 
                        success = True
                        total_lat += path_lat
                
                if success: successes += 1
                
                chart_data['Packet'].append(i + 1)
                chart_data['Algorithm'].append(algo_name)
                current_pdr = (successes / (i + 1)) * 100
                chart_data['Cumulative PDR'].append(current_pdr)
                
                current_op += 1
                if current_op % 5 == 0:
                    my_bar.progress(current_op / total_ops)
            
            # Restore epsilon
            if bench_original_eps is not None:
                bench_routing.agent.epsilon = bench_original_eps

            final_pdr = (successes / comp_steps) * 100
            avg_bench_lat = total_lat / successes if successes > 0 else 0
            results.append({"Algorithm": algo_name, "Final PDR (%)": final_pdr, "Avg Latency (ms)": avg_bench_lat})
        
        my_bar.empty()
        
        st.subheader("📈 Performance Evolution")
        chart_df = pd.DataFrame(chart_data)
        st.line_chart(chart_df, x="Packet", y="Cumulative PDR", color="Algorithm")
        
        st.subheader("📊 Final Results")
        res_df = pd.DataFrame(results).set_index("Algorithm")
        st.bar_chart(res_df)
        st.dataframe(res_df)

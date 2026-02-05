import simpy
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
from utils import setup_logger
from network_sim import NetworkSimulation
from trust_model import TrustModel
from routing import ShortestPathRouting, IntelligentRouting, RLRouting, TrustAwareRLRouting
from rl_agent import QLearningAgent, TrustQLearningAgent
from ospf_routing import OSPFRouting
import networkx as nx

# Disable inner logs for cleaner output
logging.getLogger("NetworkSim").setLevel(logging.WARNING)
logging.getLogger("Main").setLevel(logging.WARNING)

def run_enhanced_scenario(algo_name, routing_class, agent=None, trust_model=None, 
                          packets=3000, warmup_ratio=0.4, blackhole_nodes=None):
    """
    Runs an enhanced scenario with 3000 packets and warmup period.
    
    Args:
        algo_name: Name of the algorithm
        routing_class: Routing class to use
        agent: RL agent (if applicable)
        trust_model: Trust model (if applicable)
        packets: Total number of packets (default 3000)
        warmup_ratio: Ratio of packets to ignore (default 0.4 = 40%)
        blackhole_nodes: List of blackhole node IDs
    
    Returns:
        dict: Statistics including PDR, latency, trust_convergence, time_series
    """
    env = simpy.Environment()
    net_sim = NetworkSimulation(env)
    
    # Deterministic Topology
    random.seed(42) 
    net_sim.create_topology(num_nodes=20, connectivity=0.3)
    
    # Set Blackhole Nodes
    # If not provided, pick high-centrality nodes so the attack meaningfully impacts shortest-path protocols.
    if blackhole_nodes is None:
        try:
            und = net_sim.graph.to_undirected()
            centrality = nx.betweenness_centrality(und, normalized=True)
            blackhole_nodes = [n for n, _ in sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:3]]
        except Exception:
            blackhole_nodes = [3, 7, 12]  # Fallback
    
    for n in net_sim.nodes: 
        net_sim.graph.nodes[n]['reliability'] = 0.99
    for n in blackhole_nodes:
        if n in net_sim.graph.nodes:
            net_sim.graph.nodes[n]['reliability'] = 0.1  # 90% drop rate (blackhole)

    # Patch simulation with enhanced metrics tracking
    # Return: (success, total_delay, fail_hop_index)
    # fail_hop_index is the hop index i where u=path[i], v=path[i+1] failed; None if success.
    def realistic_simulate_packet(path, trust_model=None):
        if not path: 
            return False, 0, None
        
        success = True
        total_delay = 0
        fail_hop_index = None
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            reliability = net_sim.graph.nodes[v].get('reliability', 1.0)
            
            # Calculate delay
            delay = net_sim.graph[u][v].get('weight', 10)
            total_delay += delay
            
            # Get bandwidth
            bandwidth = net_sim.graph[u][v].get('capacity', 50)
            
            # Check if packet is dropped
            if random.random() > reliability:
                success = False
                if trust_model: 
                    trust_model.update_trust(v, False, delay=delay, bandwidth=bandwidth)
                fail_hop_index = i
                break
            else:
                if trust_model: 
                    trust_model.update_trust(v, True, delay=delay, bandwidth=bandwidth)
        
        return success, total_delay, fail_hop_index
    
    net_sim.simulate_packet = realistic_simulate_packet

    # Setup Routing
    if not trust_model and algo_name not in ["RIP", "OSPF", "Standard"]:
        trust_model = TrustModel()
    
    if algo_name == "Q-Learning":
        routing_algo = routing_class(net_sim.graph, agent)
    elif algo_name == "Trust-Aware Q-Learning":
        routing_algo = routing_class(net_sim.graph, agent, trust_model)
    elif algo_name == "Intelligent":
        routing_algo = routing_class(net_sim.graph, trust_model)
    elif algo_name in ["RIP", "OSPF"]:
        routing_algo = routing_class(net_sim.graph)
    else:
        routing_algo = routing_class(net_sim.graph)

    # Traffic Generation
    flows = []
    nodes = net_sim.nodes
    random.seed(101)
    for _ in range(20):
        src, dst = random.sample(nodes, 2)
        flows.append((src, dst))

    # Statistics tracking
    stats = {
        'total': 0,
        'success': 0,
        'latency': 0,
        'warmup_packets': int(packets * warmup_ratio),
        'time_series': {'pdr': [], 'trust_avg': [], 'latency_avg': []},
        'trust_convergence_time': None
    }
    
    window_size = 100  # For moving average

    def traffic_gen():
        for i in range(packets):
            src, dst = random.choice(flows)
            
            # End of warmup: switch RL policies to exploit for fair PDR measurement.
            if i == stats['warmup_packets'] and agent is not None and hasattr(agent, "epsilon"):
                agent.epsilon = 0.0

            # Find path
            path = routing_algo.find_path(src, dst)
            
            if path:
                # Simulate packet
                success, path_latency, fail_hop_index = net_sim.simulate_packet(path, trust_model)
                
                # Online learning for RL agents (both trust-aware and vanilla)
                if agent is not None and hasattr(agent, "learn") and len(path) > 1:
                    for hop_i in range(len(path) - 1):
                        u, v = path[hop_i], path[hop_i + 1]
                        nxt_nbrs = list(net_sim.graph.neighbors(v)) if v in net_sim.graph else []
                        
                        hop_delay = net_sim.graph[u][v].get('weight', 10)
                        
                        # Reward shaping:
                        # - Encourage progress/successful hops
                        # - Penalize failures strongly
                        # - Small penalty for delay to avoid very long/slow paths
                        hop_success = success and (fail_hop_index is None or hop_i < fail_hop_index)
                        done = (v == dst) or (fail_hop_index == hop_i)
                        reward = (1.0 if hop_success else -25.0) - (0.05 * float(hop_delay))
                        if done and hop_success and v == dst:
                            reward += 10.0
                        
                        agent.learn(u, v, reward, v, nxt_nbrs, target_node=dst, done=done)
                        if done:
                            break
                
                # Only count stats after warmup period
                if i >= stats['warmup_packets']:
                    stats['total'] += 1
                    if success:
                        stats['success'] += 1
                        stats['latency'] += path_latency
                
                # Track time series (every 50 packets)
                if i % 50 == 0:
                    current_pdr = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    stats['time_series']['pdr'].append(current_pdr)
                    stats['time_series']['latency_avg'].append(
                        stats['latency'] / stats['success'] if stats['success'] > 0 else 0
                    )
                    
                    # Track average trust
                    if trust_model:
                        avg_trust = sum(trust_model.node_trust.values()) / len(trust_model.node_trust) if trust_model.node_trust else 1.0
                        stats['time_series']['trust_avg'].append(avg_trust)
                        
                        # Detect trust convergence (when trust stabilizes)
                        if stats['trust_convergence_time'] is None and i > 500:
                            recent_trust = stats['time_series']['trust_avg'][-10:]
                            if len(recent_trust) >= 10:
                                trust_variance = np.var(recent_trust)
                                if trust_variance < 0.01:  # Trust has converged
                                    stats['trust_convergence_time'] = i
            
            # Decay epsilon for RL agents (warmup only; evaluation uses exploitation)
            if agent and i < stats['warmup_packets'] and i % 100 == 0:
                agent.decay_epsilon()
            
            yield env.timeout(0.1)

    proc = env.process(traffic_gen())
    env.run(until=proc)
    
    # Calculate final metrics
    pdr = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
    avg_lat = (stats['latency'] / stats['success']) if stats['success'] > 0 else 0
    
    return {
        'pdr': pdr,
        'latency': avg_lat,
        'trust_convergence_time': stats['trust_convergence_time'],
        'time_series': stats['time_series']
    }

def main():
    print("=" * 60)
    print("Enhanced Trust-Aware Q-Routing Evaluation")
    print("=" * 60)
    print("Configuration:")
    print("  - Total Packets: 3000")
    print("  - Warmup Period: 40% (first 1200 packets ignored)")
    print(f"  - Trust Initial: 1.0")
    print(f"  - Trust Decay: x0.3 (very aggressive)")
    print(f"  - Trust Recovery: +0.02 (slow)")
    print(f"  - Trust Threshold: 0.4")
    print(f"  - Blackhole Detection: 3 consecutive failures")
    print("=" * 60)
    
    results = {}
    
    # Pre-scan nodes for agents
    dummy_net = NetworkSimulation(simpy.Environment())
    random.seed(42)
    dummy_net.create_topology(num_nodes=20, connectivity=0.3)

    # Choose blackholes once (shared across all algorithms for a fair comparison).
    try:
        und = dummy_net.graph.to_undirected()
        centrality = nx.betweenness_centrality(und, normalized=True)
        blackhole_nodes = [n for n, _ in sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    except Exception:
        blackhole_nodes = [3, 7, 12]
    print(f"  - Blackhole Nodes: {blackhole_nodes}")
    
    # 1. RIP (Hop Count)
    print("\n[1/4] Simulating RIP (Hop Count)...")
    from routing import RIPRouting
    result = run_enhanced_scenario("RIP", RIPRouting, packets=3000, blackhole_nodes=blackhole_nodes)
    results['RIP'] = result
    print(f"  [OK] PDR: {result['pdr']:.2f}%, Latency: {result['latency']:.2f}ms")
    
    # 2. OSPF (Link State)
    print("\n[2/4] Simulating OSPF (Link State)...")
    result = run_enhanced_scenario("OSPF", OSPFRouting, packets=3000, blackhole_nodes=blackhole_nodes)
    results['OSPF'] = result
    print(f"  [OK] PDR: {result['pdr']:.2f}%, Latency: {result['latency']:.2f}ms")
    
    # 3. Standard Q-Learning (No Trust)
    print("\n[3/4] Simulating Q-Learning (No Trust)...")
    agent = QLearningAgent(dummy_net.nodes, alpha=0.6, gamma=0.8, epsilon=0.9)
    result = run_enhanced_scenario("Q-Learning", RLRouting, agent=agent, packets=3000, blackhole_nodes=blackhole_nodes)
    results['Q-Learning'] = result
    print(f"  [OK] PDR: {result['pdr']:.2f}%, Latency: {result['latency']:.2f}ms")
    
    # 4. Trust-Aware Q-Learning (Enhanced)
    print("\n[4/4] Simulating Trust-Aware Q-Learning (Enhanced)...")
    trust_agent = TrustQLearningAgent(
        dummy_net.nodes, 
        alpha=0.6, 
        gamma=0.8, 
        epsilon=0.9,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        trust_impact=3.0
    )
    ta_trust_model = TrustModel(
        initial_trust=1.0,
        decay_factor=0.3,  # Very aggressive decay
        bonus_factor=0.02,
        trust_threshold=0.4
    )
    result = run_enhanced_scenario(
        "Trust-Aware Q-Learning", 
        TrustAwareRLRouting, 
        agent=trust_agent, 
        trust_model=ta_trust_model, 
        packets=3000,
        blackhole_nodes=blackhole_nodes
    )
    results['Trust-Aware Q-Learning'] = result
    print(f"  [OK] PDR: {result['pdr']:.2f}%, Latency: {result['latency']:.2f}ms")
    if result['trust_convergence_time']:
        print(f"  [OK] Trust Converged at: {result['trust_convergence_time']} packets")
    
    # Generate Plots
    print("\n" + "=" * 60)
    print("Generating Comparison Plots...")
    print("=" * 60)
    
    # Plot 1: Final Performance Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = list(results.keys())
    pdrs = [results[l]['pdr'] for l in labels]
    lats = [results[l]['latency'] for l in labels]
    
    # PDR Bar Chart
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars1 = ax1.bar(labels, pdrs, color=colors)
    ax1.set_ylabel('Packet Delivery Ratio (%)', fontsize=12)
    ax1.set_title('PDR Comparison (Blackhole Attack)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Latency Bar Chart
    bars2 = ax2.bar(labels, lats, color=colors)
    ax2.set_ylabel('Average Latency (ms)', fontsize=12)
    ax2.set_title('Latency Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('enhanced_comparison_final.png', dpi=300)
    print("  [OK] Saved: enhanced_comparison_final.png")
    
    # Plot 2: PDR vs Time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for algo_name, color in zip(labels, colors):
        pdr_series = results[algo_name]['time_series']['pdr']
        x_vals = list(range(0, len(pdr_series) * 50, 50))
        ax.plot(x_vals, pdr_series, label=algo_name, color=color, linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Packets Sent', fontsize=12)
    ax.set_ylabel('Packet Delivery Ratio (%)', fontsize=12)
    ax.set_title('PDR vs Time (Blackhole Attack)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axvline(x=1200, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    
    plt.tight_layout()
    plt.savefig('pdr_vs_time.png', dpi=300)
    print("  [OK] Saved: pdr_vs_time.png")
    
    # Plot 3: Trust Convergence
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'trust_avg' in results['Trust-Aware Q-Learning']['time_series']:
        trust_series = results['Trust-Aware Q-Learning']['time_series']['trust_avg']
        x_vals = list(range(0, len(trust_series) * 50, 50))
        ax.plot(x_vals, trust_series, color='#96CEB4', linewidth=2, marker='o', markersize=3)
        
        if results['Trust-Aware Q-Learning']['trust_convergence_time']:
            conv_time = results['Trust-Aware Q-Learning']['trust_convergence_time']
            ax.axvline(x=conv_time, color='red', linestyle='--', alpha=0.7, 
                      label=f'Convergence at {conv_time} packets')
    
    ax.set_xlabel('Packets Sent', fontsize=12)
    ax.set_ylabel('Average Trust Score', fontsize=12)
    ax.set_title('Trust Convergence Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('trust_convergence.png', dpi=300)
    print("  [OK] Saved: trust_convergence.png")
    
    # Print Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for algo_name in labels:
        print(f"\n{algo_name}:")
        print(f"  PDR: {results[algo_name]['pdr']:.2f}%")
        print(f"  Latency: {results[algo_name]['latency']:.2f}ms")
        if results[algo_name]['trust_convergence_time']:
            print(f"  Trust Convergence: {results[algo_name]['trust_convergence_time']} packets")
    
    print("\n" + "=" * 60)
    print("[OK] Evaluation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

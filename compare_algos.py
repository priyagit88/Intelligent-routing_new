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

# Disable inner logs for cleaner output
logging.getLogger("NetworkSim").setLevel(logging.WARNING)
logging.getLogger("Main").setLevel(logging.WARNING)

def run_scenario(algo_name, routing_class, agent=None, congestion=True, packets=100, training=False, trust_model=None):
    """
    Runs a single scenario and returns stats.
    """
    env = simpy.Environment()
    net_sim = NetworkSimulation(env)
    
    # Deterministic Topology
    random.seed(42) 
    net_sim.create_topology(num_nodes=20, connectivity=0.3)
    
    # Set Bad Nodes (3, 7, 12)
    bad_nodes = [3, 7, 12]
    for n in net_sim.nodes: 
        net_sim.graph.nodes[n]['reliability'] = 0.99
    for n in bad_nodes:
        if n in net_sim.graph.nodes:
            net_sim.graph.nodes[n]['reliability'] = 0.5 # 50% drop rate!

    # Patch simulation
    def realistic_simulate_packet(path, trust_model=None):
        if not path: return False
        success = True
        for i in range(len(path) - 1):
             u, v = path[i], path[i+1]
             reliability = net_sim.graph.nodes[v].get('reliability', 1.0)
             if random.random() > reliability:
                 success = False
                 if trust_model: trust_model.update_trust(v, False)
                 break
             else:
                 if trust_model: trust_model.update_trust(v, True)
        return success
    net_sim.simulate_packet = realistic_simulate_packet

    if congestion:
        env.process(net_sim.update_congestion())

    # Setup Routing
    if not trust_model and algo_name != "Standard" and algo_name != "Q-Learning":
        trust_model = TrustModel()
    
    if algo_name == "Q-Learning":
        routing_algo = routing_class(net_sim.graph, agent)
    elif algo_name == "Trust-Aware Q-Learning":
        # Ensure we pass the trust model to the routing algo locally if needed,
        # but TrustAwareRLRouting signature is (graph, agent, trust_model)
        routing_algo = routing_class(net_sim.graph, agent, trust_model)
    elif algo_name == "Intelligent":
        routing_algo = routing_class(net_sim.graph, trust_model)
    else:
        routing_algo = routing_class(net_sim.graph)

    # Traffic Generation
    flows = []
    nodes = net_sim.nodes
    random.seed(101) # Seed for traffic
    for _ in range(20):
        src, dst = random.sample(nodes, 2)
        flows.append((src, dst))

    stats = {'success': 0, 'latency': 0, 'total': packets}

    def traffic_gen():
        for i in range(packets):
            src, dst = random.choice(flows)
            
            # RL Learning Step
            if training and isinstance(routing_algo, RLRouting):
                # Custom loop for training
                path = []
                curr = src
                visited = set([src])
                while curr != dst and len(path) < 20:
                    path.append(curr)
                    nbrs = list(net_sim.graph.neighbors(curr))
                    if not nbrs: break
                    
                    # NEW: Support Trust-Aware choice in training loop
                    if isinstance(routing_algo.agent, TrustQLearningAgent) and trust_model:
                        trust_scores = {n: trust_model.get_trust(n) for n in nbrs}
                        nxt = routing_algo.agent.choose_action(curr, nbrs, trust_scores=trust_scores, avoid_nodes=visited)
                    else:
                        nxt = routing_algo.agent.choose_action(curr, nbrs, avoid_nodes=visited) # Add visited check to standard too for fairness?
                        # Actually standard Q-Learning in this repo didn't have avoid_nodes until now? 
                        # RL agent code has it in signature now? Yes.
                    
                    if nxt is None: break
                    curr = nxt
                    visited.add(curr)
                    
                if curr == dst: path.append(dst)

                # Eval path
                success = net_sim.simulate_packet(path, trust_model)
                path_latency = sum(net_sim.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])) if len(path)>1 else 0
                
                stats['total'] = packets # Ensure correct count logic
                if success: stats['success'] += 1
                
                # Learn
                reward = -path_latency if success else -100
                for idx in range(len(path) - 1):
                    u, v = path[idx], path[idx+1]
                    nxt_nbrs = list(net_sim.graph.neighbors(v)) if v in net_sim.graph else []
                    routing_algo.agent.learn(u, v, reward, v, nxt_nbrs)
            
            else:
                # Normal Packet Routing
                path = routing_algo.find_path(src, dst)
                if path:
                    path_latency = sum(net_sim.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    success = net_sim.simulate_packet(path, trust_model)
                    
                    if success:
                        stats['success'] += 1
                        stats['latency'] += path_latency
            
            yield env.timeout(0.1)

    proc = env.process(traffic_gen())
    env.run(until=proc)
    
    # Calc Metrics
    pdr = (stats['success'] / stats['total']) * 100
    avg_lat = (stats['latency'] / stats['success']) if stats['success'] > 0 else 0
    
    return pdr, avg_lat

def main():
    print("Running Comparative Analysis...")
    
    results = {
        'Standard (Dijkstra)': {},
        'Intelligent (Trust)': {},
        'Q-Learning (AI)': {},
        'Trust-Aware Q (New)': {}
    }

    # 1. Standard
    print("Simulating Standard...")
    pdr, lat = run_scenario("Standard", ShortestPathRouting)
    results['Standard (Dijkstra)'] = {'PDR': pdr, 'Latency': lat}

    # 2. Intelligent
    print("Simulating Intelligent...")
    pdr, lat = run_scenario("Intelligent", IntelligentRouting)
    results['Intelligent (Trust)'] = {'PDR': pdr, 'Latency': lat}

    # 3. RL (Train then Test)
    print("Simulating Q-Learning...")
    # Pre-scan nodes for agent
    dummy_net = NetworkSimulation(simpy.Environment())
    random.seed(42); dummy_net.create_topology(num_nodes=20, connectivity=0.3)
    agent = QLearningAgent(dummy_net.nodes, epsilon=0.5)
    
    # Train
    print("  - Training RL Agent...")
    run_scenario("Q-Learning", RLRouting, agent=agent, packets=500, training=True)
    
    # Test
    agent.epsilon = 0.05
    pdr, lat = run_scenario("Q-Learning", RLRouting, agent=agent, packets=100, training=False)
    results['Q-Learning (AI)'] = {'PDR': pdr, 'Latency': lat}

    # 4. Trust-Aware RL
    print("Simulating Trust-Aware Q-Routing...")
    trust_agent = TrustQLearningAgent(dummy_net.nodes, epsilon=0.5, trust_impact=3.0) # Higher impact
    ta_trust_model = TrustModel() # One trust model shared/updated during training
    
    # Train
    print("  - Training Trust-RL Agent...")
    run_scenario("Trust-Aware Q-Learning", TrustAwareRLRouting, agent=trust_agent, trust_model=ta_trust_model, packets=500, training=True)
    
    # Test
    trust_agent.epsilon = 0.05
    pdr, lat = run_scenario("Trust-Aware Q-Learning", TrustAwareRLRouting, agent=trust_agent, trust_model=ta_trust_model, packets=100, training=False)
    results['Trust-Aware Q (New)'] = {'PDR': pdr, 'Latency': lat}

    # Plotting
    labels = list(results.keys())
    pdrs = [results[l]['PDR'] for l in labels]
    lats = [results[l]['Latency'] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Routing Protocol')
    ax1.set_ylabel('Packet Delivery Ratio (%)', color=color)
    bars1 = ax1.bar(x - width/2, pdrs, width, color=color, label='PDR')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Avg Latency (ms)', color=color)
    bars2 = ax2.bar(x + width/2, lats, width, color=color, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.title('Comparison of Routing Algorithms (High Load + Bad Nodes)')
    
    # Add values on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}%', ha='center', va='bottom')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('comparison_chart.png')
    print("Chart saved to comparison_chart.png")

if __name__ == "__main__":
    main()

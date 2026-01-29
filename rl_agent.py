import random
import numpy as np

class QLearningAgent:
    def __init__(self, nodes, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        Q-Learning Agent for Routing.
        nodes: list of all node IDs
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.nodes = nodes
        
        # Q-Table: Q[current_node][next_hop] -> Value
        # We can use a dict of dicts for flexibility
        self.q_table = {node: {} for node in nodes}
        
        # Initialize Q-values
        for u in nodes:
            for v in nodes:
                if u != v:
                    # We only care about neighbors, but we might not know neighbors upfront 
                    # if we assume model-free. However, for routing, next_hop MUST be a neighbor.
                    # We will initialize entries lazily or assume 0.
                    pass

    def get_q_value(self, state, action):
        """State = current node, Action = next hop neighbor"""
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, current_node, neighbors, target_node=None, avoid_nodes=None):
        """
        Epsilon-greedy selection of next hop.
        """
        if not neighbors:
            return None
            
        valid_neighbors = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
        if not valid_neighbors:
             return None
            
        if random.random() < self.epsilon:
            # Explore
            return random.choice(valid_neighbors)
        else:
            # Exploit: Choose neighbor with max Q-value
            q_values = [self.get_q_value(current_node, n) for n in valid_neighbors]
            max_q = max(q_values)
            
            # Tie-breaking
            best_actions = [n for n, q in zip(valid_neighbors, q_values) if q == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_neighbors, target_node=None):
        """
        Q-Learning Update Rule:
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)
        
        # Calculate max Q for next state
        if not next_neighbors:
            max_next_q = 0.0 # Terminal or dead end
        else:
            max_next_q = max([self.get_q_value(next_state, n) for n in next_neighbors])
            
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def add_node(self, node_id):
        """Dynamically adds a new node to the Q-table"""
        if node_id not in self.nodes:
            self.nodes.append(node_id)
            self.q_table[node_id] = {}

class TrustQLearningAgent(QLearningAgent):
    def __init__(self, nodes, alpha=0.1, gamma=0.9, epsilon=0.1, trust_impact=2.0):
        """
        trust_impact: How strongly trust affects exploration probability.
        """
        super().__init__(nodes, alpha, gamma, epsilon)
        self.trust_impact = trust_impact

    def choose_action(self, current_node, neighbors, target_node=None, avoid_nodes=None, trust_scores=None):
        """
        Biased exploration based on Trust Scores.
        trust_scores: dict {node_id: trust_value (0.0 to 1.0)}
        """
        if not neighbors:
            return None
            
        valid_neighbors = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
        if not valid_neighbors:
            return None
            
        # Epsilon-Greedy with Trust Bias
        if random.random() < self.epsilon:
            # Exploration: Instead of uniform random, weight by trust
            if trust_scores:
                weights = []
                for n in valid_neighbors:
                    t = trust_scores.get(n, 0.5) # Default trust 0.5
                    # Weight = Trust^Impact (e.g., 0.9^2 = 0.81, 0.1^2 = 0.01)
                    weights.append(t ** self.trust_impact)
                
                # Check if total weight is 0 (all bad nodes?)
                total_weight = sum(weights)
                if total_weight > 0:
                    # Normalized probabilities
                    probs = [w / total_weight for w in weights]
                    # Numpy choice is cleaner but keeping it pure python/start implementation simple
                    # To use numpy:
                    # return np.random.choice(valid_neighbors, p=probs)
                    # Let's use simple weighted choice logic
                    r = random.random()
                    cumulative = 0
                    for n, p in zip(valid_neighbors, probs):
                        cumulative += p
                        if r < cumulative:
                            return n
                    return valid_neighbors[-1] # Fallback
            
            return random.choice(valid_neighbors)
        else:
            # Exploit: Same as Q-Learning (max Q)
            # Optionally, we could blend Q + Trust here too, but standard Q-routing uses Q for exploitation.
            # The Trust bias in exploration helps finding GOOD paths faster.
            return super().choose_action(current_node, neighbors, target_node, avoid_nodes)

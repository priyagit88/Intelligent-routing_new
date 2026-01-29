import numpy as np
import random
import logging
from collections import deque

logger = logging.getLogger("AdvancedAgents")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. Advanced agents (DQN, GNN) will act randomly or fail.")

class DQNAgent:
    def __init__(self, nodes, alpha=0.001, gamma=0.95, epsilon=1.0, hidden_dim=128, memory_size=2000, batch_size=32):
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        self.num_nodes = len(nodes)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        if TORCH_AVAILABLE:
            # Input: One-hot current + One-hot target (Size: 2 * num_nodes)
            # Output: Q-value for each neighbor (Size: num_nodes, masked later)
            self.model = nn.Sequential(
                nn.Linear(2 * self.num_nodes, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_nodes)
            )
            
            # Target Network
            self.target_model = nn.Sequential(
                nn.Linear(2 * self.num_nodes, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_nodes)
            )
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
            self.criterion = nn.MSELoss()
        
    def _get_state_vector(self, current_node, target_node):
        state = torch.zeros(2 * self.num_nodes)
        if current_node in self.node_to_idx:
            state[self.node_to_idx[current_node]] = 1.0
        if target_node in self.node_to_idx:
            state[self.num_nodes + self.node_to_idx[target_node]] = 1.0
        return state

    def choose_action(self, current_node, neighbors, target_node, avoid_nodes=None):
        if not neighbors:
            return None
            
        if not TORCH_AVAILABLE:
            # Filter avoided nodes for random choice too
            valid = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
            if not valid: return None
            return random.choice(valid)

        # Explore
        if random.random() < self.epsilon:
            valid = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
            if not valid: return None
            return random.choice(valid)
        
        # Exploit
        with torch.no_grad():
            state_vec = self._get_state_vector(current_node, target_node)
            q_values = self.model(state_vec) # [num_nodes]
            
            # Mask invalid neighbors (set to -inf)
            masked_q = torch.full_like(q_values, -float('inf'))
            found_valid = False
            for n in neighbors:
                if avoid_nodes and n in avoid_nodes:
                    continue
                if n in self.node_to_idx:
                    idx = self.node_to_idx[n]
                    masked_q[idx] = q_values[idx]
                    found_valid = True
            
            if not found_valid:
                return None

            best_idx = torch.argmax(masked_q).item()
            return self.idx_to_node.get(best_idx, None)

    def _soft_update(self, tau=0.01):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn(self, current_node, next_node, reward, next_state, next_neighbors, target_node=None, done=False):
        """
        DQN Update with Experience Replay and Target Network
        """
        if not TORCH_AVAILABLE:
            return

        # Store transition in memory
        # Store as indices to save space/time, or raw values. Raw is easier.
        self.memory.append((current_node, next_node, reward, next_state, next_neighbors, target_node, done))
        
        if len(self.memory) < self.batch_size:
            return
            
        # Sample Batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare Batch Tensors
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        
        valid_next_neighbor_masks = [] # List of lists of neighbor indices
        
        for (curr, act, rew, nxt, nxt_neigh, tgt, d) in batch:
            states.append(self._get_state_vector(curr, tgt))
            next_states.append(self._get_state_vector(nxt, tgt))
            actions.append(self.node_to_idx[act])
            rewards.append(rew)
            dones.append(1.0 if d else 0.0)
            
            # For next state max Q, we need neighbor indices
            neigh_indices = [self.node_to_idx[n] for n in nxt_neigh if n in self.node_to_idx]
            valid_next_neighbor_masks.append(neigh_indices)

        states_tensor = torch.stack(states)
        next_states_tensor = torch.stack(next_states)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        # 1. Prediction (Current State)
        q_values = self.model(states_tensor)
        q_pred = q_values.gather(1, actions_tensor)
        
        # 2. Target (Next State) using Target Network
        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor)
            
            # Compute Max Q for next state (constrained to neighbors)
            max_next_qs = []
            for i, neighbors_idx in enumerate(valid_next_neighbor_masks):
                if not neighbors_idx:
                    max_next_qs.append(0.0)
                else:
                    # Select only neighbor Q-values
                    neighbor_qs = next_q_values[i, neighbors_idx]
                    max_q = torch.max(neighbor_qs).item()
                    max_next_qs.append(max_q)
            
            max_next_qs_tensor = torch.tensor(max_next_qs, dtype=torch.float32).unsqueeze(1)
            q_target = rewards_tensor + (self.gamma * max_next_qs_tensor * (1 - dones_tensor))
        
        # 3. Optimization
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. Soft Update
        self._soft_update()


class GNNRLAgent:
    """
    GNN-based Reinforcement Learning Agent with Target Networks and Replay Buffer.
    """
    def __init__(self, graph, nodes, alpha=0.001, gamma=0.95, epsilon=1.0, embedding_dim=16, memory_size=2000, batch_size=32):
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        self.num_nodes = len(nodes)
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        # Adjacency Matrix (static for now, can be updated)
        self.adj_matrix = None 
        self._build_adj_matrix()
        
        if TORCH_AVAILABLE:
            # GCN Layer: H_new = ReLU(A * H * W)
            # Input Features per node: [is_target, trust_score, degree_centrality]
            self.input_dim = 3 
            self.embedding_dim = embedding_dim
            
            # GCN weights
            self.gcn_weight = nn.Parameter(torch.randn(self.input_dim, self.embedding_dim))
            
            # Q-Net Head
            self.q_head = nn.Sequential(
                nn.Linear(2 * self.embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
            # Target Network parameters (Store separate copies)
            self.target_gcn_weight = nn.Parameter(self.gcn_weight.clone().detach())
            self.target_q_head = nn.Sequential(
                nn.Linear(2 * self.embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.target_q_head.load_state_dict(self.q_head.state_dict())
            
            self.optimizer = optim.Adam(list(self.q_head.parameters()) + [self.gcn_weight], lr=self.alpha)
            
    def update_graph(self, graph, nodes):
        """Updates the internal graph structure when topology changes"""
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        self.num_nodes = len(nodes)
        self._build_adj_matrix()
            
    def _build_adj_matrix(self):
        if not TORCH_AVAILABLE: return
        adj = torch.eye(self.num_nodes) # Self loops
        for u, v in self.graph.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                i, j = self.node_to_idx[u], self.node_to_idx[v]
                adj[i, j] = 1.0
        
        # Normalize: D^-0.5 * A * D^-0.5
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        diag = torch.diag(deg_inv_sqrt)
        
        self.adj_matrix = torch.mm(torch.mm(diag, adj), diag)
        
    def _get_node_features(self, target_node):
        # Feature Matrix X: [num_nodes, input_dim]
        # Features: [Is_Target(0/1), Trust(0-1), Degree(normalized)]
        features = torch.zeros((self.num_nodes, self.input_dim))
        
        degrees = dict(self.graph.degree())
        max_deg = max(degrees.values()) if degrees else 1
        
        for i, node in enumerate(self.nodes):
            # 1. Is Target
            if node == target_node:
                features[i, 0] = 1.0
            # 2. Trust 
            features[i, 1] = 1.0 
            # 3. Degree
            features[i, 2] = degrees.get(node, 0) / max_deg
            
        return features

    def _forward(self, target_node, use_target=False):
        """Returns Embeddings for all nodes"""
        X = self._get_node_features(target_node)
        A = self.adj_matrix
        W = self.target_gcn_weight if use_target else self.gcn_weight
        
        support = torch.mm(X, W)
        output = torch.mm(A, support)
        embeddings = F.relu(output)
        return embeddings

    def choose_action(self, current_node, neighbors, target_node, avoid_nodes=None):
        if not neighbors: return None
        
        if not TORCH_AVAILABLE: 
            valid = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
            if not valid: return None
            return random.choice(valid)
        
        if random.random() < self.epsilon:
            valid = [n for n in neighbors if not avoid_nodes or n not in avoid_nodes]
            if not valid: return None
            return random.choice(valid)
            
        with torch.no_grad():
            embeddings = self._forward(target_node, use_target=False)
            curr_emb = embeddings[self.node_to_idx[current_node]]
            
            best_n = None
            max_q = -float('inf')
            found_valid = False
            
            for n in neighbors:
                if avoid_nodes and n in avoid_nodes: continue
                if n not in self.node_to_idx: continue
                neigh_emb = embeddings[self.node_to_idx[n]]
                
                cat_emb = torch.cat([curr_emb, neigh_emb])
                q_val = self.q_head(cat_emb).item()
                
                if q_val > max_q:
                    max_q = q_val
                    best_n = n
                    found_valid = True
            
            if not found_valid:
                return None
            return best_n
            
    def _soft_update(self, tau=0.01):
        # Update GCN weight
        self.target_gcn_weight.data.copy_(tau * self.gcn_weight.data + (1.0 - tau) * self.target_gcn_weight.data)
        # Update Q-Head
        for target_param, local_param in zip(self.target_q_head.parameters(), self.q_head.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn(self, current_node, next_node, reward, next_state, next_neighbors, target_node=None, done=False):
        if not TORCH_AVAILABLE: return
        
        try:
            self.memory.append((current_node, next_node, reward, next_state, next_neighbors, target_node, done))
            if len(self.memory) < self.batch_size: return
            
            batch = random.sample(self.memory, self.batch_size)
            loss_total = 0
            
            self.optimizer.zero_grad()
            
            for (curr, act, rew, nxt, nxt_neigh, tgt, d) in batch:
                # 1. Prediction with Local Net
                emb = self._forward(tgt, use_target=False)
                curr_emb = emb[self.node_to_idx[curr]]
                act_emb = emb[self.node_to_idx[act]]
                pred_q = self.q_head(torch.cat([curr_emb, act_emb]))
                
                # 2. Target with Target Net
                with torch.no_grad():
                    target_q_val = 0.0
                    if d:
                        target_q_val = rew
                    else:
                        target_emb = self._forward(tgt, use_target=True)
                        nxt_s_emb = target_emb[self.node_to_idx[nxt]]
                        
                        max_next_q = -float('inf')
                        if not nxt_neigh:
                            max_next_q = 0.0
                        else:
                             for n in nxt_neigh:
                                if n in self.node_to_idx:
                                    n_emb = target_emb[self.node_to_idx[n]]
                                    q = self.target_q_head(torch.cat([nxt_s_emb, n_emb])).item()
                                    if q > max_next_q: max_next_q = q
                        
                        if max_next_q == -float('inf'): max_next_q = 0.0
                        target_q_val = rew + self.gamma * max_next_q
                
                target_q_tensor = torch.tensor([target_q_val], dtype=torch.float32)
                loss = F.mse_loss(pred_q, target_q_tensor)
                loss_total += loss
            
            # Average loss
            loss_total = loss_total / self.batch_size
            loss_total.backward()
            self.optimizer.step()
            
            self._soft_update()
        except Exception as e:
            logger.error(f"GNN Learn Error: {e}")
            pass # Skip this batch step to prevent crash

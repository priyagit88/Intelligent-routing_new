import networkx as nx

class RoutingAlgorithm:
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, source, target):
        raise NotImplementedError("Subclasses must implement find_path")

class ShortestPathRouting(RoutingAlgorithm):
    def find_path(self, source, target):
        try:
            return nx.shortest_path(self.graph, source=source, target=target, weight='weight')
        except nx.NetworkXNoPath:
            return None

class IntelligentRouting(RoutingAlgorithm):
    def __init__(self, graph, trust_model):
        super().__init__(graph)
        self.trust_model = trust_model

    def calculate_cost(self, u, v, data):
        """
        Custom cost function that considers:
        - Link latency (weight)
        - Destination Node Trust (or next hop trust)
        """
        base_cost = data.get('weight', 1)
        trust_score = self.trust_model.get_trust(v)
        
        # Invert trust so lower trust = higher cost
        # cost = base_weight * (1 + (1 - trust))
        # If trust is 1.0, cost = base_weight
        # If trust is 0.0, cost = base_weight * 51 (extreme penalty)
        trust_penalty = (1 - trust_score) * 50 # Multiplier for impact (Increased from 10 to 50)
        
        return base_cost * (1 + trust_penalty)

    def find_path(self, source, target):
        try:
            return nx.shortest_path(self.graph, source=source, target=target, weight=self.calculate_cost)
        except nx.NetworkXNoPath:
            return None

class RLRouting(RoutingAlgorithm):
    def __init__(self, graph, agent):
        super().__init__(graph)
        self.agent = agent
    
    def find_path(self, source, target):
        """
        Route packet hop-by-hop using Q-Learning Agent.
        Note: This is different from Dijkstra. We don't return a full path upfront typically in RL routing,
        but for this simulation compatibility, we will simulate the hop-by-hop decisions to generate a 'path'.
        """
        path = [source]
        current = source
        visited = set([source])
        max_hops = 20
        
        while current != target and len(path) < max_hops:
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
                
            # Pass visited nodes to exclude loops
            # Note: We must check if the agent supports 'avoid_nodes' (QLearningAgent might not yet)
            # But we should update QLearningAgent too if needed, or check signature.
            # For now, let's assume advanced agents support it. QLearningAgent checks args? 
            # Actually, Python accepts **kwargs or we updated QLearningAgent signature?
            # We updated advanced_agents.py. We need to check QLearningAgent in rl_agent.py.
            # Let's inspect signature first via try/except or just pass specific kwargs for robustnes
            if "avoid_nodes" in self.agent.choose_action.__code__.co_varnames:
                 next_hop = self.agent.choose_action(current, neighbors, target, avoid_nodes=visited)
            else:
                 next_hop = self.agent.choose_action(current, neighbors, target)
            
            if next_hop is None:
                break
            
            # Prevent simple loops
            if next_hop in visited:
                 # Avoid loops by choosing an alternate unvisited neighbor if possible.
                 unvisited = [n for n in neighbors if n not in visited]
                 if not unvisited:
                     break
                 if "avoid_nodes" in self.agent.choose_action.__code__.co_varnames:
                     next_hop = self.agent.choose_action(current, unvisited, target, avoid_nodes=visited)
                 else:
                     next_hop = self.agent.choose_action(current, unvisited, target)
                 if next_hop is None or next_hop in visited:
                     break

            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            
        if current == target:
            return path
        return None

class RIPRouting(RoutingAlgorithm):
    """
    Simulates RIP (Routing Information Protocol).
    - Metric: Hop Count (Distance Vector).
    - Ignores Link Latency/Congestion.
    """
    def __init__(self, graph):
        self.graph = graph

    def find_path(self, source, target):
        try:
            # Shortest path with weight=None implies BFS (Hop Count)
            return nx.shortest_path(self.graph, source=source, target=target, weight=None)
        except nx.NetworkXNoPath:
            return None

class TrustAwareRLRouting(RLRouting):
    def __init__(self, graph, agent, trust_model, use_multi_metric=True):
        super().__init__(graph, agent)
        self.trust_model = trust_model
        self.use_multi_metric = use_multi_metric

    def calculate_multi_metric_score(self, node_id):
        """
        Calculate multi-metric routing score for a node.
        Score = 0.4 × trust + 0.3 × norm_bw - 0.2 × norm_delay - 0.1 × pkt_loss
        
        Returns:
            float: Routing score (higher is better)
        """
        trust = self.trust_model.get_trust(node_id)
        metrics = self.trust_model.get_metrics(node_id)
        
        score = (
            0.4 * trust +
            0.3 * metrics["bandwidth"] -
            0.2 * metrics["delay"] -
            0.1 * metrics["packet_loss"]
        )
        return score

    def _has_path(self, src, dst):
        """Fast reachability guard for directed graphs."""
        try:
            return nx.has_path(self.graph, src, dst)
        except Exception:
            return False

    def _progress_score(self, node_id, target):
        """
        Heuristic: how close node_id is to target (higher is better).
        Uses hop distance (unweighted) for speed and robustness.
        """
        try:
            hops = nx.shortest_path_length(self.graph, source=node_id, target=target)
        except Exception:
            return 0.0
        # 1 hop away -> 0.5, 2 hops -> 0.33, etc.
        return 1.0 / (1.0 + float(hops))

    def find_path(self, source, target):
        path = [source]
        current = source
        visited = set([source])
        max_hops = 20
        
        while current != target and len(path) < max_hops:
            neighbors = list(self.graph.neighbors(current))
            if not neighbors:
                break
            
            # TRUST GATE: Filter out low-trust nodes (trust < 0.4)
            trusted_neighbors = [
                n for n in neighbors 
                if self.trust_model.is_trusted(n) and n not in visited
            ]
            
            # If no trusted neighbors, fall back to all unvisited neighbors
            # (This handles edge cases where all neighbors are untrusted)
            if not trusted_neighbors:
                trusted_neighbors = [n for n in neighbors if n not in visited]
            
            if not trusted_neighbors:
                break

            # REACHABILITY GATE: avoid dead-ends that cannot reach target.
            # This is critical for directed/random graphs; otherwise the hop-by-hop
            # policy can easily get stuck even though a global path exists.
            reachable_neighbors = [n for n in trusted_neighbors if self._has_path(n, target)]
            if reachable_neighbors:
                trusted_neighbors = reachable_neighbors
            
            # Get Trust Scores for trusted neighbors
            trust_scores = {}
            for n in trusted_neighbors:
                base = self.trust_model.get_trust(n)
                if self.use_multi_metric:
                    base = self.calculate_multi_metric_score(n)

                # Add a light goal-directed heuristic so the agent doesn't "wander".
                prog = self._progress_score(n, target)
                edge_delay = self.graph[current][n].get('weight', 10) if current in self.graph and n in self.graph[current] else 10
                delay_score = 1.0 / (1.0 + float(edge_delay))

                # If the trust model has already flagged a node as blackhole, heavily down-rank it.
                is_blackhole = bool(self.trust_model.stats.get(n, {}).get("is_blackhole", False))
                blackhole_penalty = 0.0 if not is_blackhole else -1.0

                # Blend: trust/metrics dominate, but we still prefer neighbors that can reach target cheaply.
                trust_scores[n] = (0.60 * base) + (0.30 * prog) + (0.10 * delay_score) + blackhole_penalty
            
            # Pass trust_scores to the TrustQLearningAgent
            if "trust_scores" in self.agent.choose_action.__code__.co_varnames:
                 next_hop = self.agent.choose_action(current, trusted_neighbors, target, avoid_nodes=visited, trust_scores=trust_scores)
            else:
                 # Fallback if agent doesn't support trust (shouldn't happen if wired correctly)
                 next_hop = self.agent.choose_action(current, trusted_neighbors, target, avoid_nodes=visited)
            
            if next_hop is None:
                break
            
            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            
        if current == target:
            return path
        return None

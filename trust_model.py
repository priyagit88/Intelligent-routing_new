class TrustModel:
    def __init__(self, initial_trust=1.0, decay_factor=0.3, bonus_factor=0.02, trust_threshold=0.4):
        """
        Enhanced Trust Model with very aggressive decay and blackhole detection.
        
        Args:
            initial_trust: Initial trust value for all nodes (1.0 = fully trusted)
            decay_factor: Multiplier for trust on failure (0.3 = very aggressive decay)
            bonus_factor: Additive bonus for trust on success (0.02 = slow recovery)
            trust_threshold: Minimum trust for routing consideration (0.4)
        """
        self.node_trust = {}
        self.initial_trust = initial_trust
        self.decay_factor = decay_factor  # Penalty for bad behavior (×0.3)
        self.bonus_factor = bonus_factor  # Bonus for good behavior (+0.02)
        self.trust_threshold = trust_threshold  # Hard threshold for routing
        
        # Tracking statistics for diagnosis and metrics
        # {node_id: {"data_success": 0, "data_fail": 0, "voice_success": 0, "voice_fail": 0, 
        #            "history": [], "bandwidth_util": [], "delays": [], "packet_loss": 0,
        #            "consecutive_failures": 0, "is_blackhole": False}}
        self.stats = {}
        
        # Neighbor relationships for trust propagation
        self.neighbors = {}  # {node_id: [neighbor_ids]}

    def initialize_node(self, node_id, neighbors=None):
        """Initialize a node with default trust and statistics."""
        if node_id not in self.node_trust:
            self.node_trust[node_id] = self.initial_trust
            
        if node_id not in self.stats:
            self.stats[node_id] = {
                "data_success": 0, "data_fail": 0, 
                "voice_success": 0, "voice_fail": 0,
                "history": [],
                "bandwidth_util": [],  # Track bandwidth utilization
                "delays": [],  # Track delays
                "packet_loss": 0,  # Track packet loss count
                "total_packets": 0,  # Total packets handled
                "consecutive_failures": 0,  # Track consecutive failures
                "is_blackhole": False  # Blackhole detection flag
            }
            
        if neighbors:
            self.neighbors[node_id] = neighbors

    def update_trust(self, node_id, success, **kwargs):
        """
        Updates trust score based on transaction success/failure.
        Implements very aggressive decay (×0.3) and slow recovery (+0.02).
        Detects blackhole nodes after 3 consecutive failures.
        
        Args:
            node_id: Node to update
            success: True if packet forwarded successfully, False otherwise
            **kwargs: Additional parameters (priority, delay, bandwidth)
        """
        priority = kwargs.get('priority', 0)
        delay = kwargs.get('delay', None)
        bandwidth = kwargs.get('bandwidth', None)
        
        if node_id not in self.node_trust or node_id not in self.stats:
            self.initialize_node(node_id)
        
        # Update Stats
        p_type = "voice" if priority == 1 else "data"
        s_type = "success" if success else "fail"
        self.stats[node_id][f"{p_type}_{s_type}"] += 1
        self.stats[node_id]["total_packets"] += 1
        
        if not success:
            self.stats[node_id]["packet_loss"] += 1
            self.stats[node_id]["consecutive_failures"] += 1
            
            # Blackhole detection: 3 consecutive failures
            if self.stats[node_id]["consecutive_failures"] >= 3:
                self.stats[node_id]["is_blackhole"] = True
                # Immediately drop trust to 0 for detected blackholes
                self.node_trust[node_id] = 0.0
        else:
            # Reset consecutive failures on success
            self.stats[node_id]["consecutive_failures"] = 0
        
        # Track delay if provided
        if delay is not None:
            self.stats[node_id]["delays"].append(delay)
            if len(self.stats[node_id]["delays"]) > 50:
                self.stats[node_id]["delays"].pop(0)
        
        # Track bandwidth if provided
        if bandwidth is not None:
            self.stats[node_id]["bandwidth_util"].append(bandwidth)
            if len(self.stats[node_id]["bandwidth_util"]) > 50:
                self.stats[node_id]["bandwidth_util"].pop(0)
        
        # Keep short history for On-Off analysis
        self.stats[node_id]["history"].append(1 if success else 0)
        if len(self.stats[node_id]["history"]) > 20:
            self.stats[node_id]["history"].pop(0)

        # Very aggressive asymmetric trust update (only if not already blackhole)
        if not self.stats[node_id]["is_blackhole"]:
            current = self.node_trust[node_id]
            if success:
                # Slow recovery: +0.02
                self.node_trust[node_id] = min(1.0, current + self.bonus_factor)
            else:
                # Very aggressive decay: ×0.3
                self.node_trust[node_id] = max(0.0, current * self.decay_factor)
    
    def get_trust(self, node_id, use_propagation=False):
        """
        Get trust value for a node, optionally with neighbor propagation.
        
        Args:
            node_id: Node to query
            use_propagation: If True, apply trust propagation formula
        
        Returns:
            Trust value (0.0 to 1.0)
        """
        if node_id not in self.node_trust:
            return self.initial_trust
        
        direct_trust = self.node_trust[node_id]
        
        if not use_propagation or node_id not in self.neighbors:
            return direct_trust
        
        # Trust Propagation: final_trust = 0.7 × direct + 0.3 × avg_neighbor
        neighbor_ids = self.neighbors.get(node_id, [])
        if not neighbor_ids:
            return direct_trust
        
        neighbor_trusts = [self.node_trust.get(n, self.initial_trust) for n in neighbor_ids]
        avg_neighbor_trust = sum(neighbor_trusts) / len(neighbor_trusts)
        
        final_trust = 0.7 * direct_trust + 0.3 * avg_neighbor_trust
        return final_trust
    
    def is_trusted(self, node_id, use_propagation=False):
        """
        Check if a node meets the trust threshold for routing.
        
        Args:
            node_id: Node to check
            use_propagation: If True, use propagated trust value
        
        Returns:
            True if trust >= threshold, False otherwise
        """
        trust = self.get_trust(node_id, use_propagation)
        return trust >= self.trust_threshold
    
    def get_metrics(self, node_id):
        """
        Get normalized metrics for multi-metric routing score.
        
        Returns:
            dict with normalized bandwidth, delay, and packet_loss
        """
        if node_id not in self.stats:
            return {"bandwidth": 0.5, "delay": 0.5, "packet_loss": 0.0}
        
        s = self.stats[node_id]
        
        # Normalized bandwidth (higher is better, 0-1)
        avg_bw = sum(s["bandwidth_util"]) / len(s["bandwidth_util"]) if s["bandwidth_util"] else 50
        norm_bw = min(1.0, avg_bw / 100.0)  # Assume max 100 Mbps
        
        # Normalized delay (lower is better, invert for scoring)
        avg_delay = sum(s["delays"]) / len(s["delays"]) if s["delays"] else 10
        norm_delay = min(1.0, avg_delay / 100.0)  # Assume max 100ms
        
        # Packet loss rate (0-1)
        pkt_loss = s["packet_loss"] / s["total_packets"] if s["total_packets"] > 0 else 0.0
        
        return {
            "bandwidth": norm_bw,
            "delay": norm_delay,
            "packet_loss": pkt_loss
        }

    def diagnose_node(self, node_id):
        """
        Heuristic diagnosis of node behavior.
        """
        if node_id not in self.stats:
            return "Unknown"
        
        s = self.stats[node_id]
        total_data = s["data_success"] + s["data_fail"]
        total_voice = s["voice_success"] + s["voice_fail"]
        
        if total_data + total_voice < 5:
            return "Insufficient Data"
            
        data_pdr = s["data_success"] / total_data if total_data > 0 else 1.0
        voice_pdr = s["voice_success"] / total_voice if total_voice > 0 else 1.0
        
        # Check for Grayhole (Selective drop)
        if voice_pdr > 0.8 and data_pdr < 0.4:
            return "🔍 Potential Grayhole (Selective Dropping)"
            
        # Check for Blackhole (Constant drop)
        if voice_pdr < 0.3 and data_pdr < 0.3:
            return "🚫 Potential Blackhole (Constant Dropping)"
            
        # Check for On-Off (High variance in history)
        if len(s["history"]) >= 10:
            variance = sum((x - sum(s["history"])/len(s["history"]))**2 for x in s["history"]) / len(s["history"])
            if 0.1 < variance < 0.3 and sum(s["history"])/len(s["history"]) < 0.8:
                return "🔄 Potential On-Off Attack (Flapping)"
                
        if self.node_trust[node_id] < 0.7:
             return "⚠️ Low Trust / Performance Issues"
             
        return "✅ Normal / Healthy"

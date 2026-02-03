class TrustModel:
    def __init__(self, initial_trust=1.0, decay_factor=0.7, bonus_factor=0.1):
        self.node_trust = {}
        self.initial_trust = initial_trust
        self.decay_factor = decay_factor # Penalty for bad behavior (Lower = more aggressive)
        self.bonus_factor = bonus_factor # Bonus for good behavior (Higher = faster recovery)
        
        # New: Tracking statistics for diagnosis
        # {node_id: {"data_success": 0, "data_fail": 0, "voice_success": 0, "voice_fail": 0, "history": []}}
        self.stats = {}

    def initialize_node(self, node_id):
        if node_id not in self.node_trust:
            self.node_trust[node_id] = self.initial_trust
            self.stats[node_id] = {
                "data_success": 0, "data_fail": 0, 
                "voice_success": 0, "voice_fail": 0,
                "history": [] 
            }

    def update_trust(self, node_id, success, **kwargs):
        """
        Updates trust score based on transaction success/failure.
        Accepts 'priority' (0 for Data, 1 for Voice) via kwargs.
        """
        priority = kwargs.get('priority', 0)
        if node_id not in self.node_trust:
            self.initialize_node(node_id)
        
        # Update Stats
        p_type = "voice" if priority == 1 else "data"
        s_type = "success" if success else "fail"
        self.stats[node_id][f"{p_type}_{s_type}"] += 1
        
        # Keep short history for On-Off analysis
        self.stats[node_id]["history"].append(1 if success else 0)
        if len(self.stats[node_id]["history"]) > 20:
            self.stats[node_id]["history"].pop(0)

        current = self.node_trust[node_id]
        if success:
            self.node_trust[node_id] = min(1.0, current + self.bonus_factor)
        else:
            self.node_trust[node_id] = max(0.0, current * self.decay_factor)
    
    def get_trust(self, node_id):
        return self.node_trust.get(node_id, self.initial_trust)

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

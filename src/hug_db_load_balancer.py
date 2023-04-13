# src/hug_db_load_balancer.py

import itertools
from typing import List, Dict, Any, Tuple

class HugDBLoadBalancer:
    def __init__(self, hugdb_replicas: List[Any]) -> None:
        self.hugdb_replicas = hugdb_replicas
        self.replica_cycle = itertools.cycle(hugdb_replicas)
    
    def search(self, query_vector: List[float], k: init = 5, metadata_filter: Dict[str, Any] = None) -> Tuple[List[float], List[List[float]]]:
        selected_replica = next(self.replica_cycle)
        distances, results = selected_replica.search(query_vector, k, metadata_filter)
        return distances, results
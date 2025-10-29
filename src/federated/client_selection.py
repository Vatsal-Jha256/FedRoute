"""
Multi-Objective Client Selection for FedRoute Framework

This module implements advanced participant selection algorithms specifically
designed for the dual-task IoV environment.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from collections import defaultdict


@dataclass
class ClientMetrics:
    """Container for client performance metrics."""
    client_id: str
    path_accuracy: float
    music_accuracy: float
    data_quality: float
    communication_cost: float
    computation_time: float
    energy_consumption: float
    privacy_budget_used: float
    last_selected_round: int
    participation_count: int


class ClientSelectionStrategy(ABC):
    """Abstract base class for client selection strategies."""
    
    @abstractmethod
    def select_clients(self, 
                      available_clients: List[ClientMetrics], 
                      num_clients: int,
                      round_number: int) -> List[str]:
        """Select clients for the current round."""
        pass


class RandomSelection(ClientSelectionStrategy):
    """Random client selection baseline."""
    
    def select_clients(self, 
                      available_clients: List[ClientMetrics], 
                      num_clients: int,
                      round_number: int) -> List[str]:
        """Randomly select clients."""
        selected = random.sample(available_clients, min(num_clients, len(available_clients)))
        return [client.client_id for client in selected]


class TopKSelection(ClientSelectionStrategy):
    """Top-K selection based on combined accuracy."""
    
    def __init__(self, task_weights: Tuple[float, float] = (0.5, 0.5)):
        self.task_weights = task_weights
    
    def select_clients(self, 
                      available_clients: List[ClientMetrics], 
                      num_clients: int,
                      round_number: int) -> List[str]:
        """Select top-K clients based on weighted accuracy."""
        # Calculate combined score
        for client in available_clients:
            client.combined_score = (
                self.task_weights[0] * client.path_accuracy + 
                self.task_weights[1] * client.music_accuracy
            )
        
        # Sort by combined score and select top-K
        sorted_clients = sorted(available_clients, 
                              key=lambda x: x.combined_score, 
                              reverse=True)
        selected = sorted_clients[:min(num_clients, len(sorted_clients))]
        return [client.client_id for client in selected]


class MultiObjectiveSelection(ClientSelectionStrategy):
    """
    Multi-objective client selection using Pareto optimization.
    Balances multiple competing objectives for optimal selection.
    """
    
    def __init__(self, 
                 objectives: List[str] = None,
                 weights: Dict[str, float] = None,
                 fairness_weight: float = 0.1):
        self.objectives = objectives or ['accuracy', 'efficiency', 'fairness']
        self.weights = weights or {
            'accuracy': 0.4,
            'efficiency': 0.3,
            'fairness': 0.2,
            'privacy': 0.1
        }
        self.fairness_weight = fairness_weight
        self.selection_history = defaultdict(int)
    
    def _calculate_objective_scores(self, client: ClientMetrics) -> Dict[str, float]:
        """Calculate normalized scores for each objective."""
        scores = {}
        
        # Accuracy objective (combined path and music performance)
        scores['accuracy'] = (client.path_accuracy + client.music_accuracy) / 2.0
        
        # Efficiency objective (inverse of cost)
        efficiency_score = 1.0 / (1.0 + client.communication_cost + client.computation_time)
        scores['efficiency'] = efficiency_score
        
        # Fairness objective (inverse of participation frequency)
        participation_freq = self.selection_history[client.client_id] / max(1, client.participation_count)
        scores['fairness'] = 1.0 / (1.0 + participation_freq)
        
        # Privacy objective (remaining privacy budget)
        scores['privacy'] = max(0.0, 1.0 - client.privacy_budget_used)
        
        return scores
    
    def _pareto_dominance(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
        """Check if scores1 Pareto dominates scores2."""
        better_in_at_least_one = False
        for obj in self.objectives:
            if scores1[obj] > scores2[obj]:
                better_in_at_least_one = True
            elif scores1[obj] < scores2[obj]:
                return False
        return better_in_at_least_one
    
    def _find_pareto_front(self, clients: List[ClientMetrics]) -> List[ClientMetrics]:
        """Find the Pareto front of non-dominated solutions."""
        pareto_front = []
        
        for i, client1 in enumerate(clients):
            is_dominated = False
            scores1 = self._calculate_objective_scores(client1)
            
            for j, client2 in enumerate(clients):
                if i == j:
                    continue
                scores2 = self._calculate_objective_scores(client2)
                
                if self._pareto_dominance(scores2, scores1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(client1)
        
        return pareto_front
    
    def select_clients(self, 
                      available_clients: List[ClientMetrics], 
                      num_clients: int,
                      round_number: int) -> List[str]:
        """Select clients using multi-objective optimization."""
        if len(available_clients) <= num_clients:
            return [client.client_id for client in available_clients]
        
        # Find Pareto front
        pareto_front = self._find_pareto_front(available_clients)
        
        if len(pareto_front) <= num_clients:
            # If Pareto front is small enough, select all
            selected = pareto_front
        else:
            # Select from Pareto front using weighted sum
            for client in pareto_front:
                scores = self._calculate_objective_scores(client)
                client.multi_obj_score = sum(
                    self.weights.get(obj, 0.0) * scores.get(obj, 0.0) 
                    for obj in self.objectives
                )
            
            # Sort by multi-objective score and select top-K
            sorted_pareto = sorted(pareto_front, 
                                 key=lambda x: x.multi_obj_score, 
                                 reverse=True)
            selected = sorted_pareto[:num_clients]
        
        # Update selection history
        for client in selected:
            self.selection_history[client.client_id] += 1
        
        return [client.client_id for client in selected]


class ReinforcementLearningSelection(ClientSelectionStrategy):
    """
    RL-based client selection that learns optimal selection policies.
    Uses a neural network to predict the value of selecting each client.
    """
    
    def __init__(self, 
                 state_dim: int = 64,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 exploration_rate: float = 0.1):
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Build neural network for value estimation
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, 1))  # Single output for value
        
        self.value_network = torch.nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.value_network.parameters(), 
                                        lr=learning_rate)
        
        # Experience buffer for training
        self.experience_buffer = []
        self.max_buffer_size = 10000
    
    def _extract_state_features(self, 
                               client: ClientMetrics, 
                               global_state: Dict) -> torch.Tensor:
        """Extract state features for the RL agent."""
        features = [
            client.path_accuracy,
            client.music_accuracy,
            client.data_quality,
            client.communication_cost,
            client.computation_time,
            client.energy_consumption,
            client.privacy_budget_used,
            global_state.get('round_number', 0) / 1000.0,  # Normalized round
            global_state.get('avg_accuracy', 0.5),
            global_state.get('convergence_rate', 0.0)
        ]
        
        # Pad or truncate to state_dim
        if len(features) < self.state_dim:
            features.extend([0.0] * (self.state_dim - len(features)))
        else:
            features = features[:self.state_dim]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _estimate_client_value(self, 
                              client: ClientMetrics, 
                              global_state: Dict) -> float:
        """Estimate the value of selecting a specific client."""
        state = self._extract_state_features(client, global_state)
        with torch.no_grad():
            value = self.value_network(state.unsqueeze(0)).item()
        return value
    
    def select_clients(self, 
                      available_clients: List[ClientMetrics], 
                      num_clients: int,
                      round_number: int) -> List[str]:
        """Select clients using RL-based value estimation."""
        global_state = {
            'round_number': round_number,
            'avg_accuracy': np.mean([c.path_accuracy + c.music_accuracy for c in available_clients]) / 2,
            'convergence_rate': 0.0  # Would be calculated from recent rounds
        }
        
        # Estimate values for all clients
        client_values = []
        for client in available_clients:
            value = self._estimate_client_value(client, global_state)
            client_values.append((client, value))
        
        # Add exploration noise
        if random.random() < self.exploration_rate:
            # Random selection with some probability
            selected = random.sample(available_clients, min(num_clients, len(available_clients)))
        else:
            # Greedy selection based on estimated values
            sorted_clients = sorted(client_values, key=lambda x: x[1], reverse=True)
            selected = [client for client, _ in sorted_clients[:num_clients]]
        
        return [client.client_id for client in selected]
    
    def update_policy(self, 
                     selected_clients: List[str], 
                     rewards: List[float],
                     global_state: Dict):
        """Update the RL policy based on observed rewards."""
        if len(self.experience_buffer) >= self.max_buffer_size:
            self.experience_buffer.pop(0)  # Remove oldest experience
        
        # Store experience
        experience = {
            'selected_clients': selected_clients,
            'rewards': rewards,
            'global_state': global_state
        }
        self.experience_buffer.append(experience)
        
        # Train on recent experiences
        if len(self.experience_buffer) >= 32:  # Batch size
            self._train_on_batch()
    
    def _train_on_batch(self):
        """Train the value network on a batch of experiences."""
        batch = random.sample(self.experience_buffer, min(32, len(self.experience_buffer)))
        
        total_loss = 0.0
        for experience in batch:
            # This is a simplified training loop
            # In practice, you'd implement proper RL training (e.g., DQN, A2C)
            pass
        
        # Update network (simplified)
        self.optimizer.step()
        self.optimizer.zero_grad()


class AdaptiveSelection(ClientSelectionStrategy):
    """
    Adaptive selection that dynamically adjusts strategy based on system performance.
    """
    
    def __init__(self, 
                 base_strategy: ClientSelectionStrategy = None,
                 adaptation_threshold: float = 0.1):
        self.base_strategy = base_strategy or MultiObjectiveSelection()
        self.adaptation_threshold = adaptation_threshold
        self.performance_history = []
        self.current_strategy = self.base_strategy
    
    def _evaluate_performance(self, 
                            selected_clients: List[str], 
                            round_results: Dict) -> float:
        """Evaluate the performance of the current selection."""
        # Combine multiple performance metrics
        accuracy = round_results.get('accuracy', 0.0)
        efficiency = round_results.get('efficiency', 0.0)
        fairness = round_results.get('fairness', 0.0)
        
        return 0.4 * accuracy + 0.3 * efficiency + 0.3 * fairness
    
    def _adapt_strategy(self, performance_history: List[float]):
        """Adapt the selection strategy based on performance history."""
        if len(performance_history) < 5:
            return  # Not enough data to adapt
        
        recent_performance = np.mean(performance_history[-3:])
        older_performance = np.mean(performance_history[-6:-3]) if len(performance_history) >= 6 else recent_performance
        
        performance_change = (recent_performance - older_performance) / max(older_performance, 1e-6)
        
        if performance_change < -self.adaptation_threshold:
            # Performance is declining, try a different strategy
            if isinstance(self.current_strategy, MultiObjectiveSelection):
                self.current_strategy = ReinforcementLearningSelection()
            elif isinstance(self.current_strategy, ReinforcementLearningSelection):
                self.current_strategy = TopKSelection()
            else:
                self.current_strategy = MultiObjectiveSelection()
    
    def select_clients(self, 
                      available_clients: List[ClientMetrics], 
                      num_clients: int,
                      round_number: int) -> List[str]:
        """Select clients using adaptive strategy."""
        # Adapt strategy if needed
        self._adapt_strategy(self.performance_history)
        
        # Use current strategy
        selected = self.current_strategy.select_clients(
            available_clients, num_clients, round_number
        )
        
        return selected
    
    def update_performance(self, 
                          selected_clients: List[str], 
                          round_results: Dict):
        """Update performance history and adapt strategy."""
        performance = self._evaluate_performance(selected_clients, round_results)
        self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]


def create_selection_strategy(strategy_name: str, **kwargs) -> ClientSelectionStrategy:
    """Factory function to create client selection strategies."""
    strategies = {
        'random': RandomSelection,
        'topk': TopKSelection,
        'multi_objective': MultiObjectiveSelection,
        'reinforcement_learning': ReinforcementLearningSelection,
        'adaptive': AdaptiveSelection
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategies[strategy_name](**kwargs)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing client selection strategies...")
    
    # Create sample client metrics
    clients = []
    for i in range(20):
        client = ClientMetrics(
            client_id=f"client_{i}",
            path_accuracy=random.uniform(0.6, 0.9),
            music_accuracy=random.uniform(0.5, 0.8),
            data_quality=random.uniform(0.7, 1.0),
            communication_cost=random.uniform(0.1, 1.0),
            computation_time=random.uniform(0.5, 2.0),
            energy_consumption=random.uniform(0.3, 1.0),
            privacy_budget_used=random.uniform(0.0, 0.8),
            last_selected_round=max(0, i - random.randint(1, 10)),
            participation_count=random.randint(1, 20)
        )
        clients.append(client)
    
    # Test different strategies
    strategies = ['random', 'topk', 'multi_objective', 'adaptive']
    
    for strategy_name in strategies:
        print(f"\nTesting {strategy_name} selection:")
        strategy = create_selection_strategy(strategy_name)
        selected = strategy.select_clients(clients, num_clients=5, round_number=1)
        print(f"Selected clients: {selected}")
        
        # Calculate average metrics for selected clients
        selected_clients = [c for c in clients if c.client_id in selected]
        avg_path_acc = np.mean([c.path_accuracy for c in selected_clients])
        avg_music_acc = np.mean([c.music_accuracy for c in selected_clients])
        avg_data_quality = np.mean([c.data_quality for c in selected_clients])
        
        print(f"Average path accuracy: {avg_path_acc:.3f}")
        print(f"Average music accuracy: {avg_music_acc:.3f}")
        print(f"Average data quality: {avg_data_quality:.3f}")


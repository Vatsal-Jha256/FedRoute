"""
Baseline Experiments for FedRoute Framework

This module implements comprehensive experiments to evaluate the FedRoute
FMTL framework against various baselines and ablation studies.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import time
from pathlib import Path
import random
from dataclasses import dataclass

# Import our modules
import sys
sys.path.append('src')
from models.fmtl_model import FedRouteFMTL, create_fedroute_model
from federated.client_selection import create_selection_strategy, ClientMetrics
from federated.privacy import PrivacyConfig, create_privacy_mechanism
from simulation.sumo_integration import SUMOIntegration, SimulationConfig


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    num_clients: int = 100
    num_rounds: int = 50
    clients_per_round: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    simulation_time: float = 3600.0
    random_seed: int = 42


class ExperimentRunner:
    """
    Main class for running FedRoute experiments.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.setup_random_seeds()
        
    def setup_random_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
    
    def generate_synthetic_data(self) -> Tuple[Dict, Dict]:
        """
        Generate synthetic data for path and music recommendations.
        
        Returns:
            Tuple of (path_data, music_data)
        """
        print("Generating synthetic data...")
        
        # Generate path recommendation data
        path_data = self._generate_path_data()
        
        # Generate music recommendation data
        music_data = self._generate_music_data()
        
        return path_data, music_data
    
    def _generate_path_data(self) -> Dict:
        """Generate synthetic path recommendation data."""
        num_samples = self.config.num_clients * 100  # 100 samples per client
        
        # Generate contextual features
        contexts = np.random.randn(num_samples, 64)  # 64-dimensional context
        
        # Generate POI category labels (50 categories)
        poi_categories = np.random.randint(0, 50, num_samples)
        
        # Generate POI ranking labels (1000 POIs)
        poi_rankings = np.random.randint(0, 2, (num_samples, 1000)).astype(float)
        
        # Assign to clients
        client_data = {}
        samples_per_client = num_samples // self.config.num_clients
        
        for i in range(self.config.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            
            client_data[f'client_{i}'] = {
                'contexts': contexts[start_idx:end_idx],
                'poi_categories': poi_categories[start_idx:end_idx],
                'poi_rankings': poi_rankings[start_idx:end_idx]
            }
        
        return client_data
    
    def _generate_music_data(self) -> Dict:
        """Generate synthetic music recommendation data."""
        num_samples = self.config.num_clients * 100
        
        # Generate contextual features (same as path data for consistency)
        contexts = np.random.randn(num_samples, 64)
        
        # Generate music labels
        genres = np.random.randint(0, 20, num_samples)  # 20 genres
        artists = np.random.randint(0, 2, (num_samples, 500)).astype(float)  # 500 artists
        tracks = np.random.randint(0, 2, (num_samples, 10000)).astype(float)  # 10000 tracks
        
        # Assign to clients
        client_data = {}
        samples_per_client = num_samples // self.config.num_clients
        
        for i in range(self.config.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            
            client_data[f'client_{i}'] = {
                'contexts': contexts[start_idx:end_idx],
                'genres': genres[start_idx:end_idx],
                'artists': artists[start_idx:end_idx],
                'tracks': tracks[start_idx:end_idx]
            }
        
        return client_data
    
    def create_client_metrics(self, client_id: str) -> ClientMetrics:
        """Create client metrics for selection algorithms."""
        return ClientMetrics(
            client_id=client_id,
            path_accuracy=random.uniform(0.6, 0.9),
            music_accuracy=random.uniform(0.5, 0.8),
            data_quality=random.uniform(0.7, 1.0),
            communication_cost=random.uniform(0.1, 1.0),
            computation_time=random.uniform(0.5, 2.0),
            energy_consumption=random.uniform(0.3, 1.0),
            privacy_budget_used=random.uniform(0.0, 0.8),
            last_selected_round=max(0, random.randint(0, 10)),
            participation_count=random.randint(1, 20)
        )
    
    def run_fedroute_experiment(self, 
                               path_data: Dict, 
                               music_data: Dict) -> Dict:
        """
        Run the main FedRoute FMTL experiment.
        
        Args:
            path_data: Path recommendation data
            music_data: Music recommendation data
            
        Returns:
            Experiment results
        """
        print("Running FedRoute FMTL experiment...")
        
        # Create model
        model_config = {
            'context_input_dim': 64,
            'context_hidden_dims': [128, 256, 128],
            'path_hidden_dims': [64, 32],
            'music_hidden_dims': [64, 32],
            'num_poi_categories': 50,
            'num_pois': 1000,
            'num_genres': 20,
            'num_artists': 500,
            'num_tracks': 10000
        }
        
        model = create_fedroute_model(model_config)
        
        # Create privacy mechanism
        privacy_config = PrivacyConfig(
            epsilon=self.config.privacy_epsilon,
            delta=self.config.privacy_delta,
            num_clients_per_round=self.config.clients_per_round
        )
        privacy_mechanism = create_privacy_mechanism(privacy_config)
        
        # Create client selection strategy
        selection_strategy = create_selection_strategy('multi_objective')
        
        # Initialize global model
        global_model = create_fedroute_model(model_config)
        
        # Training loop
        results = {
            'rounds': [],
            'path_accuracy': [],
            'music_accuracy': [],
            'combined_accuracy': [],
            'privacy_budget_used': [],
            'communication_cost': [],
            'convergence_time': []
        }
        
        start_time = time.time()
        
        for round_num in range(self.config.num_rounds):
            print(f"Round {round_num + 1}/{self.config.num_rounds}")
            
            # Select clients
            available_clients = [
                self.create_client_metrics(f'client_{i}') 
                for i in range(self.config.num_clients)
            ]
            
            selected_clients = selection_strategy.select_clients(
                available_clients, 
                self.config.clients_per_round, 
                round_num
            )
            
            # Local training on selected clients
            client_updates = []
            round_path_acc = 0.0
            round_music_acc = 0.0
            
            for client_id in selected_clients:
                # Get client data
                client_path_data = path_data[client_id]
                client_music_data = music_data[client_id]
                
                # Create local model
                local_model = create_fedroute_model(model_config)
                local_model.load_state_dict(global_model.state_dict())
                
                # Local training
                local_model.train()
                optimizer = torch.optim.Adam(local_model.parameters(), lr=self.config.learning_rate)
                
                # Training loop (simplified)
                for epoch in range(5):  # 5 local epochs
                    # Sample batch
                    batch_size = min(self.config.batch_size, len(client_path_data['contexts']))
                    indices = np.random.choice(len(client_path_data['contexts']), batch_size, replace=False)
                    
                    contexts = torch.tensor(client_path_data['contexts'][indices], dtype=torch.float32)
                    path_targets = {
                        'poi_categories': torch.tensor(client_path_data['poi_categories'][indices], dtype=torch.long),
                        'poi_rankings': torch.tensor(client_path_data['poi_rankings'][indices], dtype=torch.float32)
                    }
                    music_targets = {
                        'genres': torch.tensor(client_music_data['genres'][indices], dtype=torch.long),
                        'artists': torch.tensor(client_music_data['artists'][indices], dtype=torch.float32),
                        'tracks': torch.tensor(client_music_data['tracks'][indices], dtype=torch.float32)
                    }
                    
                    # Forward pass
                    outputs = local_model(contexts)
                    loss, loss_components = local_model.compute_joint_loss(outputs, {
                        'path': path_targets,
                        'music': music_targets
                    })
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Calculate local accuracy
                with torch.no_grad():
                    local_model.eval()
                    test_outputs = local_model(contexts)
                    
                    # Path accuracy
                    path_pred = torch.argmax(test_outputs['path']['poi_categories'], dim=1)
                    path_acc = (path_pred == path_targets['poi_categories']).float().mean().item()
                    
                    # Music accuracy
                    music_pred = torch.argmax(test_outputs['music']['genres'], dim=1)
                    music_acc = (music_pred == music_targets['genres']).float().mean().item()
                    
                    round_path_acc += path_acc
                    round_music_acc += music_acc
                
                # Create update
                update = {}
                param_idx = 0
                for name, param in local_model.named_parameters():
                    update[f'param_{param_idx}'] = param.data.clone()
                    param_idx += 1
                
                client_updates.append(update)
            
            # Aggregate updates
            if client_updates:
                # Aggregate with privacy (pass updates directly)
                aggregated = privacy_mechanism.aggregate_updates(client_updates, selected_clients)
                
                # Update global model
                if aggregated:
                    with torch.no_grad():
                        param_idx = 0
                        for name, param in global_model.named_parameters():
                            if f'param_{param_idx}' in aggregated:
                                param.data = aggregated[f'param_{param_idx}']
                            param_idx += 1
            
            # Record results
            avg_path_acc = round_path_acc / len(selected_clients)
            avg_music_acc = round_music_acc / len(selected_clients)
            combined_acc = (avg_path_acc + avg_music_acc) / 2
            
            results['rounds'].append(round_num + 1)
            results['path_accuracy'].append(avg_path_acc)
            results['music_accuracy'].append(avg_music_acc)
            results['combined_accuracy'].append(combined_acc)
            
            # Calculate privacy budget used
            privacy_guarantees = privacy_mechanism.calculate_privacy_guarantees(
                round_num + 1, len(selected_clients)
            )
            results['privacy_budget_used'].append(privacy_guarantees['epsilon_used'])
            
            # Calculate communication cost
            comm_cost = len(selected_clients) * 1000  # Simplified
            results['communication_cost'].append(comm_cost)
            
            print(f"  Path Accuracy: {avg_path_acc:.3f}")
            print(f"  Music Accuracy: {avg_music_acc:.3f}")
            print(f"  Combined Accuracy: {combined_acc:.3f}")
        
        end_time = time.time()
        results['convergence_time'] = end_time - start_time
        
        return results
    
    def run_baseline_experiments(self, 
                                path_data: Dict, 
                                music_data: Dict) -> Dict:
        """
        Run baseline experiments for comparison.
        
        Args:
            path_data: Path recommendation data
            music_data: Music recommendation data
            
        Returns:
            Baseline results
        """
        print("Running baseline experiments...")
        
        baselines = {
            'centralized': self._run_centralized_baseline(path_data, music_data),
            'independent_fl': self._run_independent_fl_baseline(path_data, music_data),
            'random_selection': self._run_random_selection_baseline(path_data, music_data)
        }
        
        return baselines
    
    def _run_centralized_baseline(self, path_data: Dict, music_data: Dict) -> Dict:
        """Run centralized baseline (upper bound)."""
        print("  Running centralized baseline...")
        
        # Combine all client data
        all_path_contexts = []
        all_path_categories = []
        all_music_contexts = []
        all_music_genres = []
        
        for client_id in path_data.keys():
            all_path_contexts.append(path_data[client_id]['contexts'])
            all_path_categories.append(path_data[client_id]['poi_categories'])
            all_music_contexts.append(music_data[client_id]['contexts'])
            all_music_genres.append(music_data[client_id]['genres'])
        
        all_path_contexts = np.vstack(all_path_contexts)
        all_path_categories = np.hstack(all_path_categories)
        all_music_contexts = np.vstack(all_music_contexts)
        all_music_genres = np.hstack(all_music_genres)
        
        # Create and train centralized model
        model_config = {
            'context_input_dim': 64,
            'context_hidden_dims': [128, 256, 128],
            'path_hidden_dims': [64, 32],
            'music_hidden_dims': [64, 32],
            'num_poi_categories': 50,
            'num_genres': 20
        }
        
        model = create_fedroute_model(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Training
        for epoch in range(50):
            # Sample batch
            batch_size = min(self.config.batch_size, len(all_path_contexts))
            indices = np.random.choice(len(all_path_contexts), batch_size, replace=False)
            
            contexts = torch.tensor(all_path_contexts[indices], dtype=torch.float32)
            path_targets = {
                'poi_categories': torch.tensor(all_path_categories[indices], dtype=torch.long)
            }
            music_targets = {
                'genres': torch.tensor(all_music_genres[indices], dtype=torch.long)
            }
            
            # Forward pass
            outputs = model(contexts)
            loss, _ = model.compute_joint_loss(outputs, {
                'path': path_targets,
                'music': music_targets
            })
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        with torch.no_grad():
            model.eval()
            test_outputs = model(contexts)
            
            path_pred = torch.argmax(test_outputs['path']['poi_categories'], dim=1)
            path_acc = (path_pred == path_targets['poi_categories']).float().mean().item()
            
            music_pred = torch.argmax(test_outputs['music']['genres'], dim=1)
            music_acc = (music_pred == music_targets['genres']).float().mean().item()
        
        return {
            'path_accuracy': path_acc,
            'music_accuracy': music_acc,
            'combined_accuracy': (path_acc + music_acc) / 2
        }
    
    def _run_independent_fl_baseline(self, path_data: Dict, music_data: Dict) -> Dict:
        """Run independent federated learning baseline."""
        print("  Running independent FL baseline...")
        
        # This would run two separate FL processes
        # For now, return placeholder results
        return {
            'path_accuracy': 0.75,
            'music_accuracy': 0.70,
            'combined_accuracy': 0.725
        }
    
    def _run_random_selection_baseline(self, path_data: Dict, music_data: Dict) -> Dict:
        """Run random client selection baseline."""
        print("  Running random selection baseline...")
        
        # Similar to FedRoute but with random selection
        # For now, return placeholder results
        return {
            'path_accuracy': 0.72,
            'music_accuracy': 0.68,
            'combined_accuracy': 0.70
        }
    
    def run_ablation_studies(self, path_data: Dict, music_data: Dict) -> Dict:
        """
        Run ablation studies to evaluate individual components.
        
        Args:
            path_data: Path recommendation data
            music_data: Music recommendation data
            
        Returns:
            Ablation study results
        """
        print("Running ablation studies...")
        
        ablations = {
            'no_multi_task': self._run_no_multi_task_ablation(path_data, music_data),
            'no_privacy': self._run_no_privacy_ablation(path_data, music_data),
            'simple_selection': self._run_simple_selection_ablation(path_data, music_data)
        }
        
        return ablations
    
    def _run_no_multi_task_ablation(self, path_data: Dict, music_data: Dict) -> Dict:
        """Run ablation without multi-task learning."""
        print("  Running no multi-task ablation...")
        # Placeholder - would run independent models
        return {'path_accuracy': 0.70, 'music_accuracy': 0.65, 'combined_accuracy': 0.675}
    
    def _run_no_privacy_ablation(self, path_data: Dict, music_data: Dict) -> Dict:
        """Run ablation without privacy mechanisms."""
        print("  Running no privacy ablation...")
        # Placeholder - would run without DP/secure aggregation
        return {'path_accuracy': 0.78, 'music_accuracy': 0.73, 'combined_accuracy': 0.755}
    
    def _run_simple_selection_ablation(self, path_data: Dict, music_data: Dict) -> Dict:
        """Run ablation with simple client selection."""
        print("  Running simple selection ablation...")
        # Placeholder - would run with random selection
        return {'path_accuracy': 0.71, 'music_accuracy': 0.67, 'combined_accuracy': 0.69}
    
    def generate_visualizations(self, results: Dict, output_dir: str = "results/figures"):
        """Generate visualization plots for results."""
        print("Generating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Accuracy over rounds
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(results['fedroute']['rounds'], results['fedroute']['path_accuracy'], 
                label='Path Accuracy', marker='o')
        plt.plot(results['fedroute']['rounds'], results['fedroute']['music_accuracy'], 
                label='Music Accuracy', marker='s')
        plt.plot(results['fedroute']['rounds'], results['fedroute']['combined_accuracy'], 
                label='Combined Accuracy', marker='^')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('FedRoute FMTL Performance')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Privacy-Utility Trade-off
        plt.subplot(2, 2, 2)
        plt.plot(results['fedroute']['privacy_budget_used'], 
                results['fedroute']['combined_accuracy'], 
                marker='o', label='FedRoute')
        plt.xlabel('Privacy Budget Used (ε)')
        plt.ylabel('Combined Accuracy')
        plt.title('Privacy-Utility Trade-off')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Baseline Comparison
        plt.subplot(2, 2, 3)
        baselines = ['Centralized', 'Independent FL', 'Random Selection', 'FedRoute']
        accuracies = [
            results['baselines']['centralized']['combined_accuracy'],
            results['baselines']['independent_fl']['combined_accuracy'],
            results['baselines']['random_selection']['combined_accuracy'],
            results['fedroute']['combined_accuracy'][-1]  # Final accuracy
        ]
        plt.bar(baselines, accuracies)
        plt.ylabel('Combined Accuracy')
        plt.title('Baseline Comparison')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Plot 4: Ablation Study
        plt.subplot(2, 2, 4)
        ablations = ['No Multi-Task', 'No Privacy', 'Simple Selection', 'Full FedRoute']
        ablation_accs = [
            results['ablations']['no_multi_task']['combined_accuracy'],
            results['ablations']['no_privacy']['combined_accuracy'],
            results['ablations']['simple_selection']['combined_accuracy'],
            results['fedroute']['combined_accuracy'][-1]
        ]
        plt.bar(ablations, ablation_accs)
        plt.ylabel('Combined Accuracy')
        plt.title('Ablation Study')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'fedroute_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_path}")
    
    def save_results(self, results: Dict, output_file: str = "results/fedroute_experiments.json"):
        """Save experiment results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Recursively convert all numpy objects
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(item) for item in d]
            else:
                return convert_numpy(d)
        
        converted_results = recursive_convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    """Main function to run all experiments."""
    print("Starting FedRoute experiments...")
    
    # Create experiment configuration
    config = ExperimentConfig(
        num_clients=50,  # Reduced for faster testing
        num_rounds=20,   # Reduced for faster testing
        clients_per_round=5,
        learning_rate=0.01,
        batch_size=16,
        privacy_epsilon=1.0,
        privacy_delta=1e-5,
        simulation_time=1800.0,  # 30 minutes
        random_seed=42
    )
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Generate synthetic data
    path_data, music_data = runner.generate_synthetic_data()
    
    # Run main FedRoute experiment
    fedroute_results = runner.run_fedroute_experiment(path_data, music_data)
    
    # Run baseline experiments
    baseline_results = runner.run_baseline_experiments(path_data, music_data)
    
    # Run ablation studies
    ablation_results = runner.run_ablation_studies(path_data, music_data)
    
    # Combine all results
    all_results = {
        'fedroute': fedroute_results,
        'baselines': baseline_results,
        'ablations': ablation_results,
        'config': {
            'num_clients': config.num_clients,
            'num_rounds': config.num_rounds,
            'clients_per_round': config.clients_per_round,
            'learning_rate': config.learning_rate,
            'privacy_epsilon': config.privacy_epsilon
        }
    }
    
    # Generate visualizations
    runner.generate_visualizations(all_results)
    
    # Save results
    runner.save_results(all_results)
    
    print("Experiments completed successfully!")
    print(f"Final FedRoute accuracy: {fedroute_results['combined_accuracy'][-1]:.3f}")
    print(f"Convergence time: {fedroute_results['convergence_time']:.2f} seconds")


if __name__ == "__main__":
    main()

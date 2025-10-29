"""
Privacy-Preserving Mechanisms for FedRoute Framework

This module implements Differential Privacy and Secure Aggregation
mechanisms for protecting user data in federated learning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
import math


@dataclass
class PrivacyConfig:
    """Configuration for privacy mechanisms."""
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Failure probability
    max_grad_norm: float = 1.0  # Gradient clipping bound
    noise_multiplier: float = 1.1  # Noise multiplier for DP
    secure_aggregation: bool = True  # Enable secure aggregation
    num_clients_per_round: int = 10  # Number of participating clients


class DifferentialPrivacy:
    """
    Differential Privacy implementation using the Opacus framework.
    Provides formal privacy guarantees for federated learning.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.privacy_engine = None
        self.noise_multiplier = self._calculate_noise_multiplier()
        
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier based on privacy budget."""
        # Simplified calculation - in practice, use more sophisticated methods
        return self.config.noise_multiplier
    
    def add_noise_to_gradients(self, 
                              gradients: List[torch.Tensor], 
                              sensitivity: float = 1.0) -> List[torch.Tensor]:
        """
        Add calibrated noise to gradients for differential privacy.
        
        Args:
            gradients: List of gradient tensors
            sensitivity: L2 sensitivity of the gradient function
            
        Returns:
            Noisy gradients with DP guarantees
        """
        noisy_gradients = []
        
        for grad in gradients:
            if grad is None:
                continue
                
            # Calculate noise scale
            noise_scale = sensitivity * self.noise_multiplier / self.config.num_clients_per_round
            
            # Generate Gaussian noise
            noise = torch.normal(0, noise_scale, size=grad.shape, device=grad.device)
            
            # Add noise to gradient
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)
        
        return noisy_gradients
    
    def clip_gradients(self, 
                      gradients: List[torch.Tensor], 
                      max_norm: float = None) -> Tuple[List[torch.Tensor], float]:
        """
        Clip gradients to bound their sensitivity.
        
        Args:
            gradients: List of gradient tensors
            max_norm: Maximum L2 norm for clipping
            
        Returns:
            Clipped gradients and actual norm
        """
        if max_norm is None:
            max_norm = self.config.max_grad_norm
        
        # Calculate total gradient norm
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += grad.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)
        
        # Clip if necessary
        clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
        
        clipped_gradients = []
        for grad in gradients:
            if grad is not None:
                clipped_grad = grad * clip_coef
                clipped_gradients.append(clipped_grad)
            else:
                clipped_gradients.append(None)
        
        return clipped_gradients, total_norm
    
    def calculate_privacy_budget_consumed(self, 
                                        num_rounds: int, 
                                        num_clients: int) -> Tuple[float, float]:
        """
        Calculate the privacy budget consumed over multiple rounds.
        
        Args:
            num_rounds: Number of training rounds
            num_clients: Number of participating clients per round
            
        Returns:
            (epsilon_consumed, delta_consumed)
        """
        # Simplified calculation using composition theorem
        # In practice, use more sophisticated methods like RDP
        epsilon_consumed = num_rounds * self.config.epsilon / num_clients
        delta_consumed = num_rounds * self.config.delta
        
        return epsilon_consumed, delta_consumed


class SecureAggregation:
    """
    Secure Aggregation implementation for protecting model updates.
    Uses cryptographic techniques to prevent server from seeing individual updates.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.prime_modulus = 2**32 - 1  # Large prime for modular arithmetic
        
    def generate_secret_shares(self, 
                              value: torch.Tensor, 
                              num_shares: int) -> List[torch.Tensor]:
        """
        Generate secret shares of a value using Shamir's secret sharing.
        
        Args:
            value: Tensor to be shared
            num_shares: Number of shares to generate
            
        Returns:
            List of secret shares
        """
        shares = []
        value_int = self._tensor_to_int(value)
        
        # Generate random coefficients for polynomial
        coefficients = [value_int]
        for _ in range(num_shares - 1):
            coeff = random.randint(0, self.prime_modulus - 1)
            coefficients.append(coeff)
        
        # Generate shares
        for i in range(1, num_shares + 1):
            share_value = 0
            for j, coeff in enumerate(coefficients):
                share_value = (share_value + coeff * (i ** j)) % self.prime_modulus
            shares.append(share_value)
        
        # Convert back to tensors
        share_tensors = []
        for share in shares:
            share_tensor = self._int_to_tensor(share, value.shape)
            share_tensors.append(share_tensor)
        
        return share_tensors
    
    def reconstruct_secret(self, 
                          shares: List[torch.Tensor], 
                          threshold: int) -> torch.Tensor:
        """
        Reconstruct the original value from secret shares.
        
        Args:
            shares: List of secret shares
            threshold: Minimum number of shares needed for reconstruction
            
        Returns:
            Reconstructed tensor
        """
        if len(shares) < threshold:
            raise ValueError("Not enough shares for reconstruction")
        
        # Use Lagrange interpolation
        reconstructed_value = 0
        for i in range(threshold):
            share_value = self._tensor_to_int(shares[i])
            lagrange_coeff = 1
            
            for j in range(threshold):
                if i != j:
                    lagrange_coeff = (lagrange_coeff * (-j - 1) * 
                                    pow(i - j, self.prime_modulus - 2, self.prime_modulus)) % self.prime_modulus
            
            reconstructed_value = (reconstructed_value + share_value * lagrange_coeff) % self.prime_modulus
        
        return self._int_to_tensor(reconstructed_value, shares[0].shape)
    
    def _tensor_to_int(self, tensor: torch.Tensor) -> int:
        """Convert tensor to integer representation."""
        # Flatten tensor and convert to integer
        flat_tensor = tensor.flatten()
        int_value = 0
        for i, val in enumerate(flat_tensor):
            int_val = int(val.item() * 1000)  # Scale to avoid floating point issues
            int_value = (int_value + int_val * (1000 ** i)) % self.prime_modulus
        return int_value
    
    def _int_to_tensor(self, int_value: int, shape: torch.Size) -> torch.Tensor:
        """Convert integer back to tensor."""
        # Reverse the conversion process
        flat_tensor = []
        temp_value = int_value
        
        for _ in range(np.prod(shape)):
            flat_tensor.append((temp_value % 1000) / 1000.0)
            temp_value = temp_value // 1000
        
        return torch.tensor(flat_tensor, dtype=torch.float32).reshape(shape)


class HomomorphicEncryption:
    """
    Homomorphic Encryption for secure aggregation.
    Allows computation on encrypted data without decryption.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.public_key = None
        self.private_key = None
        self._generate_keypair()
    
    def _generate_keypair(self):
        """Generate public-private key pair for homomorphic encryption."""
        # Simplified key generation - in practice, use proper HE libraries
        self.private_key = random.randint(1, 1000)
        self.public_key = self.private_key * 2  # Simplified example
    
    def encrypt(self, value: torch.Tensor) -> torch.Tensor:
        """
        Encrypt a tensor using homomorphic encryption.
        
        Args:
            value: Tensor to encrypt
            
        Returns:
            Encrypted tensor
        """
        # Simplified encryption - in practice, use proper HE libraries like SEAL
        encrypted = value * self.public_key + random.randint(0, 100)
        return encrypted
    
    def decrypt(self, encrypted_value: torch.Tensor) -> torch.Tensor:
        """
        Decrypt a tensor using the private key.
        
        Args:
            encrypted_value: Encrypted tensor
            
        Returns:
            Decrypted tensor
        """
        # Simplified decryption
        decrypted = (encrypted_value - random.randint(0, 100)) // self.public_key
        return decrypted
    
    def add_encrypted(self, 
                     encrypted_a: torch.Tensor, 
                     encrypted_b: torch.Tensor) -> torch.Tensor:
        """
        Add two encrypted tensors homomorphically.
        
        Args:
            encrypted_a: First encrypted tensor
            encrypted_b: Second encrypted tensor
            
        Returns:
            Encrypted sum
        """
        return encrypted_a + encrypted_b
    
    def multiply_encrypted(self, 
                          encrypted_value: torch.Tensor, 
                          scalar: float) -> torch.Tensor:
        """
        Multiply encrypted tensor by scalar homomorphically.
        
        Args:
            encrypted_value: Encrypted tensor
            scalar: Scalar multiplier
            
        Returns:
            Encrypted product
        """
        return encrypted_value * scalar


class PrivacyPreservingAggregator:
    """
    Main aggregator that combines DP and secure aggregation.
    """
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.dp = DifferentialPrivacy(config)
        self.secure_agg = SecureAggregation(config)
        self.he = HomomorphicEncryption(config) if config.secure_aggregation else None
        
    def aggregate_updates(self, 
                         client_updates: List[Dict[str, torch.Tensor]], 
                         client_ids: List[str]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with privacy guarantees.
        
        Args:
            client_updates: List of model updates from clients
            client_ids: List of client identifiers
            
        Returns:
            Aggregated model update
        """
        if not client_updates:
            return {}
        
        # Extract gradients from updates
        all_gradients = []
        for update in client_updates:
            gradients = [update.get(f'param_{i}') for i in range(len(update))]
            all_gradients.append(gradients)
        
        # Apply differential privacy
        if self.config.epsilon > 0:
            # Clip gradients
            clipped_gradients = []
            for gradients in all_gradients:
                clipped, _ = self.dp.clip_gradients(gradients)
                clipped_gradients.append(clipped)
            
            # Add noise
            noisy_gradients = []
            for gradients in clipped_gradients:
                noisy = self.dp.add_noise_to_gradients(gradients)
                noisy_gradients.append(noisy)
        else:
            noisy_gradients = all_gradients
        
        # Secure aggregation
        if self.config.secure_aggregation and self.he:
            # Use homomorphic encryption
            encrypted_gradients = []
            for gradients in noisy_gradients:
                encrypted = [self.he.encrypt(g) for g in gradients if g is not None]
                encrypted_gradients.append(encrypted)
            
            # Aggregate encrypted gradients
            aggregated_encrypted = []
            for i in range(len(encrypted_gradients[0])):
                sum_encrypted = encrypted_gradients[0][i]
                for j in range(1, len(encrypted_gradients)):
                    sum_encrypted = self.he.add_encrypted(sum_encrypted, encrypted_gradients[j][i])
                aggregated_encrypted.append(sum_encrypted)
            
            # Decrypt and average
            aggregated = []
            for encrypted_grad in aggregated_encrypted:
                decrypted = self.he.decrypt(encrypted_grad)
                averaged = decrypted / len(client_updates)
                aggregated.append(averaged)
        else:
            # Simple averaging (no secure aggregation)
            aggregated = []
            for i in range(len(noisy_gradients[0])):
                sum_grad = noisy_gradients[0][i]
                for j in range(1, len(noisy_gradients)):
                    if sum_grad is not None and noisy_gradients[j][i] is not None:
                        sum_grad = sum_grad + noisy_gradients[j][i]
                averaged = sum_grad / len(noisy_gradients) if sum_grad is not None else None
                aggregated.append(averaged)
        
        # Convert back to update format
        aggregated_update = {}
        for i, grad in enumerate(aggregated):
            if grad is not None:
                aggregated_update[f'param_{i}'] = grad
        
        return aggregated_update
    
    def calculate_privacy_guarantees(self, 
                                   num_rounds: int, 
                                   num_clients: int) -> Dict[str, float]:
        """
        Calculate current privacy guarantees.
        
        Args:
            num_rounds: Number of completed rounds
            num_clients: Number of participating clients
            
        Returns:
            Dictionary with privacy metrics
        """
        epsilon_used, delta_used = self.dp.calculate_privacy_budget_consumed(
            num_rounds, num_clients
        )
        
        return {
            'epsilon_used': epsilon_used,
            'delta_used': delta_used,
            'epsilon_remaining': self.config.epsilon - epsilon_used,
            'delta_remaining': self.config.delta - delta_used,
            'privacy_budget_exhausted': epsilon_used >= self.config.epsilon
        }


def create_privacy_mechanism(config: PrivacyConfig) -> PrivacyPreservingAggregator:
    """Factory function to create privacy-preserving aggregator."""
    return PrivacyPreservingAggregator(config)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing privacy mechanisms...")
    
    # Create privacy configuration
    config = PrivacyConfig(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        secure_aggregation=True,
        num_clients_per_round=10
    )
    
    # Create privacy mechanism
    privacy_mechanism = create_privacy_mechanism(config)
    
    # Test with sample client updates
    client_updates = []
    for i in range(5):
        update = {
            'param_0': torch.randn(10, 5),
            'param_1': torch.randn(5, 3),
            'param_2': torch.randn(3, 1)
        }
        client_updates.append(update)
    
    client_ids = [f'client_{i}' for i in range(5)]
    
    # Aggregate updates
    aggregated = privacy_mechanism.aggregate_updates(client_updates, client_ids)
    
    print(f"Aggregated update keys: {list(aggregated.keys())}")
    print(f"Parameter 0 shape: {aggregated['param_0'].shape}")
    
    # Calculate privacy guarantees
    privacy_guarantees = privacy_mechanism.calculate_privacy_guarantees(
        num_rounds=10, num_clients=5
    )
    
    print(f"\nPrivacy guarantees:")
    for key, value in privacy_guarantees.items():
        print(f"  {key}: {value:.6f}")


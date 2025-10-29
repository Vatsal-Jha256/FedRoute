"""
FedRoute Federated Learning Client

Represents a vehicle (edge device) participating in federated learning.
Integrates with SUMO for realistic vehicle simulation.

Author: FedRoute Team
Date: October 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import time
import socket
import pickle
import argparse

from src.models.fmtl_model import create_fedroute_model


class FedRouteClient:
    """
    Federated Learning Client (Vehicle).
    """
    
    def __init__(self, client_id: str, server_host: str = 'localhost', 
                 server_port: int = 8080):
        """Initialize FL client."""
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        
        # Calculate unique port for this client
        client_num = int(client_id.split('_')[-1]) if '_' in client_id else 0
        self.listen_port = server_port + 100 + client_num
        
        # Model configuration (same as server)
        self.model_config = {
            'context_input_dim': 10,
            'context_hidden_dims': [64, 128, 64],
            'path_hidden_dims': [32, 16],
            'music_hidden_dims': [32, 16],
            'num_poi_categories': 10,
            'num_pois': 100,
            'num_genres': 10,
            'num_artists': 100,
            'num_tracks': 200,
            'dropout_rate': 0.1
        }
        
        # Local model
        self.local_model = create_fedroute_model(self.model_config)
        
        # Local data (simulated)
        self.num_samples = np.random.randint(50, 200)
        self.data = self._generate_local_data()
        
        print(f"🚗 Client {self.client_id} initialized ({self.num_samples} samples)")
    
    def _generate_local_data(self) -> Dict:
        """Generate synthetic local training data."""
        contexts = torch.randn(self.num_samples, 10)
        path_labels = torch.randint(0, 10, (self.num_samples,))
        music_labels = torch.randint(0, 10, (self.num_samples,))
        
        return {
            'contexts': contexts,
            'path_labels': path_labels,
            'music_labels': music_labels
        }
    
    def connect(self) -> bool:
        """Connect to FL server and register."""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5)
                client_socket.connect((self.server_host, self.server_port))
                
                # Send client info including listen port
                info = {
                    'client_id': self.client_id,
                    'num_samples': self.num_samples,
                    'listen_port': self.listen_port
                }
                client_socket.send(pickle.dumps(info))
                
                # Receive acknowledgment
                ack = pickle.loads(client_socket.recv(4096))
                client_socket.close()
                
                if ack['status'] == 'connected':
                    print(f"✅ Connected to server at {self.server_host}:{self.server_port}")
                    return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⏳ Connection attempt {attempt+1}/{max_retries} failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Failed to connect after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    def listen(self):
        """Listen for training requests from server."""
        print(f"👂 Listening for training requests...")
        
        # Create socket to receive training requests
        listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            listen_socket.bind((self.server_host, self.listen_port))
            listen_socket.listen(1)
            listen_socket.settimeout(60)  # Timeout after 60 seconds
            
            print(f"👂 Listening on port {self.listen_port}...")
        except Exception as e:
            print(f"❌ Failed to bind to port {self.listen_port}: {e}")
            return
        
        try:
            while True:
                try:
                    client_socket, _ = listen_socket.accept()
                    
                    # Receive training request
                    size_bytes = client_socket.recv(4)
                    if not size_bytes:
                        break
                    size = int.from_bytes(size_bytes, 'big')
                    
                    data = b''
                    while len(data) < size:
                        chunk = client_socket.recv(min(4096, size - len(data)))
                        if not chunk:
                            break
                        data += chunk
                    
                    request = pickle.loads(data)
                    
                    if request['action'] == 'train':
                        # Perform local training
                        update = self.train(request['model_state'], request['round'])
                        
                        # Send update back
                        response = pickle.dumps(update)
                        client_socket.sendall(len(response).to_bytes(4, 'big'))
                        client_socket.sendall(response)
                    
                    client_socket.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error handling request: {e}")
                    continue
                
        except KeyboardInterrupt:
            print(f"\n🛑 {self.client_id} stopped")
        finally:
            listen_socket.close()
    
    def train(self, global_model_state: Dict, round_num: int) -> Dict:
        """
        Perform local training on private data.
        
        Args:
            global_model_state: Global model parameters from server
            round_num: Current FL round number
            
        Returns:
            Local model update
        """
        print(f"\n  🚗 {self.client_id}: Starting local training (Round {round_num})...")
        
        # Load global model
        self.local_model.load_state_dict(global_model_state)
        self.local_model.train()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.01)
        
        # Local training (3 epochs)
        total_loss = 0.0
        path_correct = 0
        music_correct = 0
        
        for epoch in range(3):
            outputs = self.local_model(self.data['contexts'])
            
            # Compute losses
            path_loss = nn.CrossEntropyLoss()(
                outputs['path']['poi_categories'], 
                self.data['path_labels']
            )
            music_loss = nn.CrossEntropyLoss()(
                outputs['music']['genres'], 
                self.data['music_labels']
            )
            
            loss = path_loss + music_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (last epoch)
            if epoch == 2:
                with torch.no_grad():
                    path_pred = outputs['path']['poi_categories'].argmax(dim=1)
                    music_pred = outputs['music']['genres'].argmax(dim=1)
                    path_correct = (path_pred == self.data['path_labels']).sum().item()
                    music_correct = (music_pred == self.data['music_labels']).sum().item()
        
        accuracy = (path_correct + music_correct) / (2 * self.num_samples)
        avg_loss = total_loss / 3
        
        # Simulate training time
        time.sleep(np.random.uniform(0.5, 1.5))
        
        return {
            'model_state': self.local_model.state_dict(),
            'num_samples': self.num_samples,
            'loss': avg_loss,
            'accuracy': accuracy
        }


def main():
    """Run a federated learning client."""
    parser = argparse.ArgumentParser(description='FedRoute FL Client')
    parser.add_argument('--id', type=str, required=True, help='Client ID (e.g., vehicle_00)')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    args = parser.parse_args()
    
    print(f"\n🚗 Starting FedRoute Client: {args.id}")
    
    client = FedRouteClient(args.id, args.host, args.port)
    
    if client.connect():
        client.listen()


if __name__ == '__main__':
    main()



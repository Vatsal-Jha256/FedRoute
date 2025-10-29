"""
Comprehensive Synthetic Data Generator for FedRoute Framework

This module generates realistic synthetic data for the IoV federated learning system
by combining and augmenting real-world NYC Taxi and Last.fm datasets with synthetic POI data.

Author: FedRoute Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path


class SyntheticDataGenerator:
    """
    Generates comprehensive synthetic data for IoV federated learning experiments.
    
    This generator creates:
    1. Vehicle trajectory data (from NYC Taxi data)
    2. Music listening preferences (from Last.fm data)
    3. POI (Points of Interest) data for NYC
    4. User profiles linking vehicles, music preferences, and POI visits
    """
    
    def __init__(self, 
                 taxi_data_path: str,
                 music_data_path: str,
                 output_dir: str,
                 num_clients: int = 100,
                 random_seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            taxi_data_path: Path to NYC Taxi dataset directory
            music_data_path: Path to Last.fm music dataset
            output_dir: Directory to save generated synthetic data
            num_clients: Number of federated learning clients (vehicles)
            random_seed: Random seed for reproducibility
        """
        self.taxi_data_path = Path(taxi_data_path)
        self.music_data_path = Path(music_data_path)
        self.output_dir = Path(output_dir)
        self.num_clients = num_clients
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NYC geographical bounds (Manhattan focus)
        self.nyc_bounds = {
            'lat_min': 40.700,
            'lat_max': 40.800,
            'lon_min': -74.020,
            'lon_max': -73.930
        }
        
        # POI categories for path recommendations
        self.poi_categories = [
            'Restaurant', 'Shopping', 'Entertainment', 'Park', 'Museum',
            'Hospital', 'School', 'Office', 'Gym', 'Cafe',
            'Gas Station', 'Parking', 'Hotel', 'Bank', 'Pharmacy',
            'Supermarket', 'Cinema', 'Theater', 'Library', 'Airport',
            'Train Station', 'Bus Stop', 'Charging Station', 'Scenic Spot', 'Beach'
        ]
        
        # Music genres for music recommendations
        self.music_genres = [
            'Rock', 'Pop', 'Hip Hop', 'Electronic', 'Jazz',
            'Classical', 'Country', 'R&B', 'Metal', 'Indie',
            'Folk', 'Blues', 'Reggae', 'Latin', 'Alternative',
            'Soul', 'Punk', 'Disco', 'House', 'Techno'
        ]
        
        # Time of day categories
        self.time_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
        
        # Day of week
        self.day_categories = ['Weekday', 'Weekend']
        
    def load_real_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load real NYC Taxi and Last.fm datasets.
        
        Returns:
            Tuple of (taxi_df, music_df)
        """
        print("Loading real-world datasets...")
        
        # Load NYC Taxi data (sample from first file for efficiency)
        taxi_files = list(self.taxi_data_path.glob("*.csv"))
        if not taxi_files:
            raise FileNotFoundError(f"No taxi data found in {self.taxi_data_path}")
        
        print(f"  Loading taxi data from {taxi_files[0].name}...")
        taxi_df = pd.read_csv(taxi_files[0], nrows=100000)  # Load first 100k rows
        
        # Load Last.fm music data
        print(f"  Loading music data from {self.music_data_path}...")
        music_df = pd.read_csv(self.music_data_path, nrows=50000)  # Load first 50k rows
        
        print(f"  Loaded {len(taxi_df)} taxi trips and {len(music_df)} music listens")
        
        return taxi_df, music_df
    
    def generate_poi_dataset(self, num_pois: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic POI (Points of Interest) dataset for NYC.
        
        Args:
            num_pois: Number of POIs to generate
            
        Returns:
            DataFrame with POI information
        """
        print(f"Generating {num_pois} synthetic POIs...")
        
        pois = []
        for i in range(num_pois):
            # Generate POI with realistic distribution
            category = np.random.choice(self.poi_categories)
            
            # Cluster POIs based on category (e.g., restaurants in specific areas)
            if category in ['Restaurant', 'Cafe', 'Shopping']:
                # More in midtown/downtown
                lat = np.random.normal(40.750, 0.015)
                lon = np.random.normal(-73.985, 0.015)
            elif category in ['Park', 'Scenic Spot', 'Beach']:
                # Near parks/waterfronts
                lat = np.random.normal(40.775, 0.020)
                lon = np.random.normal(-73.965, 0.020)
            else:
                # Random distribution
                lat = np.random.uniform(self.nyc_bounds['lat_min'], self.nyc_bounds['lat_max'])
                lon = np.random.uniform(self.nyc_bounds['lon_min'], self.nyc_bounds['lon_max'])
            
            # Clip to NYC bounds
            lat = np.clip(lat, self.nyc_bounds['lat_min'], self.nyc_bounds['lat_max'])
            lon = np.clip(lon, self.nyc_bounds['lon_min'], self.nyc_bounds['lon_max'])
            
            # Generate POI attributes
            poi = {
                'poi_id': f'POI_{i:04d}',
                'name': f'{category}_{i}',
                'category': category,
                'latitude': lat,
                'longitude': lon,
                'rating': np.random.uniform(3.0, 5.0),
                'popularity': np.random.exponential(100),
                'average_visit_duration_min': np.random.randint(15, 180),
                'price_level': np.random.randint(1, 5),
                'open_hour': np.random.choice([0, 6, 8, 9, 10]),
                'close_hour': np.random.choice([18, 20, 22, 24]),
            }
            pois.append(poi)
        
        poi_df = pd.DataFrame(pois)
        
        # Save POI dataset
        poi_output_path = self.output_dir / 'synthetic_pois.csv'
        poi_df.to_csv(poi_output_path, index=False)
        print(f"  Saved POI dataset to {poi_output_path}")
        
        return poi_df
    
    def generate_vehicle_trajectories(self, 
                                     taxi_df: pd.DataFrame,
                                     num_trajectories: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic vehicle trajectories based on real taxi data.
        
        Args:
            taxi_df: Real NYC taxi trip data
            num_trajectories: Number of trajectories to generate
            
        Returns:
            DataFrame with vehicle trajectory data
        """
        print(f"Generating {num_trajectories} vehicle trajectories...")
        
        trajectories = []
        
        # Sample and augment real taxi trips
        for i in range(num_trajectories):
            # Sample a real trip
            trip = taxi_df.sample(1).iloc[0]
            
            # Assign to a random client (vehicle)
            vehicle_id = f'vehicle_{np.random.randint(0, self.num_clients):03d}'
            
            # Extract and augment trip data
            try:
                pickup_lat = float(trip.get('pickup_latitude', 0))
                pickup_lon = float(trip.get('pickup_longitude', 0))
                dropoff_lat = float(trip.get('dropoff_latitude', 0))
                dropoff_lon = float(trip.get('dropoff_longitude', 0))
                
                # Add small random noise for variation
                pickup_lat += np.random.normal(0, 0.001)
                pickup_lon += np.random.normal(0, 0.001)
                dropoff_lat += np.random.normal(0, 0.001)
                dropoff_lon += np.random.normal(0, 0.001)
                
                # Generate timestamp
                base_date = datetime(2024, 1, 1)
                trip_date = base_date + timedelta(days=np.random.randint(0, 365))
                trip_hour = np.random.randint(0, 24)
                trip_minute = np.random.randint(0, 60)
                timestamp = trip_date.replace(hour=trip_hour, minute=trip_minute)
                
                # Determine time and day category
                if 5 <= trip_hour < 12:
                    time_category = 'Morning'
                elif 12 <= trip_hour < 17:
                    time_category = 'Afternoon'
                elif 17 <= trip_hour < 21:
                    time_category = 'Evening'
                else:
                    time_category = 'Night'
                
                day_category = 'Weekend' if trip_date.weekday() >= 5 else 'Weekday'
                
                # Generate contextual features
                trajectory = {
                    'trajectory_id': f'traj_{i:05d}',
                    'vehicle_id': vehicle_id,
                    'timestamp': timestamp,
                    'pickup_latitude': pickup_lat,
                    'pickup_longitude': pickup_lon,
                    'dropoff_latitude': dropoff_lat,
                    'dropoff_longitude': dropoff_lon,
                    'trip_distance': max(0.1, float(trip.get('trip_distance', np.random.uniform(0.5, 10)))),
                    'trip_duration_min': np.random.randint(5, 60),
                    'speed_mph': np.random.uniform(10, 40),
                    'time_of_day': time_category,
                    'day_of_week': day_category,
                    'weather': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Snowy'], p=[0.5, 0.3, 0.15, 0.05]),
                    'temperature_f': np.random.normal(60, 15),
                    'traffic_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
                }
                
                trajectories.append(trajectory)
            except (ValueError, TypeError, KeyError) as e:
                # Skip invalid entries
                continue
        
        traj_df = pd.DataFrame(trajectories)
        
        # Save trajectory dataset
        traj_output_path = self.output_dir / 'synthetic_trajectories.csv'
        traj_df.to_csv(traj_output_path, index=False)
        print(f"  Saved trajectory dataset to {traj_output_path}")
        
        return traj_df
    
    def generate_music_preferences(self, 
                                   music_df: pd.DataFrame,
                                   num_listens: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic music listening data based on real Last.fm data.
        
        Args:
            music_df: Real Last.fm music listening data
            num_listens: Number of listening events to generate
            
        Returns:
            DataFrame with music listening data
        """
        print(f"Generating {num_listens} music listening events...")
        
        listens = []
        
        # Extract unique artists and tracks from real data
        real_artists = music_df['Artist'].dropna().unique()[:500]  # Top 500 artists
        real_tracks = music_df['Track'].dropna().unique()[:1000]  # Top 1000 tracks
        
        for i in range(num_listens):
            # Assign to a random client
            vehicle_id = f'vehicle_{np.random.randint(0, self.num_clients):03d}'
            
            # Sample real artist and track
            artist = np.random.choice(real_artists)
            track = np.random.choice(real_tracks)
            genre = np.random.choice(self.music_genres)
            
            # Generate timestamp
            base_date = datetime(2024, 1, 1)
            listen_date = base_date + timedelta(days=np.random.randint(0, 365))
            listen_hour = np.random.randint(0, 24)
            listen_minute = np.random.randint(0, 60)
            timestamp = listen_date.replace(hour=listen_hour, minute=listen_minute)
            
            # Contextual features
            listen = {
                'listen_id': f'listen_{i:06d}',
                'vehicle_id': vehicle_id,
                'timestamp': timestamp,
                'artist': artist,
                'track': track,
                'genre': genre,
                'duration_sec': np.random.randint(120, 300),
                'skip': np.random.choice([0, 1], p=[0.7, 0.3]),
                'explicit_like': np.random.choice([0, 1], p=[0.85, 0.15]),
                'play_count': np.random.randint(1, 50),
                'time_of_day': 'Morning' if 5 <= listen_hour < 12 else 
                              'Afternoon' if 12 <= listen_hour < 17 else
                              'Evening' if 17 <= listen_hour < 21 else 'Night',
                'day_of_week': 'Weekend' if listen_date.weekday() >= 5 else 'Weekday',
            }
            
            listens.append(listen)
        
        music_df_synth = pd.DataFrame(listens)
        
        # Save music dataset
        music_output_path = self.output_dir / 'synthetic_music.csv'
        music_df_synth.to_csv(music_output_path, index=False)
        print(f"  Saved music dataset to {music_output_path}")
        
        return music_df_synth
    
    def generate_client_profiles(self,
                                traj_df: pd.DataFrame,
                                music_df: pd.DataFrame,
                                poi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate client (vehicle) profiles linking trajectories, music, and POIs.
        
        Args:
            traj_df: Vehicle trajectory data
            music_df: Music listening data
            poi_df: POI data
            
        Returns:
            DataFrame with client profiles
        """
        print(f"Generating {self.num_clients} client profiles...")
        
        profiles = []
        
        for i in range(self.num_clients):
            vehicle_id = f'vehicle_{i:03d}'
            
            # Get client's trajectories and music listens
            client_trajs = traj_df[traj_df['vehicle_id'] == vehicle_id]
            client_music = music_df[music_df['vehicle_id'] == vehicle_id]
            
            # Sample favorite POIs
            num_fav_pois = np.random.randint(5, 20)
            favorite_poi_ids = poi_df.sample(num_fav_pois)['poi_id'].tolist()
            
            # Sample favorite genres
            num_fav_genres = np.random.randint(2, 6)
            favorite_genres = np.random.choice(self.music_genres, num_fav_genres, replace=False).tolist()
            
            # Generate profile
            profile = {
                'vehicle_id': vehicle_id,
                'num_trips': len(client_trajs),
                'num_listens': len(client_music),
                'avg_trip_distance': client_trajs['trip_distance'].mean() if len(client_trajs) > 0 else 0,
                'favorite_pois': ','.join(favorite_poi_ids),
                'favorite_genres': ','.join(favorite_genres),
                'data_quality': np.random.uniform(0.6, 1.0),  # Heterogeneity
                'computational_capacity': np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.5, 0.3]),
                'privacy_sensitivity': np.random.uniform(0.3, 1.0),
            }
            
            profiles.append(profile)
        
        profile_df = pd.DataFrame(profiles)
        
        # Save client profiles
        profile_output_path = self.output_dir / 'client_profiles.csv'
        profile_df.to_csv(profile_output_path, index=False)
        print(f"  Saved client profiles to {profile_output_path}")
        
        return profile_df
    
    def generate_federated_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Generate complete federated learning datasets.
        
        Returns:
            Dictionary containing all generated datasets
        """
        print("="*60)
        print("FEDROUTE SYNTHETIC DATA GENERATION")
        print("="*60)
        
        # Load real data
        taxi_df, music_df = self.load_real_data()
        
        # Generate synthetic datasets
        poi_df = self.generate_poi_dataset(num_pois=1000)
        traj_df = self.generate_vehicle_trajectories(taxi_df, num_trajectories=5000)
        music_synth_df = self.generate_music_preferences(music_df, num_listens=10000)
        profile_df = self.generate_client_profiles(traj_df, music_synth_df, poi_df)
        
        # Generate metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'num_clients': self.num_clients,
            'num_pois': len(poi_df),
            'num_trajectories': len(traj_df),
            'num_music_listens': len(music_synth_df),
            'random_seed': self.random_seed,
            'poi_categories': self.poi_categories,
            'music_genres': self.music_genres,
        }
        
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved metadata to {metadata_path}")
        
        print("\n" + "="*60)
        print("DATA GENERATION COMPLETE!")
        print("="*60)
        print(f"Generated datasets:")
        print(f"  - {len(poi_df)} POIs")
        print(f"  - {len(traj_df)} vehicle trajectories")
        print(f"  - {len(music_synth_df)} music listening events")
        print(f"  - {len(profile_df)} client profiles")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*60)
        
        return {
            'pois': poi_df,
            'trajectories': traj_df,
            'music': music_synth_df,
            'profiles': profile_df,
            'metadata': metadata
        }


def main():
    """Main function to generate synthetic data."""
    
    # Configuration
    base_dir = Path(__file__).parent.parent.parent
    taxi_data_path = base_dir / 'data' / 'NYC_Taxi_Dataset'
    music_data_path = base_dir / 'data' / 'Last.fm_data.csv'
    output_dir = base_dir / 'data' / 'synthetic'
    
    # Generate data
    generator = SyntheticDataGenerator(
        taxi_data_path=str(taxi_data_path),
        music_data_path=str(music_data_path),
        output_dir=str(output_dir),
        num_clients=100,
        random_seed=42
    )
    
    datasets = generator.generate_federated_datasets()
    
    return datasets


if __name__ == '__main__':
    main()



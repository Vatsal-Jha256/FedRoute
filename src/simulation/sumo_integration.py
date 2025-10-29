"""
SUMO Integration for FedRoute Framework

This module provides integration with SUMO traffic simulator for generating
realistic vehicle mobility patterns and contextual data.
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import traci
import traci.constants as tc
from dataclasses import dataclass
import time
import random
from pathlib import Path


@dataclass
class VehicleState:
    """Container for vehicle state information."""
    vehicle_id: str
    position: Tuple[float, float]
    speed: float
    angle: float
    road_id: str
    lane_id: str
    route: List[str]
    departure_time: float
    arrival_time: Optional[float] = None


@dataclass
class SimulationConfig:
    """Configuration for SUMO simulation."""
    sumo_binary: str = "sumo"
    config_file: str = "simulation.sumocfg"
    network_file: str = "network.net.xml"
    route_file: str = "routes.rou.xml"
    simulation_time: float = 3600.0  # 1 hour
    step_size: float = 1.0  # 1 second
    random_seed: int = 42
    gui: bool = False


class SUMOIntegration:
    """
    Main class for integrating SUMO with FedRoute simulation.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sumo_process = None
        self.vehicle_states = {}
        self.simulation_data = []
        self.current_time = 0.0
        
    def start_simulation(self) -> bool:
        """
        Start SUMO simulation.
        
        Returns:
            True if simulation started successfully
        """
        try:
            # Prepare SUMO command
            cmd = [
                self.config.sumo_binary,
                "-c", self.config.config_file,
                "--step-length", str(self.config.step_size),
                "--random", str(self.config.random_seed),
                "--no-warnings", "true"
            ]
            
            if not self.config.gui:
                cmd.append("--no-gui")
            
            # Start SUMO process
            self.sumo_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Connect to SUMO
            traci.init(port=8813)
            
            print(f"SUMO simulation started with PID: {self.sumo_process.pid}")
            return True
            
        except Exception as e:
            print(f"Failed to start SUMO simulation: {e}")
            return False
    
    def stop_simulation(self):
        """Stop SUMO simulation and cleanup."""
        if self.sumo_process:
            traci.close()
            self.sumo_process.terminate()
            self.sumo_process.wait()
            self.sumo_process = None
            print("SUMO simulation stopped")
    
    def step_simulation(self) -> bool:
        """
        Advance simulation by one step.
        
        Returns:
            True if simulation is still running
        """
        try:
            traci.simulationStep()
            self.current_time = traci.simulation.getTime()
            
            # Update vehicle states
            self._update_vehicle_states()
            
            # Check if simulation is complete
            if self.current_time >= self.config.simulation_time:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in simulation step: {e}")
            return False
    
    def _update_vehicle_states(self):
        """Update the state of all vehicles in the simulation."""
        # Get all vehicle IDs
        vehicle_ids = traci.vehicle.getIDList()
        
        # Update existing vehicles
        for vehicle_id in vehicle_ids:
            if vehicle_id not in self.vehicle_states:
                # New vehicle
                self.vehicle_states[vehicle_id] = self._create_vehicle_state(vehicle_id)
            else:
                # Update existing vehicle
                self._update_single_vehicle(vehicle_id)
        
        # Remove vehicles that are no longer in simulation
        active_vehicles = set(vehicle_ids)
        self.vehicle_states = {
            vid: state for vid, state in self.vehicle_states.items() 
            if vid in active_vehicles
        }
    
    def _create_vehicle_state(self, vehicle_id: str) -> VehicleState:
        """Create initial state for a new vehicle."""
        try:
            position = traci.vehicle.getPosition(vehicle_id)
            speed = traci.vehicle.getSpeed(vehicle_id)
            angle = traci.vehicle.getAngle(vehicle_id)
            road_id = traci.vehicle.getRoadID(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            departure_time = traci.vehicle.getDeparture(vehicle_id)
            
            return VehicleState(
                vehicle_id=vehicle_id,
                position=position,
                speed=speed,
                angle=angle,
                road_id=road_id,
                lane_id=lane_id,
                route=route,
                departure_time=departure_time
            )
        except Exception as e:
            print(f"Error creating vehicle state for {vehicle_id}: {e}")
            return None
    
    def _update_single_vehicle(self, vehicle_id: str):
        """Update state of a single vehicle."""
        try:
            state = self.vehicle_states[vehicle_id]
            state.position = traci.vehicle.getPosition(vehicle_id)
            state.speed = traci.vehicle.getSpeed(vehicle_id)
            state.angle = traci.vehicle.getAngle(vehicle_id)
            state.road_id = traci.vehicle.getRoadID(vehicle_id)
            state.lane_id = traci.vehicle.getLaneID(vehicle_id)
            
            # Check if vehicle has arrived
            if traci.vehicle.getArrival(vehicle_id) != -1:
                state.arrival_time = self.current_time
                
        except Exception as e:
            print(f"Error updating vehicle {vehicle_id}: {e}")
    
    def get_vehicle_context(self, vehicle_id: str) -> Dict:
        """
        Get contextual information for a specific vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            Dictionary containing contextual features
        """
        if vehicle_id not in self.vehicle_states:
            return None
        
        state = self.vehicle_states[vehicle_id]
        
        # Extract contextual features
        context = {
            'vehicle_id': vehicle_id,
            'timestamp': self.current_time,
            'position': state.position,
            'speed': state.speed,
            'angle': state.angle,
            'road_id': state.road_id,
            'lane_id': state.lane_id,
            'time_of_day': self._get_time_of_day(),
            'day_of_week': self._get_day_of_week(),
            'weather': self._get_weather_condition(),
            'traffic_density': self._get_traffic_density(state.road_id),
            'road_type': self._get_road_type(state.road_id)
        }
        
        return context
    
    def _get_time_of_day(self) -> float:
        """Get normalized time of day (0-1)."""
        return (self.current_time % 86400) / 86400.0  # 86400 seconds in a day
    
    def _get_day_of_week(self) -> int:
        """Get day of week (0-6, Monday=0)."""
        # Simplified - in practice, would use actual date
        return int(self.current_time // 86400) % 7
    
    def _get_weather_condition(self) -> int:
        """Get weather condition (simplified)."""
        # Simplified weather simulation
        weather_conditions = [0, 1, 2]  # 0: clear, 1: rain, 2: snow
        return random.choices(weather_conditions, weights=[0.7, 0.2, 0.1])[0]
    
    def _get_traffic_density(self, road_id: str) -> float:
        """Get traffic density for a road (0-1)."""
        try:
            # Get number of vehicles on the road
            vehicle_count = len(traci.edge.getLastStepVehicleNumber(road_id))
            # Normalize by road capacity (simplified)
            max_capacity = 50  # Simplified assumption
            return min(vehicle_count / max_capacity, 1.0)
        except:
            return 0.0
    
    def _get_road_type(self, road_id: str) -> int:
        """Get road type (simplified classification)."""
        # Simplified road type classification
        if 'highway' in road_id.lower():
            return 0  # Highway
        elif 'primary' in road_id.lower():
            return 1  # Primary road
        elif 'secondary' in road_id.lower():
            return 2  # Secondary road
        else:
            return 3  # Local road
    
    def get_all_vehicle_contexts(self) -> List[Dict]:
        """Get contextual information for all vehicles."""
        contexts = []
        for vehicle_id in self.vehicle_states.keys():
            context = self.get_vehicle_context(vehicle_id)
            if context:
                contexts.append(context)
        return contexts
    
    def save_simulation_data(self, filename: str):
        """Save simulation data to file."""
        df = pd.DataFrame(self.simulation_data)
        df.to_csv(filename, index=False)
        print(f"Simulation data saved to {filename}")
    
    def run_full_simulation(self) -> List[Dict]:
        """
        Run the complete simulation and return all data.
        
        Returns:
            List of all simulation data points
        """
        if not self.start_simulation():
            return []
        
        print(f"Running simulation for {self.config.simulation_time} seconds...")
        
        step_count = 0
        while self.step_simulation():
            # Collect data every 10 steps (10 seconds)
            if step_count % 10 == 0:
                contexts = self.get_all_vehicle_contexts()
                self.simulation_data.extend(contexts)
                print(f"Step {step_count}: {len(contexts)} vehicles active")
            
            step_count += 1
        
        self.stop_simulation()
        print(f"Simulation completed. Collected {len(self.simulation_data)} data points.")
        
        return self.simulation_data


class NetworkGenerator:
    """
    Generate SUMO network files from OpenStreetMap data.
    """
    
    def __init__(self, osm_file: str, output_dir: str = "sumo_data"):
        self.osm_file = osm_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_network(self) -> str:
        """
        Generate SUMO network file from OSM data.
        
        Returns:
            Path to generated network file
        """
        try:
            # Use netconvert to convert OSM to SUMO network
            network_file = self.output_dir / "network.net.xml"
            
            cmd = [
                "netconvert",
                "--osm-files", self.osm_file,
                "--output-file", str(network_file),
                "--remove-edges.by-vclass", "pedestrian,bicycle",
                "--remove-edges.by-type", "highway.footway,highway.cycleway"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Network file generated: {network_file}")
                return str(network_file)
            else:
                print(f"Error generating network: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error in network generation: {e}")
            return None
    
    def generate_routes(self, num_vehicles: int = 1000) -> str:
        """
        Generate random routes for vehicles.
        
        Args:
            num_vehicles: Number of vehicles to generate
            
        Returns:
            Path to generated route file
        """
        try:
            route_file = self.output_dir / "routes.rou.xml"
            
            cmd = [
                "randomTrips.py",
                "-n", str(self.output_dir / "network.net.xml"),
                "-o", str(route_file),
                "-r", str(self.output_dir / "routes.rou.xml"),
                "--begin", "0",
                "--end", "3600",
                "--period", "1.0",
                "--vehicles", str(num_vehicles)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Route file generated: {route_file}")
                return str(route_file)
            else:
                print(f"Error generating routes: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Error in route generation: {e}")
            return None
    
    def generate_config(self, 
                       network_file: str, 
                       route_file: str) -> str:
        """
        Generate SUMO configuration file.
        
        Args:
            network_file: Path to network file
            route_file: Path to route file
            
        Returns:
            Path to generated config file
        """
        try:
            config_file = self.output_dir / "simulation.sumocfg"
            
            # Create configuration XML
            config = ET.Element("configuration")
            
            # Input section
            input_elem = ET.SubElement(config, "input")
            ET.SubElement(input_elem, "net-file", value=network_file)
            ET.SubElement(input_elem, "route-files", value=route_file)
            
            # Time section
            time_elem = ET.SubElement(config, "time")
            ET.SubElement(time_elem, "begin", value="0")
            ET.SubElement(time_elem, "end", value="3600")
            ET.SubElement(time_elem, "step-length", value="1.0")
            
            # Processing section
            processing_elem = ET.SubElement(config, "processing")
            ET.SubElement(processing_elem, "no-warnings", value="true")
            
            # Write to file
            tree = ET.ElementTree(config)
            tree.write(config_file, encoding='utf-8', xml_declaration=True)
            
            print(f"Config file generated: {config_file}")
            return str(config_file)
            
        except Exception as e:
            print(f"Error generating config: {e}")
            return None


def create_simulation_environment(osm_file: str, 
                                 output_dir: str = "sumo_data",
                                 num_vehicles: int = 1000) -> SUMOIntegration:
    """
    Create a complete SUMO simulation environment.
    
    Args:
        osm_file: Path to OpenStreetMap file
        output_dir: Directory for SUMO files
        num_vehicles: Number of vehicles in simulation
        
    Returns:
        Configured SUMOIntegration instance
    """
    # Generate SUMO files
    generator = NetworkGenerator(osm_file, output_dir)
    
    network_file = generator.generate_network()
    if not network_file:
        raise RuntimeError("Failed to generate network file")
    
    route_file = generator.generate_routes(num_vehicles)
    if not route_file:
        raise RuntimeError("Failed to generate route file")
    
    config_file = generator.generate_config(network_file, route_file)
    if not config_file:
        raise RuntimeError("Failed to generate config file")
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        config_file=config_file,
        network_file=network_file,
        route_file=route_file,
        simulation_time=3600.0,
        step_size=1.0,
        random_seed=42,
        gui=False
    )
    
    return SUMOIntegration(sim_config)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing SUMO integration...")
    
    # Create a simple test simulation
    # Note: This requires SUMO to be installed and an OSM file
    try:
        # For testing, create a minimal simulation
        sim_config = SimulationConfig(
            config_file="test_config.sumocfg",
            simulation_time=60.0,  # 1 minute test
            gui=False
        )
        
        simulation = SUMOIntegration(sim_config)
        
        # Test simulation steps
        if simulation.start_simulation():
            print("Simulation started successfully")
            
            step_count = 0
            while simulation.step_simulation() and step_count < 10:
                contexts = simulation.get_all_vehicle_contexts()
                print(f"Step {step_count}: {len(contexts)} vehicles")
                step_count += 1
            
            simulation.stop_simulation()
            print("Test simulation completed")
        else:
            print("Failed to start simulation")
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: SUMO must be installed and properly configured")


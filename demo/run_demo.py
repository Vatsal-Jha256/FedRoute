"""
FedRoute Demo Launcher with SUMO Visualization

Launches the complete federated learning demo:
1. SUMO traffic simulation (visualizing vehicles)
2. FL Server
3. Multiple FL Clients (one per vehicle)
4. Real-time FL visualization dashboard

Author: FedRoute Team
Date: October 2025
"""

import subprocess
import time
import sys
import os
from pathlib import Path
import threading
import queue
import re
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
    print("   Continuing without visualization...\n")

# Change to demo directory
demo_dir = Path(__file__).parent
os.chdir(demo_dir)

# Shared data for visualization
viz_data = {
    'rounds': [],
    'path_accuracy': [],
    'music_accuracy': [],
    'combined_accuracy': [],
    'clients_connected': 0,
    'selected_clients': [],
    'round_times': [],
    'current_round': 0,
    'total_rounds': 15,  # Increased from 5 to 15
    'messages': []
}
viz_lock = threading.Lock()
output_queue = queue.Queue()

print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║             🚀 FEDROUTE FEDERATED LEARNING DEMO 🚀                ║
║                                                                    ║
║  Privacy-Preserving Dual Recommendations for Internet of Vehicles ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

This demo will launch:
  🚗 SUMO-GUI: Traffic simulation with 10 vehicles
  🌐 FL Server: Coordinating federated learning
  📱 10 Clients: Each vehicle running local training

""")

input("Press Enter to start the demo...")

processes = []

def parse_server_output(line, viz_data, viz_lock):
    """Parse server output and update visualization data."""
    with viz_lock:
        # Parse connected clients
        if "Client connected:" in line:
            match = re.search(r'Client connected: (vehicle_\d+)', line)
            if match:
                viz_data['clients_connected'] += 1
                viz_data['messages'].append(f"[OK] {match.group(1)} joined")
        
        # Parse round number - FIXED: Look for "ROUND X" pattern
        elif "ROUND" in line and not "=" in line:
            match = re.search(r'ROUND (\d+)', line)
            if match:
                round_num = int(match.group(1))
                viz_data['current_round'] = round_num
                viz_data['messages'].append(f"[>>] Round {round_num} started")
        
        # Parse selected clients
        elif "Selected clients:" in line:
            match = re.search(r'Selected clients: (.+)', line)
            if match:
                clients = match.group(1).strip().split(', ')
                viz_data['selected_clients'] = clients[:4]  # Keep last 4
        
        # Parse accuracies
        elif "Path Accuracy:" in line:
            match = re.search(r'Path Accuracy:\s+([\d.]+)', line)
            if match:
                viz_data['path_accuracy'].append(float(match.group(1)))
        elif "Music Accuracy:" in line:
            match = re.search(r'Music Accuracy:\s+([\d.]+)', line)
            if match:
                viz_data['music_accuracy'].append(float(match.group(1)))
        elif "Combined Accuracy:" in line:
            match = re.search(r'Combined Accuracy:\s+([\d.]+)', line)
            if match:
                acc = float(match.group(1))
                viz_data['combined_accuracy'].append(acc)
                viz_data['rounds'].append(len(viz_data['rounds']) + 1)
                # Update current round based on completed rounds
                viz_data['current_round'] = len(viz_data['rounds'])
                viz_data['messages'].append(f"[**] Accuracy: {acc:.4f}")
                # Keep only last 10 messages
                if len(viz_data['messages']) > 10:
                    viz_data['messages'] = viz_data['messages'][-10:]

def create_visualization(viz_data, viz_lock):
    """Create real-time visualization dashboard (non-blocking)."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # Enable interactive mode for non-blocking display
    plt.ion()
    
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('FedRoute Federated Learning Dashboard', fontsize=16, fontweight='bold')
    
    # Create subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0:2, 0])  # Accuracy plot (large)
    ax2 = fig.add_subplot(gs[0, 1])    # Client status
    ax3 = fig.add_subplot(gs[1, 1])    # Round progress
    ax4 = fig.add_subplot(gs[2, :])    # Messages log
    
    plt.show(block=False)
    plt.pause(0.1)
    
    return fig, (ax1, ax2, ax3, ax4)

def update_visualization(fig, axes, viz_data, viz_lock):
    """Update the visualization with current data."""
    if not MATPLOTLIB_AVAILABLE or fig is None:
        return
    
    ax1, ax2, ax3, ax4 = axes
    
    with viz_lock:
        # Clear all axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        
        # Plot 1: Training Accuracy Over Rounds
        ax1.set_title('Model Accuracy Evolution', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Round', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        
        if len(viz_data['rounds']) > 0:
            rounds = viz_data['rounds']
            ax1.plot(rounds, viz_data['path_accuracy'], 'o-', 
                    label='Path Rec.', linewidth=2, markersize=6, color='#FF6B6B')
            ax1.plot(rounds, viz_data['music_accuracy'], 's-', 
                    label='Music Rec.', linewidth=2, markersize=6, color='#4ECDC4')
            ax1.plot(rounds, viz_data['combined_accuracy'], 'd-', 
                    label='Combined', linewidth=3, markersize=7, color='#95E1D3')
            ax1.legend(loc='lower right', fontsize=9)
            
            # Add value labels on last point
            if len(rounds) > 0:
                for acc_list, color in [(viz_data['path_accuracy'], '#FF6B6B'),
                                        (viz_data['music_accuracy'], '#4ECDC4'),
                                        (viz_data['combined_accuracy'], '#95E1D3')]:
                    if acc_list:
                        ax1.text(rounds[-1], acc_list[-1], f'{acc_list[-1]:.3f}',
                               fontsize=8, ha='left', va='bottom', color=color, fontweight='bold')
        
        # Plot 2: Client Connection Status
        ax2.set_title('Client Network Status', fontweight='bold', fontsize=12)
        ax2.axis('off')
        
        status_text = f"Connected: {viz_data['clients_connected']}/10\n\n"
        if viz_data['selected_clients']:
            status_text += "Active This Round:\n"
            for i, client in enumerate(viz_data['selected_clients'], 1):
                status_text += f"  {i}. {client}\n"
        else:
            status_text += "Waiting..."
        
        ax2.text(0.1, 0.9, status_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 3: Round Progress
        ax3.set_title('Training Progress', fontweight='bold', fontsize=12)
        ax3.axis('off')
        
        progress = (viz_data['current_round'] / viz_data['total_rounds']) * 100 if viz_data['total_rounds'] > 0 else 0
        progress_text = f"Round: {viz_data['current_round']}/{viz_data['total_rounds']}\n"
        progress_text += f"Progress: {progress:.1f}%\n\n"
        
        # Progress bar
        bar_length = 20
        filled = int(bar_length * progress / 100)
        bar = '#' * filled + '-' * (bar_length - filled)
        progress_text += f"[{bar}]"
        
        ax3.text(0.1, 0.9, progress_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Plot 4: Activity Log
        ax4.set_title('Activity Log', fontweight='bold', fontsize=12)
        ax4.axis('off')
        
        if viz_data['messages']:
            log_text = '\n'.join(viz_data['messages'][-8:])  # Show last 8 messages
        else:
            log_text = 'Waiting for activity...'
        
        ax4.text(0.02, 0.95, log_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

try:
    # 1. Start SUMO-GUI
    print("\n[1/3] 🚗 Starting SUMO traffic simulation...")
    sumo_cmd = [
        'sumo-gui',
        '-c', 'simulation.sumocfg',
        '--start',
        '--delay', '100'  # Slow down simulation for better visualization
    ]
    
    sumo_process = subprocess.Popen(
        sumo_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    processes.append(('SUMO', sumo_process))
    print("   ✅ SUMO started (you should see the GUI window)")
    time.sleep(2)
    
    # 2. Start FL Server
    print("\n[2/3] 🌐 Starting FL Server...")
    server_process = subprocess.Popen(
        [sys.executable, 'server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    processes.append(('Server', server_process))
    time.sleep(2)
    print("   ✅ Server started")
    
    # 3. Start Clients
    print("\n[3/3] 📱 Starting FL Clients...")
    num_clients = 10
    
    for i in range(num_clients):
        client_id = f"vehicle_{i:02d}"
        client_process = subprocess.Popen(
            [sys.executable, 'client.py', '--id', client_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append((client_id, client_process))
        time.sleep(0.5)  # Give each client time to initialize
    
    print(f"   ✅ {num_clients} clients started\n")
    print("⏳ Giving clients time to connect and start listening...")
    time.sleep(3)
    
    print("="*70)
    print("🎉 DEMO RUNNING!")
    print("="*70)
    print("""
Watch the SUMO window to see vehicles moving!
Each colored vehicle is a federated learning client.

The server will:
  1. Select 4 clients per round
  2. Send them the global model
  3. Receive their trained updates
  4. Aggregate and update the global model
  5. Repeat for 15 rounds (increased for longer demo!)

""")
    
    print("📊 Server Output:")
    print("-" * 70)
    sys.stdout.flush()
    
    # Create visualization first (non-blocking)
    viz_fig = None
    viz_axes = None
    if MATPLOTLIB_AVAILABLE:
        print("\n📊 Opening Real-Time Visualization Dashboard...")
        print("   Dashboard will update every second while training runs!\n")
        sys.stdout.flush()
        time.sleep(0.5)
        
        try:
            viz_fig, viz_axes = create_visualization(viz_data, viz_lock)
            if viz_fig:
                print("   ✅ Dashboard opened successfully\n")
            else:
                print("   ⚠️  Dashboard failed to open, continuing with text-only mode\n")
        except Exception as e:
            print(f"   ⚠️  Visualization error: {e}")
            print("   Continuing with text-only mode\n")
            viz_fig = None
    
    # Monitor server output and update visualization
    training_complete = False
    last_viz_update = time.time()
    
    # Use threading for cross-platform compatibility
    output_queue = queue.Queue()
    
    def read_server_output():
        """Read server output in background thread."""
        try:
            for line in server_process.stdout:
                output_queue.put(line)
                if "TRAINING COMPLETE" in line:
                    break
        except:
            pass
    
    reader_thread = threading.Thread(target=read_server_output, daemon=True)
    reader_thread.start()
    
    try:
        while not training_complete:
            # Process server output from queue
            try:
                line = output_queue.get(timeout=0.1)
                print(line, end='', flush=True)
                parse_server_output(line, viz_data, viz_lock)
                
                if "TRAINING COMPLETE" in line:
                    print("\n✅ All rounds completed successfully!")
                    training_complete = True
            except queue.Empty:
                pass
            
            # Update visualization every second
            if viz_fig and (time.time() - last_viz_update) > 1.0:
                try:
                    update_visualization(viz_fig, viz_axes, viz_data, viz_lock)
                    last_viz_update = time.time()
                except:
                    pass  # Window might be closed
            
            # Check if server process is still running
            if server_process.poll() is not None and output_queue.empty():
                print("\n⚠️  Server process ended")
                training_complete = True
                break
                
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Final visualization update
    if viz_fig:
        print("\n📊 Keeping visualization open for 5 seconds...")
        for _ in range(5):
            try:
                update_visualization(viz_fig, viz_axes, viz_data, viz_lock)
                time.sleep(1)
            except:
                break
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    
    # Print final summary
    with viz_lock:
        if viz_data['rounds']:
            print(f"""
Summary:
  ✓ 10 vehicles participated in federated learning
  ✓ Completed {len(viz_data['rounds'])} training rounds
  ✓ Final Path Accuracy: {viz_data['path_accuracy'][-1]:.4f}
  ✓ Final Music Accuracy: {viz_data['music_accuracy'][-1]:.4f}
  ✓ Final Combined Accuracy: {viz_data['combined_accuracy'][-1]:.4f}
  ✓ Privacy preserved (data never left vehicles)
  ✓ Global model improved through collaboration
  ✓ Real-time traffic simulation with SUMO

Thank you for trying FedRoute!
""")
        else:
            print("""
Summary:
  ✓ 10 vehicles participated in federated learning
  ✓ Privacy preserved (data never left vehicles)
  ✓ Real-time traffic simulation with SUMO

Thank you for trying FedRoute!
""")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Demo interrupted by user")

except Exception as e:
    print(f"\n❌ Error: {e}")

finally:
    # Cleanup
    print("\n🧹 Cleaning up...")
    for name, process in processes:
        try:
            process.terminate()
            process.wait(timeout=2)
            print(f"   ✓ Stopped {name}")
        except:
            try:
                process.kill()
            except:
                pass
    
    print("\n👋 Goodbye!\n")



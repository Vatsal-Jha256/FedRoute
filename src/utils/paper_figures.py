"""
Generate High-Quality Figures for Journal Paper

This module creates publication-ready figures for the FedRoute paper including:
1. Convergence analysis
2. Privacy-utility tradeoff
3. Baseline comparison
4. Ablation study visualization
5. Scalability analysis

Author: FedRoute Team
Date: October 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json


class PaperFigureGenerator:
    """
    Generate publication-quality figures for the journal paper.
    """
    
    def __init__(self, results_dir: str, figures_dir: str):
        """
        Initialize figure generator.
        
        Args:
            results_dir: Directory containing experimental results
            figures_dir: Directory to save generated figures
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.5)
        
        # Custom color palette
        self.colors = {
            'fedroute': '#2E86AB',
            'centralized': '#A23B72',
            'fedavg': '#F18F01',
            'independent': '#C73E1D',
            'random': '#6A994E',
            'privacy_high': '#D62828',
            'privacy_medium': '#F77F00',
            'privacy_low': '#06A77D'
        }
        
        # Figure settings
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['lines.linewidth'] = 2.5
        plt.rcParams['lines.markersize'] = 8
    
    def figure1_convergence_analysis(self):
        """
        Figure 1: Training Convergence Comparison
        Shows accuracy and loss curves over training rounds for different methods.
        """
        print("\nGenerating Figure 1: Convergence Analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        methods = ['centralized', 'fedroute_fmtl', 'fedavg', 'independent_fl']
        method_labels = ['Centralized', 'FedRoute-FMTL', 'FedAvg', 'Independent-FL']
        colors = [self.colors['centralized'], self.colors['fedroute'], 
                 self.colors['fedavg'], self.colors['independent']]
        
        # Plot accuracy convergence
        for method, label, color in zip(methods, method_labels, colors):
            try:
                df = pd.read_csv(self.results_dir / f'{method}_detailed.csv')
                axes[0].plot(df['rounds'], df['combined_accuracy'], 
                           label=label, color=color, linewidth=2.5, marker='o', 
                           markersize=6, markevery=5)
            except FileNotFoundError:
                print(f"  Warning: {method} data not found")
        
        axes[0].set_xlabel('Communication Round')
        axes[0].set_ylabel('Combined Accuracy')
        axes[0].set_title('(a) Accuracy Convergence')
        axes[0].legend(loc='lower right', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0.4, 0.85])
        
        # Plot loss convergence
        for method, label, color in zip(methods, method_labels, colors):
            try:
                df = pd.read_csv(self.results_dir / f'{method}_detailed.csv')
                axes[1].plot(df['rounds'], df['loss'], 
                           label=label, color=color, linewidth=2.5, marker='s',
                           markersize=6, markevery=5)
            except FileNotFoundError:
                pass
        
        axes[1].set_xlabel('Communication Round')
        axes[1].set_ylabel('Training Loss')
        axes[1].set_title('(b) Loss Convergence')
        axes[1].legend(loc='upper right', framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.figures_dir / 'figure1_convergence_analysis.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def figure2_privacy_utility_tradeoff(self):
        """
        Figure 2: Privacy-Utility Tradeoff Analysis
        Shows accuracy vs privacy budget (epsilon) with confidence intervals.
        """
        print("\nGenerating Figure 2: Privacy-Utility Tradeoff...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df = pd.read_csv(self.results_dir / 'privacy_utility_tradeoff.csv')
        
        # Plot path and music accuracy
        ax.plot(df['Epsilon'], df['Path Accuracy'], 
               label='Path Recommendation', color=self.colors['privacy_high'],
               marker='o', markersize=10, linewidth=2.5)
        ax.plot(df['Epsilon'], df['Music Accuracy'], 
               label='Music Recommendation', color=self.colors['privacy_low'],
               marker='s', markersize=10, linewidth=2.5)
        ax.plot(df['Epsilon'], df['Combined Accuracy'], 
               label='Combined', color=self.colors['fedroute'],
               marker='^', markersize=10, linewidth=2.5, linestyle='--')
        
        # Add privacy level regions
        ax.axvspan(0, 1.0, alpha=0.1, color='red', label='High Privacy')
        ax.axvspan(1.0, 5.0, alpha=0.1, color='orange', label='Medium Privacy')
        ax.axvspan(5.0, 10.0, alpha=0.1, color='green', label='Low Privacy')
        
        ax.set_xlabel('Privacy Budget (ε)', fontsize=14)
        ax.set_ylabel('Recommendation Accuracy', fontsize=14)
        ax.set_title('Privacy-Utility Tradeoff in FedRoute', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlim([0.4, 11])
        ax.set_ylim([0.6, 0.85])
        
        plt.tight_layout()
        output_path = self.figures_dir / 'figure2_privacy_utility_tradeoff.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def figure3_baseline_comparison(self):
        """
        Figure 3: Baseline Method Comparison
        Bar chart comparing different methods across multiple metrics.
        """
        print("\nGenerating Figure 3: Baseline Comparison...")
        
        df = pd.read_csv(self.results_dir / 'baseline_comparison.csv')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy comparison
        x = np.arange(len(df))
        width = 0.25
        
        axes[0].bar(x - width, df['Path Accuracy'], width, 
                   label='Path Accuracy', color=self.colors['privacy_high'], alpha=0.8)
        axes[0].bar(x, df['Music Accuracy'], width, 
                   label='Music Accuracy', color=self.colors['privacy_low'], alpha=0.8)
        axes[0].bar(x + width, df['Combined Accuracy'], width, 
                   label='Combined Accuracy', color=self.colors['fedroute'], alpha=0.8)
        
        axes[0].set_xlabel('Method', fontsize=14)
        axes[0].set_ylabel('Accuracy', fontsize=14)
        axes[0].set_title('(a) Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df['Method'], rotation=30, ha='right')
        axes[0].legend(loc='upper right', framealpha=0.9)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 0.9])
        
        # Communication cost comparison
        axes[1].bar(x, df['Total Communication (GB)'], 
                   color=[self.colors['centralized'], self.colors['fedroute'], 
                         self.colors['fedavg'], self.colors['independent'], 
                         self.colors['random']], alpha=0.8)
        
        axes[1].set_xlabel('Method', fontsize=14)
        axes[1].set_ylabel('Total Communication (GB)', fontsize=14)
        axes[1].set_title('(b) Communication Cost', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df['Method'], rotation=30, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.figures_dir / 'figure3_baseline_comparison.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def figure4_ablation_study(self):
        """
        Figure 4: Ablation Study Results
        Shows impact of different components on performance.
        """
        print("\nGenerating Figure 4: Ablation Study...")
        
        df = pd.read_csv(self.results_dir / 'ablation_study.csv')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create grouped bar chart
        x = np.arange(len(df))
        
        # Color based on configuration
        colors_list = []
        for config in df['Configuration']:
            if 'Full' in config:
                colors_list.append(self.colors['fedroute'])
            elif 'Minimal' in config:
                colors_list.append(self.colors['random'])
            else:
                colors_list.append(self.colors['fedavg'])
        
        bars = ax.bar(x, df['Combined Accuracy'], color=colors_list, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add component indicators
        for i, row in df.iterrows():
            y_pos = 0.05
            text = f"MT: {row['Multi-Task']}\nSel: {row['Advanced Selection']}\nPriv: {row['Privacy']}"
            ax.text(i, y_pos, text, ha='center', va='bottom', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Configuration', fontsize=14)
        ax.set_ylabel('Combined Accuracy', fontsize=14)
        ax.set_title('Ablation Study: Component Impact Analysis', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Configuration'], rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 0.85])
        
        plt.tight_layout()
        output_path = self.figures_dir / 'figure4_ablation_study.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def figure5_scalability_analysis(self):
        """
        Figure 5: Scalability Analysis
        Shows system performance with increasing number of clients.
        """
        print("\nGenerating Figure 5: Scalability Analysis...")
        
        df = pd.read_csv(self.results_dir / 'scalability_analysis.csv')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy vs Number of Clients
        axes[0].plot(df['Num Clients'], df['Final Accuracy'], 
                    marker='o', markersize=10, linewidth=2.5, 
                    color=self.colors['fedroute'])
        axes[0].set_xlabel('Number of Clients', fontsize=14)
        axes[0].set_ylabel('Final Accuracy', fontsize=14)
        axes[0].set_title('(a) Accuracy Scalability', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0.70, 0.85])
        
        # Time per Round vs Number of Clients
        axes[1].plot(df['Num Clients'], df['Avg Time per Round (s)'], 
                    marker='s', markersize=10, linewidth=2.5, 
                    color=self.colors['privacy_medium'])
        axes[1].set_xlabel('Number of Clients', fontsize=14)
        axes[1].set_ylabel('Avg Time per Round (s)', fontsize=14)
        axes[1].set_title('(b) Time Complexity', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Throughput vs Number of Clients
        axes[2].plot(df['Num Clients'], df['Throughput (clients/s)'], 
                    marker='^', markersize=10, linewidth=2.5, 
                    color=self.colors['privacy_low'])
        axes[2].set_xlabel('Number of Clients', fontsize=14)
        axes[2].set_ylabel('Throughput (clients/s)', fontsize=14)
        axes[2].set_title('(c) System Throughput', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.figures_dir / 'figure5_scalability_analysis.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    def generate_all_figures(self):
        """
        Generate all figures for the paper.
        """
        print("="*60)
        print("GENERATING JOURNAL PAPER FIGURES")
        print("="*60)
        
        self.figure1_convergence_analysis()
        self.figure2_privacy_utility_tradeoff()
        self.figure3_baseline_comparison()
        self.figure4_ablation_study()
        self.figure5_scalability_analysis()
        
        print("\n" + "="*60)
        print("ALL FIGURES GENERATED!")
        print("="*60)
        print(f"Figures saved to: {self.figures_dir}")
        print("\nGenerated files:")
        for pdf_file in sorted(self.figures_dir.glob("*.pdf")):
            print(f"  - {pdf_file.name}")
        print("="*60)


def main():
    """Main function to generate all figures."""
    
    base_dir = Path(__file__).parent.parent.parent
    results_dir = base_dir / 'results' / 'experiments'
    figures_dir = base_dir / 'results' / 'figures'
    
    generator = PaperFigureGenerator(
        results_dir=str(results_dir),
        figures_dir=str(figures_dir)
    )
    
    generator.generate_all_figures()


if __name__ == '__main__':
    main()



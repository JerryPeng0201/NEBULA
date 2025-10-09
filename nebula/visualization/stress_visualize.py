import matplotlib.pyplot as plt
import numpy as np
import os


def plot_stress_adaptability(models, inference_freq, latency, stability, adaptability, 
                              save_path, adapt_models=None):
    """
    Plot stress test and adaptability results in a 2x2 layout
    
    Parameters:
        models: list of str, model names for stress tests (inference, latency, stability)
        inference_freq: dict, inference frequency data for different versions
                       e.g., {'V1': [17.33, 4.9, ...], 'V2': [...], 'V3': [...]}
        latency: dict, latency data for different versions
                e.g., {'V1': [58, 206, ...], 'V2': [...], 'V3': [...]}
        stability: dict, stability score data for different versions
                  e.g., {'V1': [0.94, 0.97, ...], 'V2': [...], 'V3': [...]}
        adaptability: dict, adaptability data with test types as keys
                     e.g., {'Object Movement': [36, 0, 0], 'Language Change': [32, 4, 0], ...}
        save_path: str, path to save the figure
        adapt_models: list of str, optional, model names for adaptability (if different from models)
                     If None, uses the same models as stress tests
    """
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # Use same models for adaptability if not specified
    if adapt_models is None:
        adapt_models = models
    
    # Set up the plotting parameters
    x = np.arange(len(models))
    adapt_x = np.arange(len(adapt_models))
    width = 0.25  # Width of bars
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Stress Test Results', fontsize=18, fontweight='bold', y=0.98)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Colors for V1, V2, V3 or adaptability test types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Get version keys (V1, V2, V3, etc.)
    versions = list(inference_freq.keys())
    
    # Helper function to shorten model names
    def shorten_name(name):
        return name.replace('GR00T-1.5', 'GR00T').replace('Diffusion Policy', 'DP')
    
    # Chart 1: Inference Frequency (top-left)
    ax1 = axes[0]
    bars1 = []
    for i, version in enumerate(versions):
        bars = ax1.bar(x + (i - 1) * width, inference_freq[version], width, 
                      label=version, alpha=0.8, color=colors[i])
        bars1.append(bars)
    
    ax1.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Inference Frequency (Hz)', fontweight='bold', fontsize=12)
    ax1.set_title('Inference Frequency', fontweight='bold', pad=20, fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([shorten_name(m) for m in models],
                        fontweight='bold', rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in bars1:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Latency (top-right)
    ax2 = axes[1]
    bars2 = []
    for i, version in enumerate(versions):
        bars = ax2.bar(x + (i - 1) * width, latency[version], width,
                      label=version, alpha=0.8, color=colors[i])
        bars2.append(bars)
    
    ax2.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    ax2.set_title('Latency', fontweight='bold', pad=20, fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([shorten_name(m) for m in models],
                        fontweight='bold', rotation=45, ha='right', fontsize=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in bars2:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Chart 3: Stability Score (bottom-left)
    ax3 = axes[2]
    bars3 = []
    for i, version in enumerate(versions):
        bars = ax3.bar(x + (i - 1) * width, stability[version], width,
                      label=version, alpha=0.8, color=colors[i])
        bars3.append(bars)
    
    ax3.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Stability Score', fontweight='bold', fontsize=12)
    ax3.set_title('Stability Score', fontweight='bold', pad=20, fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels([shorten_name(m) for m in models],
                        fontweight='bold', rotation=45, ha='right', fontsize=10)
    ax3.legend(fontsize=10, loc='lower left')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0.8, 1.0)
    
    # Add value labels on bars
    for bars in bars3:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Chart 4: Adaptability (bottom-right)
    ax4 = axes[3]
    
    # Get adaptability test types and data
    adapt_types = list(adaptability.keys())
    bars4 = []
    
    for i, test_type in enumerate(adapt_types):
        bars = ax4.bar(adapt_x + (i - 1) * width, adaptability[test_type], width,
                      label=test_type, alpha=0.8, color=colors[i])
        bars4.append(bars)
    
    ax4.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
    ax4.set_title('Adaptability', fontweight='bold', pad=20, fontsize=14)
    ax4.set_xticks(adapt_x)
    ax4.set_xticklabels([shorten_name(m) for m in adapt_models],
                        fontweight='bold', rotation=45, ha='right', fontsize=10)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(axis='y', alpha=0.3)
    
    # Set y-axis limit dynamically based on max value
    max_adapt = max(max(values) for values in adaptability.values())
    ax4.set_ylim(0, max_adapt * 1.2)
    
    # Add value labels on bars for adaptability (including 0 values)
    for bars, test_type in zip(bars4, adapt_types):
        for bar, value in zip(bars, adaptability[test_type]):
            ax4.annotate(f'{int(value)}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    # Add subtle background to distinguish adaptability section
    ax4.set_facecolor('#f9f9f9')
    
    # Adjust layout to prevent overlap
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    except:
        pass
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {save_path}")
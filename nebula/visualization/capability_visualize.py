import numpy as np
import matplotlib.pyplot as plt
import os

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors with high contrast
    
    Parameters:
        n: int, number of colors to generate
    
    Returns:
        list of colors in RGBA format
    """
    if n <= 0:
        return []
    
    # Predefined high-contrast colors for common cases
    if n <= 12:
        # Carefully selected colors with good contrast
        base_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
            '#ff1493',  # Deep Pink
            '#00ff00',  # Lime
        ]
        return base_colors[:n]
    
    # For more colors, use HSV color space with optimized parameters
    colors = []
    
    # Use golden ratio for better distribution
    golden_ratio = 0.618033988749895
    hue = np.random.random()  # Start with random hue
    
    for i in range(n):
        hue += golden_ratio
        hue %= 1.0
        
        # Vary saturation and value for better distinction
        # Keep saturation and value high for vibrant colors
        saturation = 0.7 + 0.3 * (i % 2)  # Alternate between 0.7 and 1.0
        value = 0.8 + 0.2 * ((i // 2) % 2)  # Alternate between 0.8 and 1.0
        
        # Convert HSV to RGB
        import colorsys
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append('#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        ))
    
    return colors


def plot_radar_comparison(all_scores, model_names, save_path, labels=None, mode="separate"):
    """
    Plot radar chart comparison for multiple models with Easy/Medium/Hard difficulty levels
    
    Parameters:
        all_scores: list of tuples, each tuple contains 3 lists (easy, medium, hard)
                   e.g., [([83, 100, 92, ...], [33, 100, 86, ...], [8, 100, 79, ...]), ...]
        model_names: list of str, list of model names
        save_path: str, path to save the figure
        labels: list of str, optional, labels for each dimension, 
                default: ["Control", "Perception", "Language", "Spatial", "Dynamic", "Robustness"]
        mode: str, "separate" or "combined"
              "separate": generate separate radar chart for each model (default)
              "combined": overlay all models in three radar charts (one for each difficulty level)
    """
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    if labels is None:
        labels = ["Control", "Perception", "Language", "Spatial", "Dynamic", "Robustness"]
    
    n_vars = len(labels)
    n_models = len(model_names)
    
    def close(vals):
        """Close the radar chart by appending the first value to the end"""
        return np.concatenate([vals, vals[:1]])
    
    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    
    # Generate distinct colors
    colors = generate_distinct_colors(n_models)
    
    # Difficulty level names
    difficulty_levels = ["Easy", "Medium", "Hard"]
    
    if mode == "combined":
        # Combined mode: 1 row x 3 columns (one subplot for each difficulty level)
        fig, axs = plt.subplots(
            1, 3, subplot_kw=dict(polar=True), figsize=(60, 20),
            gridspec_kw={"hspace": 0.4, "wspace": 0.4}
        )
        
        # Global legend elements
        global_legend_lines = []
        
        for difficulty_idx, (ax, difficulty_name) in enumerate(zip(axs, difficulty_levels)):
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(angles[:-1] * 180 / np.pi, [""] * len(labels))
            
            # Add custom labels at specific positions
            for angle, label in zip(angles[:-1], labels):
                ax.text(angle, 120, label, ha="center", va="center", fontweight="bold", fontsize=36)
            
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(["20", "40", "60", "80", "100"], fontweight="bold", fontsize=18)
            
            # Plot each model with different color
            for model_idx, model_name in enumerate(model_names):
                # Get scores for this model at current difficulty level
                model_scores = all_scores[model_idx][difficulty_idx]
                
                v = close(model_scores)
                line, = ax.plot(angles, v, 'o-', linewidth=2.5, color=colors[model_idx],
                               label=model_name, markersize=10)
                ax.fill(angles, v, alpha=0.15, color=colors[model_idx])
                
                # Collect legend lines only from the first subplot (avoid duplication)
                if difficulty_idx == 0:
                    global_legend_lines.append(line)
            
            # Set subplot title
            ax.set_title(f"{difficulty_name} Level", fontsize=40, fontweight="bold", pad=150)
        
        # Determine number of columns for legend based on number of models
        legend_ncol = min(n_models, 6)  # Maximum 6 columns, adjust if more models
        
        # Add global legend at the bottom, horizontal layout
        fig.legend(
            handles=global_legend_lines,
            labels=model_names,
            loc="lower center",
            ncol=legend_ncol,
            fontsize=30,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
            prop={'size': 30, 'weight': 'bold'}
        )
        
    else:
        # Separate mode: each model gets its own subplot
        # Calculate optimal grid layout based on number of models
        n_cols = min(3, n_models)  # Maximum 3 columns
        n_rows = int(np.ceil(n_models / n_cols))
        
        # Adjust figure size based on grid dimensions
        fig_width = n_cols * 20
        fig_height = n_rows * 20
        
        fig, axs = plt.subplots(
            n_rows, n_cols, subplot_kw=dict(polar=True), figsize=(fig_width, fig_height),
            gridspec_kw={"hspace": 0.4, "wspace": 0.4}
        )
        
        # Handle case where there's only one subplot
        if n_models == 1:
            axs = np.array([axs])
        
        # Flatten axes array for easier iteration
        axs = axs.flatten() if n_models > 1 else axs
        
        # Store legend line artists for global legend
        legend_lines = []
        
        for i, (easy, medium, hard) in enumerate(all_scores):
            ax = axs[i]
            # Set starting angle to top (90 degrees)
            ax.set_theta_offset(np.pi / 2)
            # Set clockwise direction
            ax.set_theta_direction(-1)
            # Remove default tick labels
            ax.set_thetagrids(angles[:-1] * 180 / np.pi, [""] * len(labels))
            
            # Add custom labels at specific positions
            for angle, label in zip(angles[:-1], labels):
                ax.text(angle, 120, label, ha="center", va="center", fontweight="bold", fontsize=36)
            
            # Set radial axis properties
            ax.set_rlabel_position(0)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(["20", "40", "60", "80", "100"], fontweight="bold", fontsize=18)
            
            # Plot three difficulty levels
            for name, vals in [("Easy", easy), ("Medium", medium), ("Hard", hard)]:
                v = close(vals)
                line, = ax.plot(angles, v, linewidth=2, label=name)
                ax.fill(angles, v, alpha=0.1)
                
                # Collect legend lines only from the first subplot (avoid duplication)
                if i == 0:
                    legend_lines.append(line)
            
            # Set subplot title
            ax.set_title(model_names[i], fontsize=40, fontweight="bold", pad=150)
        
        # Hide unused subplots if any
        total_subplots = n_rows * n_cols
        for i in range(n_models, total_subplots):
            axs[i].set_visible(False)
        
        # Adjust legend position: move down when there's only one model
        if n_models == 1:
            bbox_anchor = (0.5, -0.05)  # Move legend down to avoid overlap
        else:
            bbox_anchor = (0.5, -0.005)
        
        # Add global legend at the bottom, horizontal layout
        fig.legend(
            handles=legend_lines,
            labels=["Easy", "Medium", "Hard"],
            loc="lower center",
            ncol=3,
            fontsize=40,
            frameon=False,
            bbox_to_anchor=bbox_anchor,
            prop={'size': 40, 'weight': 'bold'}  # Bold font
        )
    
    # Save figure
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)  # Close figure to avoid displaying
    print(f"Figure saved to: {save_path}")
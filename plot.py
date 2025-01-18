import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend (no GUI)
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_candles(
    experiments,
    base_path="",
    title="Accuracy Candle Plot",
    output_filename="candle_plot.png"
):
    """
    Plot final-spoke-accuracy distributions from multiple HSL/P2P experiments as "candles."
    
    Each experiment in the experiments list is a dictionary with keys:
      - 'filename': str, filename of the metrics JSON.
      - 'aggregator': str, either "p2p"/"p2p_local" (for P2P) or "hsl" (for HSL).
      - 'cost': float, cost value (x-axis location).
      - 'config': tuple, configuration tuple (e.g. (n_s, k) for P2P or (n_s, n_h, b_hs, b_hh, b_sh) for HSL).
    
    For each experiment, the function extracts the final evaluation (last element) from:
      - data['local_acc'] for P2P experiments, or
      - data['spoke_acc'] for HSL experiments.
    
    It computes the minimum, 25th percentile (q1), 75th percentile (q3), and maximum.
    The candle body spans from the 25th to the 75th percentile, and the
    wicks extend from min to q1 and from q3 to max.
    
    Additionally, the range values for each configuration are printed to the console,
    and the configuration is printed horizontally above (for P2P) or below (for HSL)
    the candle.
    """
    
    # Create a figure and a single subplot without relying on a display.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Colors for aggregator types.
    color_map = {
        'p2p': 'orange',
        'p2p_local': 'orange',
        'hsl': 'blue'
    }
    
    # Slight x-axis shifts to help distinguish overlapping candles if costs are similar.
    x_shift_map = {
        'p2p': -0.06,
        'p2p_local': -0.06,
        'hsl': +0.06
    }
    
    # Legend labels.
    aggregator_labels = {
        'p2p': 'P2P/Epidemic',
        'p2p_local': 'P2P/Epidemic (Local)',
        'hsl': 'HSL'
    }
    
    used_legend_labels = set()
    costs_record = []  # to record x values for setting x-axis limits later
    all_y_vals = []   # to record all y-values, if needed for y-axis limits
    
    for exp in experiments:
        filename = exp['filename']
        aggregator = exp['aggregator']
        cost = exp['cost']
        config_tuple = exp['config']
        
        fullpath = os.path.join(base_path, filename)
        if not os.path.exists(fullpath):
            print(f"[Warning] File not found: {fullpath}")
            continue
        
        # Read the JSON file containing the metrics.
        with open(fullpath, 'r') as f:
            data = json.load(f)
        
        # Read the final accuracy distribution based on aggregator type.
        if aggregator in ['p2p', 'p2p_local']:
            if 'local_acc' not in data or len(data['local_acc']) == 0:
                print(f"[Warning] No 'local_acc' found in {filename}. Skipping.")
                continue
            all_spoke_acc = data['local_acc'][-1]
        elif aggregator == 'hsl':
            if 'spoke_acc' not in data or len(data['spoke_acc']) == 0:
                print(f"[Warning] No 'spoke_acc' found in {filename}. Skipping.")
                continue
            all_spoke_acc = data['spoke_acc'][-1]
        else:
            print(f"[Warning] Unrecognized aggregator '{aggregator}' in {filename}. Skipping.")
            continue
        
        # Convert the accuracies into a NumPy array.
        all_spoke_acc = np.array(all_spoke_acc, dtype=float)
        
        # Compute the statistics for the candle.
        min_val = np.min(all_spoke_acc)
        max_val = np.max(all_spoke_acc)
        q1 = np.percentile(all_spoke_acc, 25)
        q3 = np.percentile(all_spoke_acc, 75)
        
        # Record y-values in case you want to adjust y-axis limits later.
        all_y_vals.extend([min_val, q1, q3, max_val])
        
        # Print the range values for this configuration.
        print(f"Config {config_tuple} from file '{filename}':")
        print(f"  Min: {min_val:.4f}, 25th Percentile: {q1:.4f}, 75th Percentile: {q3:.4f}, Max: {max_val:.4f}")
        
        # Calculate the x-axis position (with a slight shift based on aggregator).
        x = cost + x_shift_map.get(aggregator, 0.0)
        costs_record.append(x)
        candle_color = color_map.get(aggregator, 'gray')
        
        # Draw lower wick if there is a visible difference.
        if q1 > min_val:
            ax.plot([x, x], [min_val, q1], color=candle_color, alpha=0.8, linewidth=2)
            ax.plot([x - 0.03, x + 0.03], [min_val, min_val],
                    color=candle_color, alpha=0.8, linewidth=1)
        # Draw upper wick if there is a visible difference.
        if max_val > q3:
            ax.plot([x, x], [q3, max_val], color=candle_color, alpha=0.8, linewidth=2)
            ax.plot([x - 0.03, x + 0.03], [max_val, max_val],
                    color=candle_color, alpha=0.8, linewidth=1)
        
        # Increase the body width so it becomes visible.
        half_width = 8.00
        ax.fill_between([x - half_width, x + half_width],
                        y1=q1, y2=q3,
                        color=candle_color, alpha=0.4, linewidth=0)
        
        # Print configuration text horizontally.
        if aggregator == 'hsl':
            y_text = min_val - 0.02  # place below the candle
            va_opt = 'top'
        else:
            y_text = max_val + 0.02  # place above the candle
            va_opt = 'bottom'
        
        ax.text(x, y_text, f"{config_tuple}",
                ha='center', va=va_opt, rotation=0, fontsize=9,
                color=candle_color, alpha=0.9)
        
        # Add legend handle for this aggregator only once.
        label_for_legend = aggregator_labels.get(aggregator, aggregator)
        if label_for_legend not in used_legend_labels:
            ax.plot([], [], color=candle_color, label=label_for_legend,
                    alpha=0.7, linewidth=8)
            used_legend_labels.add(label_for_legend)
    
    # Optionally, manually set the x-axis limits using the recorded cost values.
    if costs_record:
        margin = 50  # adjust margin as needed
        ax.set_xlim(min(costs_record) - margin, max(costs_record) + margin)
    
    # Optionally, set the y-axis limits (if needed) using all_y_vals.
    if all_y_vals:
        y_margin = 0.02  # adjust as appropriate; assuming accuracy is between 0 and 1.
        ax.set_ylim(min(all_y_vals) - y_margin, max(all_y_vals) + y_margin)
    
    # Set labels, title, grid, and legend.
    ax.set_xlabel("Cost")
    ax.set_ylabel("Final Spoke Accuracy")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    print(f"[Info] Plot saved to {output_filename}")

if __name__ == "__main__":
    """
    Example usage:
    
    Suppose you have several experiments:
      - Multiple P2P experiments with varying k values and corresponding costs.
      - One HSL experiment.
    
    The following definitions use:
      1) "ell_cifar10_s100k8_seed1_metrics.json" with config (100,8) and cost 400
      2) "ell_cifar10_s100k5_seed1_metrics.json" with config (100,5) and cost 250
      3) "ell_cifar10_s100k4_seed1_metrics.json" with config (100,4) and cost 200
      4) "ell_cifar10_s100k10_seed1_metrics.json" with config (100,10) and cost 500
      5) "ell_cifar10_s100k20_seed1_metrics.json" with config (100,20) and cost 1000
      6) "hsl_cifar10_s100h10_bud_2_1_1_seed1_metrics.json" with config (100,10,2,1,1) and cost 140
    """
    experiments = [
        {
            'filename': "ell_cifar10_s100k1_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 50,
            'config': (100, 1)
        },
        {
            'filename': "ell_cifar10_s100k2_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 100,
            'config': (100, 2)
        },
        {
            'filename': "ell_cifar10_s100k3_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 150,
            'config': (100, 3)
        },
        {
            'filename': "ell_cifar10_s100k4_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 200,
            'config': (100, 4)
        },
        {
            'filename': "ell_cifar10_s100k5_seed2_metrics.json",
            'aggregator': "p2p",
            'cost': 250,
            'config': (100, 5)
        },
        {
            'filename': "ell_cifar10_s100k8_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 400,
            'config': (100, 8)
        },  
        {
            'filename': "ell_cifar10_s100k10_seed2_metrics.json",
            'aggregator': "p2p",
            'cost': 500,
            'config': (100, 10)
        },
        {
            'filename': "ell_cifar10_s100k15_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 750,
            'config': (100, 15)
        },
        {
            'filename': "ell_cifar10_s100k20_seed2_metrics.json",
            'aggregator': "p2p",
            'cost': 1000,
            'config': (100, 20)
        },

        #############HSL####################

        {
            'filename': "hsl_cifar10_s100h10_bud_2_1_1_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 140,
            'config': (100, 10, 2, 1, 1)
        },
        {
            'filename': "hsl_cifar10_s100h20_bud_2_1_2_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 180,
            'config': (100, 20, 2, 1, 1)
        },
        {
            'filename': "hsl_cifar10_s100h10_bud_2_2_2_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 250,
            'config': (100, 10, 2, 2, 2)
        },
        #{
        #    'filename': "hsl_cifar10_s100h5_bud_2_2_2_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 225,
        #    'config': (100, 5, 2, 2, 2)
        #},
        {
            'filename': "hsl_cifar10_s100h10_bud_8_2_2_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 320,
            'config': (100, 10, 8, 2, 2)
        },
        {
            'filename': "hsl_cifar10_s100h10_bud_20_2_2_seed2_metrics.json",
            'aggregator': "hsl",
            'cost': 430,
            'config': (100, 10, 20, 2, 2)
        },
        #{
        #    'filename': "hsl_cifar10_s100h10_bud_30_2_2_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 530,
        #    'config': (100, 10, 30, 2, 2)
        #},
        {
            'filename': "hsl_cifar10_s100h10_b1l5_bud_30_3_3_metrics.json",
            'aggregator': "hsl",
            'cost': 640,
            'config': (100, 15, 20, 3, 3)
        },
        {
            'filename': "hsl_cifar10_s100h20_bud_25_3_2_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 780,
            'config': (100, 20, 25, 3, 2)
        },
        
        
        

    ]
    
    plot_candles(
        experiments,
        base_path="./outputs/",  # Adjust this to your JSON files directory
        title="HSL vs P2P: Final Accuracy Distribution",
        output_filename="final_candle_plot.png"
    )

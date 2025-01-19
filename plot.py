import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend (no GUI)
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter

def plot_candles(
    experiments,
    base_path="",
    title="Accuracy Candle Plot",
    output_filename="candle_plot.png"
):
    """
    Experiment 1: Plot final-spoke-accuracy distributions from HSL / P2P experiments as "candles."
    
    Each experiment in `experiments` is a dict with:
      - 'filename'   (str): name of the JSON metrics file.
      - 'aggregator' (str): in {'p2p', 'p2p_local', 'hsl'}.
      - 'cost'       (float): x-axis coordinate (avg #edges).
      - 'config'     (tuple): config parameters, e.g., (100, k) or (100, n_h, b_hs, b_hh, b_sh).
    
    We extract the final accuracy array from:
      - data['local_acc'] (last entry) for 'p2p'/'p2p_local',
      - data['spoke_acc'] (last entry) for 'hsl'.
    
    We then compute min, 25%, 75%, and max for candle plotting.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Color scheme for the aggregators:
    color_map = {
        'p2p': 'orange',
        'p2p_local': 'orange',
        'hsl': 'blue'
    }
    # Slight shift so HSL or ELL points with close 'cost' won't overlap too much.
    x_shift_map = {
        'p2p': -0.06,
        'p2p_local': -0.06,
        'hsl': +0.06
    }
    # Labels for legend (merged P2P + p2p_local into "EL Local").
    aggregator_legend_label = {
        'p2p': 'EL Local',
        'p2p_local': 'EL Local',
        'hsl': 'HSL'
    }
    
    used_legend_labels = set()
    costs_record = []
    all_y_vals = []
    
    for exp in experiments:
        filename = exp['filename']
        aggregator = exp['aggregator']
        cost = exp['cost']
        config_tuple = exp['config']
        
        fullpath = os.path.join(base_path, filename)
        if not os.path.exists(fullpath):
            print(f"[Warning] File not found: {fullpath}")
            continue
        
        # Load JSON metrics
        with open(fullpath, 'r') as f:
            data = json.load(f)
        
        # Determine final accuracy distribution
        if aggregator in ['p2p', 'p2p_local']:
            if 'local_acc' not in data or len(data['local_acc']) == 0:
                print(f"[Warning] No 'local_acc' found in {filename}. Skipping candle plot.")
                continue
            all_spoke_acc = data['local_acc'][-1]
        elif aggregator == 'hsl':
            if 'spoke_acc' not in data or len(data['spoke_acc']) == 0:
                print(f"[Warning] No 'spoke_acc' found in {filename}. Skipping candle plot.")
                continue
            all_spoke_acc = data['spoke_acc'][-1]
        else:
            print(f"[Warning] Unrecognized aggregator '{aggregator}' in {filename}. Skipping candle plot.")
            continue
        
        all_spoke_acc = np.array(all_spoke_acc, dtype=float)
        min_val = np.min(all_spoke_acc)
        q1 = np.percentile(all_spoke_acc, 25)
        q3 = np.percentile(all_spoke_acc, 75)
        max_val = np.max(all_spoke_acc)
        
        # Print stats
        print(f"Candle Plot - Config {config_tuple} from '{filename}':")
        print(f"  Min={min_val:.4f}, 25%={q1:.4f}, 75%={q3:.4f}, Max={max_val:.4f}")
        
        # Plot candle
        x = cost + x_shift_map.get(aggregator, 0.0)
        costs_record.append(x)
        candle_color = color_map.get(aggregator, 'gray')
        
        # Lower wick
        if q1 > min_val:
            ax.plot([x, x], [min_val, q1], color=candle_color, alpha=0.8, linewidth=2)
            ax.plot([x - 0.03, x + 0.03], [min_val, min_val],
                    color=candle_color, alpha=0.8, linewidth=1)
        # Upper wick
        if max_val > q3:
            ax.plot([x, x], [q3, max_val], color=candle_color, alpha=0.8, linewidth=2)
            ax.plot([x - 0.03, x + 0.03], [max_val, max_val],
                    color=candle_color, alpha=0.8, linewidth=1)
        
        # Candle body
        half_width = 8.0
        ax.fill_between([x - half_width, x + half_width],
                        y1=q1, y2=q3,
                        color=candle_color, alpha=0.4, linewidth=0)
        
        # Place config label
        if aggregator == 'hsl':
            y_text = min_val - 0.02
            va_opt = 'top'
        else:
            y_text = max_val + 0.02
            va_opt = 'bottom'
        
        ax.text(x, y_text, f"{config_tuple}",
                ha='center', va=va_opt, rotation=0, fontsize=9,
                color=candle_color, alpha=0.9)
        
        # Add aggregator label to legend once
        legend_label = aggregator_legend_label.get(aggregator, aggregator)
        if legend_label not in used_legend_labels:
            ax.plot([], [], color=candle_color, label=legend_label,
                    alpha=0.7, linewidth=8)
            used_legend_labels.add(legend_label)
        
        all_y_vals.extend([min_val, q1, q3, max_val])
    
    # Adjust x-range
    if costs_record:
        margin = 50
        ax.set_xlim(min(costs_record) - margin, max(costs_record) + margin)
    # Adjust y-range
    if all_y_vals:
        y_margin = 0.02
        ymin = min(all_y_vals) - y_margin
        ymax = max(all_y_vals) + y_margin
        ax.set_ylim(ymin, ymax)
    
    ax.set_xlabel("#edges in the graph")
    ax.set_ylabel("Final Spoke Accuracy")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    print(f"[Info] Candle plot saved to {output_filename}")


def plot_drift(
    experiments,
    base_path="",
    title="Drift Ratio (pre_drift / post_drift) vs. Time",
    output_filename="drift_plot.png"
):
    """
    Experiment 2: 
    - X-axis: 0..500 (time).
    - Y-axis: ratio = pre_drift / post_drift, displayed on a log scale.
      We'll rename the y-axis to: '-log(mixing ratio)'.
    - We want more space on the right to see the config tuples fully.
    - We want custom ticks in normal form (e.g., 0.1, 1, 5, 10, 20, 50, 100).
    - Finer grid lines, possibly a dashed minor grid. 
    """
    
    fig = plt.figure(figsize=(8, 6))  # widen figure
    ax = fig.add_subplot(111)
    
    aggregator_legend_label = {
        'p2p': 'EL Local',
        'p2p_local': 'EL Local',
        'hsl': 'HSL'
    }
    color_map = {
        'p2p': 'orange',
        'p2p_local': 'orange',
        'hsl': 'blue'
    }
    
    used_legend_labels = set()
    
    for exp in experiments:
        filename = exp['filename']
        aggregator = exp['aggregator']
        config_tuple = exp['config']
        
        fullpath = os.path.join(base_path, filename)
        if not os.path.exists(fullpath):
            print(f"[Warning] File not found: {fullpath}")
            continue
        
        with open(fullpath, 'r') as f:
            data = json.load(f)
        
        if 'pre_drift' not in data or 'post_drift' not in data:
            print(f"[Warning] {filename} missing 'pre_drift' or 'post_drift'. Skipping drift plot.")
            continue
        
        pre_drift_list = data['pre_drift']
        post_drift_list = data['post_drift']
        
        if len(pre_drift_list) != len(post_drift_list) or len(pre_drift_list) == 0:
            print(f"[Warning] {filename} has invalid drift arrays. Skipping drift plot.")
            continue
        
        # ratio = pre_drift / post_drift
        ratio_arr = np.array(pre_drift_list) / np.array(post_drift_list)
        
        # x from 0..500
        n_points = len(ratio_arr)
        x_vals = np.linspace(0, 500, n_points)
        
        color_ = color_map.get(aggregator, 'gray')
        legend_label = aggregator_legend_label.get(aggregator, aggregator)
        
        ax.plot(
            x_vals, ratio_arr,
            marker='o', linestyle='-',
            color=color_, alpha=0.9
        )
        
        # Put config tuple near the last data point
        x_end = x_vals[-1]
        y_end = ratio_arr[-1]
        # Shift further right so it doesn't get cut off
        ax.text(
            x_end + 15,  # +15 to ensure there's space for the tuple
            y_end,
            f"{config_tuple}",
            fontsize=9,
            color=color_,
            ha='left', va='center', alpha=0.9,
            clip_on=False  # allows text beyond plot area
        )
        
        # Legend entry
        if legend_label not in used_legend_labels:
            ax.plot([], [], color=color_,
                    label=legend_label, alpha=0.7, linewidth=8)
            used_legend_labels.add(legend_label)
    
    # Force x-axis 0..500
    ax.set_xlim(0, 550)  # extra space so text isn't cut off
    # Now use log scale on y-axis
    ax.set_yscale('log')
    
    # Define custom y ticks: e.g. 0.1, 1, 5, 10, 20, 50, 100...
    ticks = [1, 5, 10, 20, 50, 100]
    ax.set_yticks(ticks)
    # Show them in normal form
    ax.set_yticklabels([str(t) for t in ticks])
    
    # Use minor ticks for sub-steps
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[0.2, 0.3, 0.4, 0.6, 0.8]))
    ax.yaxis.set_minor_formatter(LogFormatter())
    
    # Grid
    ax.grid(which='major', linewidth=1.0)
    ax.grid(which='minor', linestyle='--', alpha=0.5)
    
    # Rename y-axis
    ax.set_ylabel("-log(mixing ratio)")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.legend()
    
    # Make extra room on the right to fit the text
    plt.subplots_adjust(right=0.85)
    
    plt.savefig(output_filename, dpi=200)
    print(f"[Info] Drift plot saved to {output_filename}")


if __name__ == "__main__":
    # Candle Plot (Experiment 1) dataset
    experiments_exp1 = [
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

        # HSL
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
            'filename': "hsl_cifar10_s100h5_bud_2_2_2_seed2_metrics.json",
            'aggregator': "hsl",
            'cost': 225,
            'config': (100, 5, 2, 2, 2)
        },
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
        {
            'filename': "hsl_cifar10_s100h10_bud_30_3_2_seed2_metrics.json",
            'aggregator': "hsl",
            'cost': 540,
            'config': (100, 10, 30, 3, 2)
        },
        {
            'filename': "hsl_cifar10_s100h10_bud_30_3_3_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 640,
            'config': (100, 15, 30, 3, 3)
        },
        {
            'filename': "hsl_cifar10_s100h10_bud_40_3_3_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 740,
            'config': (100, 10, 40, 3, 3)
        },
        {
            'filename': "hsl_cifar10_s100h10_bud_40_3_4_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 840,
            'config': (100, 10, 40, 3, 4)
        },
    ]
    
    # --- Experiment 1: Candle plot for final accuracies ---
    plot_candles(
        experiments_exp1,
        base_path="./outputs/",
        title="HSL vs P2P: Final Accuracy Distribution",
        output_filename="final_candle_plot.png"
    )
    
    # --- Experiment 2: Drift ratio plot with new modifications ---
    experiments_exp2 = experiments_exp1  # Reuse the same list, so ensure JSONs have pre_drift/post_drift
    plot_drift(
        experiments_exp2,
        base_path="./outputs/",
        title="Drift Ratio (pre_drift / post_drift) vs. Time",
        output_filename="drift_plot.png"
    )

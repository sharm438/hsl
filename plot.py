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
    output_filename="candle_plot.png",
    half_width=8,
    xlim=[]
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
        'p2p': 'peru',
        'p2p_local': 'tab:red',
        'hsl': 'forestgreen'
    }
    # Slight shift so HSL or ELL points with close 'cost' won't overlap too much.
    x_shift_map = {
        'p2p': -0.06,
        'p2p_local': -0.06,
        'hsl': +0.06
    }
    # Labels for legend (merged P2P + p2p_local into "EL Local").
    aggregator_legend_label = {
        'p2p': 'EL Local ($n_s, k$)',
        'p2p_local': 'EL Local ($n_s, k$)',
        'hsl': 'HSL ($n_s, n_h, b_{hs}, b_{hh}, b_{sh}$)'
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
        #half_width = 8#*5
        ax.fill_between([x - half_width, x + half_width],
                        y1=q1, y2=q3,
                        color=candle_color, alpha=0.8, linewidth=0)
        
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
    #if costs_record:
    #    margin = 50
    #    ax.set_xlim(min(costs_record) - margin, max(costs_record) + margin)
    # Adjust y-range
    ax.set_xlim(xlim[0], xlim[1])
    if all_y_vals:
        y_margin = 0.02
        ymin = min(all_y_vals) - y_margin
        ymax = max(all_y_vals) + y_margin
        ax.set_ylim(ymin, ymax)
    
    ax.set_xlabel("#edges in the graph", fontsize=14)
    ax.set_ylabel("Final Spoke Accuracy", fontsize=14)
    #ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    print(f"[Info] Candle plot saved to {output_filename}")


def plot_drift(
    experiments,
    base_path="",
    title="Drift Ratio (pre_drift / post_drift) vs. Time",
    output_filename="drift_plot.png",
    yticks=[],
    ylim=[0,2]
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
        'p2p': 'EL Local ($n_s, k$)',
        'p2p_local': 'EL Local ($n_s, k$)',
        'hsl': 'HSL ($n_s, n_h, b_{hs}, b_{hh}, b_{sh}$)'
    }
    color_map = {
        'p2p': 'peru',
        'p2p_local': 'tab:red',
        'hsl': 'forestgreen'
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
        ratio_log = np.log10(ratio_arr)
        
        # x from 0..500
        n_points = len(ratio_arr)
        x_vals = np.linspace(0, 500, n_points)
        
        color_ = color_map.get(aggregator, 'gray')
        legend_label = aggregator_legend_label.get(aggregator, aggregator)
        
        ax.plot(
            x_vals, ratio_log,
            marker='o', linestyle='-',
            color=color_, alpha=0.9
        )
        
        # Put config tuple near the last data point
        x_end = x_vals[-1]
        y_end = ratio_log[-1]
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
    ticks = yticks#[0.25, 0.5, 1, 2]
    ax.set_yticks(ticks)
    # Show them in normal form
    ax.set_yticklabels([str(t) for t in ticks])
    #ax.set_ylim(ylim)
    
    # Use minor ticks for sub-steps
    #ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[0.2, 0.3, 0.4, 0.6]))
    #ax.yaxis.set_minor_formatter(LogFormatter())
    from matplotlib.ticker import FixedLocator
    ax.yaxis.set_minor_locator(FixedLocator([]))
    # Grid
    ax.grid(which='major', linewidth=1.0)
    ax.grid(which='minor', linestyle='--', alpha=0.5)
    
    # Rename y-axis
    ax.set_ylabel("$-log_{10}(CDR)$", fontsize=14)
    ax.set_xlabel("Time", fontsize=14)
    #ax.set_title(title)
    ax.legend(fontsize=12)
    
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
        #{
        #    'filename': "ell_cifar10_s100k3_seed1_metrics.json",
        #    'aggregator': "p2p",
        #    'cost': 150,
        #    'config': (100, 3)
        #},
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
            'filename': "ell_cifar10_s100k7_seed2_metrics.json",
            'aggregator': "p2p",
            'cost': 350,
            'config': (100, 7)
        },
        #{
        #    'filename': "ell_cifar10_s100k8_seed1_metrics.json",
        #    'aggregator': "p2p",
        #    'cost': 400,
        #    'config': (100, 8)
        #},
        {
            'filename': "ell_cifar10_s100k10_seed3_metrics.json",
            'aggregator': "p2p",
            'cost': 500,
            'config': (100, 10)
        },
        {
            'filename': "ell_cifar10_s100k13_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 650,
            'config': (100, 13)
        },
        {
            'filename': "ell_cifar10_s100k15_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 750,
            'config': (100, 15)
        },
        {
            'filename': "ell_cifar10_s100k18_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 900,
            'config': (100, 18)
        },
        {
            'filename': "ell_cifar10_s100k20_seed5_metrics.json",
            'aggregator': "p2p",
            'cost': 1000,
            'config': (100, 20)
        },
        #{
        #    'filename': "ell_cifar10_s100k22_seed2_metrics.json",
        #    'aggregator': "p2p",
        #    'cost': 1100,
        #    'config': (100, 22)
        #},
        #{
        #    'filename': "ell_cifar10_s100k24_seed1_metrics.json",
        #    'aggregator': "p2p",
        #    'cost': 1200,
        #    'config': (100, 24)
        #},
        

        # HSL
        {
            'filename': "hsl_cifar10_s100h10_bud_1_1_1_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 115,
            'config': (100, 10, 1, 1, 1)
        },
        #{
        #    'filename': "hsl_cifar10_s100h10_bud_1_2_1_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 120,
        #    'config': (100, 10, 1, 2, 1)
        #},
        #{
        #    'filename': "hsl_cifar10_s100h10_bud_2_1_1_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 125,
        #    'config': (100, 10, 2, 1, 1)
        #},
        {
            'filename': "hsl_cifar10_s100h20_bud_2_1_2_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 150,
            'config': (100, 20, 2, 1, 1)
        },
        {
            'filename': "hsl_cifar10_s100h5_bud_2_2_2_seed2_metrics.json",
            'aggregator': "hsl",
            'cost': 215,
            'config': (100, 5, 2, 2, 2)
        },
        #{
        #    'filename': "hsl_cifar10_s100h10_bud_8_2_2_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 290,
        #    'config': (100, 10, 8, 2, 2)
        #},
        {
            'filename': "hsl_cifar10_s100h20_bud_10_5_1_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 350,
            'config': (100, 20, 10, 5, 1)
        },
        #{
        #    'filename': "hsl_cifar10_s100h10_bud_20_2_2_seed2_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 410,
        #    'config': (100, 10, 20, 2, 2)
        #},
        {
            'filename': "hsl_cifar10_s100h10_bud_30_3_2_seed2_metrics.json",
            'aggregator': "hsl",
            'cost': 515,
            'config': (100, 10, 30, 3, 2)
        },
        {
            'filename': "hsl_cifar10_s100h10_bud_30_3_3_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 615,
            'config': (100, 10, 30, 3, 3)
        },
        #{
        #    'filename': "hsl_cifar10_s100h10_bud_40_3_3_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 715,
        #    'config': (100, 10, 40, 3, 3)
        #},
        {
            'filename': "hsl_cifar10_s100h10_bud_40_3_4_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 815,
            'config': (100, 10, 40, 3, 4)
        },
        #{
        #    'filename': "hsl_cifar10_s100h10_bud_40_5_5_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 925,
        #    'config': (100, 10, 40, 5, 5)
        #},
        
        {
            'filename': "hsl_cifar10_s100h15_bud_30_20_4_seed3_metrics.json",
            'aggregator': "hsl",
            'cost': 1000,
            'config': (100, 15, 30, 20, 4)
        },
        #{
        #    'filename': "hsl_cifar10_s100h20_bud_30_20_3_seed2_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 1100,
        #    'config': (100, 20, 30, 20, 3)
        #},
        #{
        #    'filename': "hsl_cifar10_s100h20_bud_30_20_4_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 1200,
        #    'config': (100, 20, 30, 20, 4)
        #},
        
        
    ]
    
    experiments_exp1b = [
        {
            'filename': "ell_cifar10_s200k2_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 200,
            'config': (200, 2)
        },
        {
            'filename': "ell_cifar10_s200k5_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 500,
            'config': (200, 5)
        },
        {
            'filename': "ell_cifar10_s200k10_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 1000,
            'config': (200, 10)
        },
        {
            'filename': "ell_cifar10_s200k15_seed1_metrics.json",
            'aggregator': "p2p",
            'cost': 1500,
            'config': (200, 15)
        },
        {
            'filename': "ell_cifar10_s200k20_seed2_metrics.json",
            'aggregator': "p2p",
            'cost': 2000,
            'config': (200, 20)
        },
        {
            'filename': "ell_cifar10_s200k25_seed3_metrics.json",
            'aggregator': "p2p",
            'cost': 2500,
            'config': (200, 25)
        },
        {
            'filename': "ell_cifar10_s200k30_seed2_metrics.json",
            'aggregator': "p2p",
            'cost': 3000,
            'config': (200, 30)
        },
        #{
        #    'filename': "ell_cifar10_s200k35_seed2_metrics.json",
        #    'aggregator': "p2p",
        #    'cost': 3500,
        #    'config': (200, 35)
        #},
        #{
        #    'filename': "ell_cifar10_s200k40_seed1_metrics.json",
        #    'aggregator': "p2p",
        #    'cost': 4000,
        #    'config': (200, 40)
        #},
        #{
        #    'filename': "hsl_cifar10_s200h15_bud_10_3_1_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 315,
        #    'config': (200, 15, 10, 3, 1)
        #}, 
        {
            'filename': "hsl_cifar10_s200h15_bud_10_3_2_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 573,
            'config': (200, 15, 10, 3, 2)
        }, 
        {
            'filename': "hsl_cifar10_s200h15_bud_20_3_3_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 923,
            'config': (200, 15, 20, 3, 3)
        },        
        #{
        #    'filename': "hsl_cifar10_s200h20_bud_20_4_3_seed1_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 1040,
        #    'config': (200, 20, 20, 4, 3)
        #},
        #{
        #    'filename': "hsl_cifar10_s200h20_bud_30_4_3_seed1_metrics.json", #seed1 not a good choice
        #    'aggregator': "hsl",
        #    'cost': 1240,
        #    'config': (200, 20, 30, 4, 3)
        #},
        {
            'filename': "hsl_cifar10_s200h20_bud_40_4_3_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 1440,
            'config': (200, 20, 40, 4, 3)
        },
        {
            'filename': "hsl_cifar10_s200h50_bud_20_5_5_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 2125,
            'config': (200, 50, 20, 5, 5)
        },
        {
            'filename': "hsl_cifar10_s200h50_bud_20_5_7_seed2_metrics.json",
            'aggregator': "hsl",
            'cost': 2525,
            'config': (200, 50, 20, 5, 7)
        },
        {
            'filename': "hsl_cifar10_s200h40_bud_25_5_10_seed1_metrics.json",
            'aggregator': "hsl",
            'cost': 3100,
            'config': (200, 40, 25, 5, 10)
        },
        #{
        #    'filename': "hsl_cifar10_s200h40_bud_25_5_12_seed2_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 3500,
        #    'config': (200, 40, 25, 5, 12)
        #},
        #{
        #    'filename': "hsl_cifar10_s200h50_bud_30_10_16_seed2_metrics.json",
        #    'aggregator': "hsl",
        #    'cost': 3950,
        #    'config': (200, 50, 30, 10, 16)
        #},
        
        
        
    ]

    # --- Experiment 1: Candle plot for final accuracies ---
    plot_candles(experiments_exp1,base_path="./outputs/",title="HSL vs P2P: Final Accuracy Distribution",output_filename="bud-acc-s100.png", half_width=8, xlim=[0, 1100])
    plot_drift(experiments_exp1, base_path="./outputs/",title="Drift Ratio (pre_drift / post_drift) vs. Time", output_filename="cdr-s100.png", yticks=[0.25,0.5,1,2], ylim=[0,2.2])

    # --- Experiment 1b: Candle plot for s=200 ---
    plot_candles(experiments_exp1b,base_path="./outputs/",title="HSL vs P2P: Final Accuracy Distribution",output_filename="bud-acc-s200.png", half_width=40, xlim=[0, 3500])
    plot_drift(experiments_exp1b, base_path="./outputs/",title="Drift Ratio (pre_drift / post_drift) vs. Time", output_filename="cdr-s200.png", yticks=[0.25,0.5,1,2], ylim=[-0.2,2])

    # --- Experiment 2: Drift ratio plot with new modifications ---
    #experiments_exp2 = experiments_exp1  # Reuse the same list, so ensure JSONs have pre_drift/post_drift

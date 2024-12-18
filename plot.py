import json
import numpy as np
import matplotlib.pyplot as plt

# Load the metrics.json file
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Assume metrics['spoke_acc'] is a list of lists, where each entry is an array of accuracies for that evaluation
# Also assume metrics['round'] gives the round numbers at which evaluation was done.
# If 'round' is not provided, we can deduce from eval_time and number of data points.

spoke_acc = metrics.get('spoke_acc', [])
rounds_recorded = metrics.get('round', [])
eval_time = 10  # Update this if needed, or if stored in metrics use that value

if not rounds_recorded:
    # If 'round' is not in metrics, deduce them:
    # If we have N data points in spoke_acc and eval_time, total rounds = N * eval_time
    # The evaluation would have occurred at eval_time, 2*eval_time, ...
    num_points = len(spoke_acc)
    rounds_recorded = [eval_time * (i+1) for i in range(num_points)]
else:
    # If rounds are provided, just use them.
    num_points = len(spoke_acc)

# Prepare arrays to store statistics
xs = []     # x positions (based on actual round number)
avgs = []   # average accuracy per evaluation point

plt.figure(figsize=(10,6))
color = 'C0'  # You can choose any color

for i, acc_values in enumerate(spoke_acc):
    if len(acc_values) == 0:
        continue
    arr = np.array(acc_values)
    minimum = np.min(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    maximum = np.max(arr)
    avg = np.mean(arr)
    
    x = rounds_recorded[i]
    xs.append(x)
    avgs.append(avg)

    # Draw the candle (box from Q1 to Q3)
    # We'll represent the candle as a rectangle centered at x, spanning from q1 to q3
    # and the wick as a vertical line from min to max.
    # Candle width
    candle_width = 0.4
    # Wick
    plt.vlines(x, minimum, maximum, color=color, linewidth=1)
    # Candle body
    plt.fill_between([x - candle_width/2, x + candle_width/2], q1, q3, color=color, alpha=0.3)
    # Average line inside the candle
    plt.hlines(avg, x - candle_width/2, x + candle_width/2, color=color, linewidth=2)

# Overlay the average accuracy as a line plot (using the same color)
plt.plot(xs, avgs, color=color, marker='o', linestyle='-', label='Average Accuracy')

plt.xlabel("Round")
plt.ylabel("Spoke Accuracy (%)")
plt.title("Spoke Accuracy Candle Plot")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

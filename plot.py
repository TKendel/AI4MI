#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
THIS IS IMPORTANT
- SPECIFY THE MODEL WHEN CALLING --model --> because this will appear in the title
- IF ANOTHER METRIC IS ADDED ADJUST THE CODE BECAUSE NOW ONLY DICE AND LOG 
- CHECK THE LIMITS- NOW IT HAS BEEN SET TO 0.4-1 FOR DICE AND 0.08-0 FOR LOSS 
BUT THIS CAN BE DIFFERENT MAKE SURE IT IS NOT CUT OFF
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def run(args: argparse.Namespace) -> None:
    metrics: np.ndarray = np.load(args.metric_file)
    
    if metrics.ndim == 2:
        E, N = metrics.shape
        K = 1
    elif metrics.ndim == 3:
        E, N, K = metrics.shape

    #If we ever get python 3.11, we can change to match and remove the upper two if statements
    """
    match metrics.ndim:
        case 2:
            E, N = metrics.shape
            K = 1
        case 3:
            E, N, K = metrics.shape
    """
    if 'loss_val' in args.metric_file.name:
        title = f"{args.model} Loss - Validation"
        ylabel = "Loss"
        llimit = 0
        ulimit = 0.08
    elif 'dice_val' in args.metric_file.name:
        title = f"{args.model} Dice Score - Validation"
        ylabel = "Dice Score"
        llimit = 0
        ulimit = 1
    elif 'dice3d_val' in args.metric_file.name:
        title = f"{args.model} Volumetric Dice Score - Validation"
        ylabel = "VDice Score"
        llimit = 0
        ulimit = 1
        ylabel = "VDice"
        

    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted size for better readability
    ax.set_title(title, fontsize=16)  # Title dynamically set
    ax.set_ylim([llimit, ulimit])  # Y-axis range from 0 to 1 as requested

    epcs = np.arange(E)

    # Loop to plot individual classes
    for k in range(1, K):
        y = metrics[:, :, k].mean(axis=1)
        ax.plot(epcs, y, label=f"Class {k}", linewidth=2)  # Increase line width

    # Plot for all classes
    if K > 2:
        ax.plot(epcs, metrics.mean(axis=1).mean(axis=1), label="All Classes", linewidth=3, color='purple')
        ax.legend(fontsize=12, loc="lower right")  # Adjust legend font size and position
    else:
        ax.plot(epcs, metrics.mean(axis=1), linewidth=3)

    # Adding labels and grid
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)  # Y-axis label dynamically set
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)  # Add grid lines for clarity

    # Adding tighter layout
    fig.tight_layout()

    # Save the plot if needed
    if args.dest:
        fig.savefig(args.dest, dpi=300, bbox_inches='tight')  # Save with high DPI for clarity

    # Show the plot
    if not args.headless:
        plt.show()
    
    
    # ORIGINAL CODE GIVEN
    """
    fig = plt.figure()
    ax = fig.gca()
    # ax.set_title(str(args.metric_file))
    ax.set_title("ENet Loss Validation")

    epcs = np.arange(E)

    for k in range(1, K):
        y = metrics[:, :, k].mean(axis=1)
        ax.plot(epcs, y, label=f"{k=}", linewidth=1.5)

    if K > 2:
        ax.plot(epcs, metrics.mean(axis=1).mean(axis=1), label="All classes", linewidth=3)
        ax.legend()
    else:
        ax.plot(epcs, metrics.mean(axis=1), linewidth=3)

    fig.tight_layout()
    if args.dest:
        fig.savefig(args.dest)

    if not args.headless:
        plt.show()
    """


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--metric_file', type=Path, required=True, metavar="METRIC_MODE.npy",
                        help="The metric file to plot.")
    parser.add_argument('--dest', type=Path, metavar="METRIC_MODE.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided.")
    parser.add_argument('--model', type=str, required=True, help="The model name to include in the plot title.")
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    run(get_args())

import sys
import os

# Adds the root "optionSimulation" folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from models.black_scholes import simulate_gbm
from gui.plots import plot_paths, plot_heston_paths, plot_sabr_paths
import yfinance as yf
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()
root.title("American Model Simulator")
root.geometry("1200x1200")

#left frame and right frames
left_frame = tk.Frame(root, padx=10)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(root, padx=10)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

canvas = None

def labeled_entry(parent, label, default=""):
    tk.Label(parent, text=label).pack()
    entry = tk.Entry(parent)
    entry.insert(0, default)
    entry.pack()
    return entry

#Inputs
tk.Label(left_frame, text="--- Monte Carlo Simulation ---").pack()


def on_closing():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)


root.mainloop()
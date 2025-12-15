# import tkinter as tk
# from tkinter import ttk, messagebox
# import time
#
# from algorithms.backtracking_warnsdorff import BacktrackingSolver
# from algorithms.cultural_algorithm import CulturalAlgorithmSolver
# from algorithms.pure_backtracking import PureBacktracking
# from algorithms.Pure_Culture import PureCulturalAlgorithm
#
#
# class KnightTourGUI:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Knight's Tour Solver")
#         self.root.geometry("820x720")
#
#         self.board_size = tk.IntVar(value=8)
#         self.current_algorithm = tk.StringVar(value="backtracking_warnsdorff")
#         self.solution = None
#
#         self.setup_gui()
#
#     def setup_gui(self):
#         frame = ttk.Frame(self.root, padding=10)
#         frame.pack(fill="both", expand=True)
#
#         # CONFIG
#         config = ttk.LabelFrame(frame, text="Configuration", padding=10)
#         config.pack(fill="x")
#
#         ttk.Label(config, text="Board Size: ").grid(row=0, column=0)
#         ttk.Spinbox(config, from_=5, to=20, textvariable=self.board_size).grid(row=0, column=1)
#
#         ttk.Label(config, text="Algorithm: ").grid(row=0, column=2)
#
#         ttk.Combobox(
#             config,
#             textvariable=self.current_algorithm,
#             values=[
#                 "backtracking_warnsdorff",
#                 "pure_backtracking",
#                 "cultural",
#                 "Pure Cultural",
#             ],
#             state="readonly"
#         ).grid(row=0, column=3)
#
#         ttk.Button(config, text="Solve", command=self.solve).grid(row=0, column=4)
#         ttk.Button(config, text="Animate", command=self.animate_solution).grid(row=0, column=5)
#         ttk.Button(config, text="Clear", command=self.clear).grid(row=0, column=6)
#         ttk.Button(config, text="Compare", command=self.compare_algorithms).grid(row=0, column=7)
#         ttk.Button(config, text="Compare10", command=self.compare_algorithms_10).grid(row=0, column=8)
#         # RESULTS
#         self.results = tk.Text(frame, height=6)
#         self.results.pack(fill="x", pady=10)
#
#         # CHESSBOARD
#         self.canvas = tk.Canvas(frame, width=600, height=600, bg="white")
#         self.canvas.pack()
#
#         self.draw_board()
#
#     def draw_board(self, solution=None, step=None):
#         self.canvas.delete("all")
#
#         n = self.board_size.get()
#         size = 550 // n
#         offset = 25
#
#         for i in range(n):
#             for j in range(n):
#                 x1 = offset + j * size
#                 y1 = offset + i * size
#                 x2 = x1 + size
#                 y2 = y1 + size
#
#                 color = "white" if (i + j) % 2 == 0 else "gray"
#
#                 # highlight during animation
#                 if solution and step is not None and solution[i][j] == step:
#                     color = "lightgreen"
#
#                 self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
#
#                 if solution and solution[i][j] != -1:
#                     self.canvas.create_text(
#                         x1 + size / 2, y1 + size / 2,
#                         text=str(solution[i][j]),
#                         font=("Arial", size // 4)
#                     )
#
#         # KNIGHT APPEARS ONLY DURING ANIMATION
#         if solution and step is not None:
#             for i in range(n):
#                 for j in range(n):
#                     if solution[i][j] == step:
#                         x = offset + j * size + size // 2
#                         y = offset + i * size + size // 2
#
#                         # Knight (horse)
#                         self.canvas.create_oval(
#                             x - 15, y - 15, x + 15, y + 15,
#                             fill="red", outline="darkred"
#                         )
#                         self.canvas.create_text(
#                             x, y, text="♘", fill="white",
#                             font=("Arial", size // 2)
#                         )
#
#     def solve(self):
#         n = self.board_size.get()
#         algo = self.current_algorithm.get()
#
#         self.results.delete("1.0", "end")
#         self.results.insert("end", f"Solving using {algo}...\n")
#
#         start = time.time()
#
#         if algo == "backtracking_warnsdorff":
#             solver = BacktrackingSolver(n)
#         elif algo == "pure_backtracking":
#             solver = PureBacktracking(n)
#         elif algo == "cultural":
#             solver = CulturalAlgorithmSolver(n)
#         else:
#             solver = PureCulturalAlgorithm(n)
#
#         success, board, attempts = solver.solve(0, 0)
#
#         end = time.time()
#         self.solution = board
#
#         self.results.insert("end", f"Success: {success}\n")
#         self.results.insert("end", f"Time: {end-start:.2f} sec\n")
#         self.results.insert("end", f"Attempts: {attempts}\n")
#
#         self.draw_board(board)
#
#     def animate_solution(self):
#         if not self.solution:
#             messagebox.showwarning("Warning", "You must solve first!")
#             return
#
#         n = self.board_size.get()
#         total_steps = n * n
#
#         def step_animation(step):
#             if step < total_steps:
#                 self.draw_board(self.solution, step)
#                 self.root.after(350, step_animation, step + 1)
#
#         step_animation(0)
#
#     def clear(self):
#         self.solution = None
#         self.results.delete("1.0", "end")
#         self.draw_board()
#
#     def compare_algorithms(self):
#         n = self.board_size.get()
#
#         self.results.delete("1.0", "end")
#         self.results.insert("end", f"Comparing All Algorithms on {n}x{n} board…\n\n")
#
#         algorithms = {
#             "Pure Backtracking": PureBacktracking(n),
#             "Warnsdorff Backtracking": BacktrackingSolver(n),
#             "Cultural Algorithm": CulturalAlgorithmSolver(n),
#             "Pure Cultural Algorithm": PureCulturalAlgorithm(n),
#         }
#
#         results = {}
#
#         for name, solver in algorithms.items():
#             self.results.insert("end", f"Running {name}…\n")
#             self.root.update()
#
#             start = time.time()
#             success, board, attempts = solver.solve(0, 0)
#             end = time.time()
#
#             results[name] = {
#                 "success": success,
#                 "time": end - start,
#                 "attempts": attempts
#             }
#
#         self.results.insert("end", "\n=== FINAL COMPARISON ===\n\n")
#
#         for name, r in results.items():
#             self.results.insert("end", f"{name}:\n")
#             self.results.insert("end", f"   Success:  {r['success']}\n")
#             self.results.insert("end", f"   Time:     {r['time']:.3f} sec\n")
#             self.results.insert("end", f"   Attempts: {r['attempts']}\n\n")
#
#     def compare_algorithms_10(self, runs=10):
#         n = self.board_size.get()
#
#         self.results.delete("1.0", "end")
#         self.results.insert("end", f"Running Comparison ({runs} runs each)…\n\n")
#
#         algorithms = {
#             "Pure Backtracking": PureBacktracking(n),
#             "Warnsdorff Backtracking": BacktrackingSolver(n),
#             "Cultural Algorithm": CulturalAlgorithmSolver(n),
#             "Pure Cultural Algorithm": PureCulturalAlgorithm(n),
#         }
#
#         results = {k: {"time": 0, "success": 0, "attempts": 0} for k in algorithms}
#
#         for name, solver in algorithms.items():
#             self.results.insert("end", f"Testing {name}…\n")
#             self.root.update()
#
#             for i in range(runs):
#                 start = time.time()
#                 success, _, attempts = solver.solve(0, 0)
#                 end = time.time()
#
#                 results[name]["time"] += (end - start)
#                 results[name]["attempts"] += attempts
#                 results[name]["success"] += 1 if success else 0
#
#         self.results.insert("end", "\n=== AVERAGES FOR ALL ALGORITHMS ===\n\n")
#
#         for name, r in results.items():
#             self.results.insert("end", f"{name}:\n")
#             self.results.insert("end", f"   Success Rate: {r['success']}/{runs}\n")
#             self.results.insert("end", f"   Avg Time:     {r['time'] / runs:.3f} sec\n")
#             self.results.insert("end", f"   Avg Attempts: {r['attempts'] / runs:.0f}\n\n")


import tkinter as tk
from tkinter import ttk, messagebox
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.backtracking_warnsdorff import BacktrackingSolver
from algorithms.cultural_algorithm import CulturalAlgorithmSolver
from algorithms.pure_backtracking import PureBacktracking
from algorithms.Pure_Culture import PureCulturalAlgorithm


class KnightTourGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Knight's Tour Solver")
        self.root.geometry("820x720")

        self.board_size = tk.IntVar(value=8)
        self.current_algorithm = tk.StringVar(value="backtracking_warnsdorff")
        self.solution = None

        self.setup_gui()

    def setup_gui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        # CONFIG
        config = ttk.LabelFrame(frame, text="Configuration", padding=10)
        config.pack(fill="x")

        ttk.Label(config, text="Board Size: ").grid(row=0, column=0)
        ttk.Spinbox(config, from_=5, to=20, textvariable=self.board_size).grid(row=0, column=1)

        ttk.Label(config, text="Algorithm: ").grid(row=0, column=2)

        ttk.Combobox(
            config,
            textvariable=self.current_algorithm,
            values=[
                "backtracking_warnsdorff",
                "pure_backtracking",
                "cultural",
                "Pure Cultural",
            ],
            state="readonly"
        ).grid(row=0, column=3)

        ttk.Button(config, text="Solve", command=self.solve).grid(row=0, column=4)
        ttk.Button(config, text="Animate", command=self.animate_solution).grid(row=0, column=5)
        ttk.Button(config, text="Clear", command=self.clear).grid(row=0, column=6)
        ttk.Button(config, text="Compare", command=self.compare_algorithms).grid(row=0, column=7)
        ttk.Button(config, text="Compare10", command=self.compare_algorithms_10).grid(row=0, column=8)
        # RESULTS
        self.results = tk.Text(frame, height=6)
        self.results.pack(fill="x", pady=10)

        # CHESSBOARD
        self.canvas = tk.Canvas(frame, width=600, height=600, bg="white")
        self.canvas.pack()

        self.draw_board()

    def draw_board(self, solution=None, step=None):
        self.canvas.delete("all")

        n = self.board_size.get()
        size = 550 // n
        offset = 25

        for i in range(n):
            for j in range(n):
                x1 = offset + j * size
                y1 = offset + i * size
                x2 = x1 + size
                y2 = y1 + size

                color = "white" if (i + j) % 2 == 0 else "gray"

                # highlight during animation
                if solution and step is not None and solution[i][j] == step:
                    color = "lightgreen"

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

                if solution and solution[i][j] != -1:
                    self.canvas.create_text(
                        x1 + size / 2, y1 + size / 2,
                        text=str(solution[i][j]),
                        font=("Arial", size // 4)
                    )

        # KNIGHT APPEARS ONLY DURING ANIMATION
        if solution and step is not None:
            for i in range(n):
                for j in range(n):
                    if solution[i][j] == step:
                        x = offset + j * size + size // 2
                        y = offset + i * size + size // 2

                        # Knight (horse)
                        self.canvas.create_oval(
                            x - 15, y - 15, x + 15, y + 15,
                            fill="red", outline="darkred"
                        )
                        self.canvas.create_text(
                            x, y, text="♘", fill="white",
                            font=("Arial", size // 2)
                        )

    def solve(self):
        n = self.board_size.get()
        algo = self.current_algorithm.get()

        self.results.delete("1.0", "end")
        self.results.insert("end", f"Solving using {algo}...\n")

        start = time.time()

        if algo == "backtracking_warnsdorff":
            solver = BacktrackingSolver(n)
        elif algo == "pure_backtracking":
            solver = PureBacktracking(n)
        elif algo == "cultural":
            solver = CulturalAlgorithmSolver(n)
        else:
            solver = PureCulturalAlgorithm(n)

        success, board, attempts = solver.solve(0, 0)

        end = time.time()
        self.solution = board

        self.results.insert("end", f"Success: {success}\n")
        self.results.insert("end", f"Time: {end - start:.2f} sec\n")
        self.results.insert("end", f"Attempts: {attempts}\n")

        self.draw_board(board)

    def animate_solution(self):
        if not self.solution:
            messagebox.showwarning("Warning", "You must solve first!")
            return

        n = self.board_size.get()
        total_steps = n * n

        def step_animation(step):
            if step < total_steps:
                self.draw_board(self.solution, step)
                self.root.after(350, step_animation, step + 1)

        step_animation(0)

    def clear(self):
        self.solution = None
        self.results.delete("1.0", "end")
        self.draw_board()

    def compare_algorithms(self):
        n = self.board_size.get()

        self.results.delete("1.0", "end")
        self.results.insert("end", f"Comparing All Algorithms on {n}x{n} board…\n\n")

        algorithms = {
            "Pure Backtracking": PureBacktracking(n),
            "Warnsdorff Backtracking": BacktrackingSolver(n),
            "Cultural Algorithm": CulturalAlgorithmSolver(n),
            "Pure Cultural Algorithm": PureCulturalAlgorithm(n),
        }

        results = {}

        for name, solver in algorithms.items():
            self.results.insert("end", f"Running {name}…\n")
            self.root.update()

            start = time.time()
            success, board, attempts = solver.solve(0, 0)
            end = time.time()

            results[name] = {
                "success": success,
                "time": end - start,
                "attempts": attempts
            }

        self.results.insert("end", "\n=== FINAL COMPARISON ===\n\n")

        for name, r in results.items():
            self.results.insert("end", f"{name}:\n")
            self.results.insert("end", f"   Success:  {r['success']}\n")
            self.results.insert("end", f"   Time:     {r['time']:.3f} sec\n")
            self.results.insert("end", f"   Attempts: {r['attempts']}\n\n")

    def compare_algorithms_10(self, runs=10):
        n = self.board_size.get()

        self.results.delete("1.0", "end")
        self.results.insert("end", f"Running Pure Cultural Algorithm {runs} times…\n\n")

        # Store data for each run
        run_data = []

        for i in range(runs):
            self.results.insert("end", f"Run {i + 1}/{runs}…\n")
            self.root.update()

            solver = PureCulturalAlgorithm(n)
            start = time.time()
            success, _, attempts = solver.solve(0, 0)
            end = time.time()

            time_taken = end - start
            status = "Success" if success else "Fail"

            run_data.append({
                "run": i + 1,
                "status": status,
                "time": time_taken,
                "attempts": attempts
            })

            self.results.insert("end", f"   Status: {status}, Time: {time_taken:.3f}s, Attempts: {attempts}\n")

        # Display summary
        success_count = sum(1 for d in run_data if d["status"] == "Success")
        avg_time = sum(d["time"] for d in run_data) / runs
        avg_attempts = sum(d["attempts"] for d in run_data) / runs

        self.results.insert("end", f"\n=== SUMMARY ===\n")
        self.results.insert("end", f"Success Rate: {success_count}/{runs}\n")
        self.results.insert("end", f"Average Time: {avg_time:.3f} sec\n")
        self.results.insert("end", f"Average Attempts: {avg_attempts:.0f}\n\n")

        # Create plot window
        self.create_plot(run_data)

    def create_plot(self, run_data):
        # Create new window for plots
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Pure Cultural Algorithm - Performance Analysis")
        plot_window.geometry("1200x800")

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Pure Cultural Algorithm - 10 Runs Analysis', fontsize=16, fontweight='bold')

        runs = [d["run"] for d in run_data]
        times = [d["time"] for d in run_data]
        attempts = [d["attempts"] for d in run_data]
        statuses = [d["status"] for d in run_data]

        # Color coding: green for success, red for fail
        colors = ['green' if s == "Success" else 'red' for s in statuses]

        # Plot 1: Time taken per run
        ax1.bar(runs, times, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Run Number', fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontweight='bold')
        ax1.set_title('Execution Time per Run')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks(runs)

        # Plot 2: Attempts per run
        ax2.bar(runs, attempts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Run Number', fontweight='bold')
        ax2.set_ylabel('Number of Attempts', fontweight='bold')
        ax2.set_title('Attempts per Run')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks(runs)

        # Plot 3: Success/Fail pie chart
        success_count = sum(1 for s in statuses if s == "Success")
        fail_count = len(statuses) - success_count
        ax3.pie([success_count, fail_count],
                labels=['Success', 'Fail'],
                colors=['green', 'red'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.1, 0))
        ax3.set_title('Success Rate Distribution')

        # Plot 4: Time vs Attempts scatter plot
        success_data = [(d["time"], d["attempts"]) for d in run_data if d["status"] == "Success"]
        fail_data = [(d["time"], d["attempts"]) for d in run_data if d["status"] == "Fail"]

        if success_data:
            success_times, success_attempts = zip(*success_data)
            ax4.scatter(success_times, success_attempts, color='green', s=100, alpha=0.6, label='Success',
                        edgecolor='black')

        if fail_data:
            fail_times, fail_attempts = zip(*fail_data)
            ax4.scatter(fail_times, fail_attempts, color='red', s=100, alpha=0.6, label='Fail', edgecolor='black')

        ax4.set_xlabel('Time (seconds)', fontweight='bold')
        ax4.set_ylabel('Number of Attempts', fontweight='bold')
        ax4.set_title('Time vs Attempts Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Embed plot in tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add close button
        close_btn = ttk.Button(plot_window, text="Close", command=plot_window.destroy)
        close_btn.pack(pady=10)
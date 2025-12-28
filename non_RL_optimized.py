import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# GA Imports
import pygad

# NSGA-II Imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
import requests
import datetime


# ==========================================
# 1. HELPER CLASSES (Physics Engines)
# ==========================================
class BatteryProblem(ElementwiseProblem):
    """
    Updated Physics Engine: 'Smart Guardrails'
    Instead of penalties, we just block invalid moves physically.
    This prevents the AI from getting scared of full batteries.
    """
    def __init__(self, solar_data, prices):
        self.solar = solar_data
        self.prices = prices
        self.capacity = 13.5
        # We still have 2 objectives: Cost and Wear (Cycles)
        super().__init__(n_var=24, n_obj=2, n_ieq_constr=0, xl=0, xu=2)

    def _evaluate(self, x, out, *args, **kwargs):
        actions = np.round(x).astype(int)
        battery = 0
        cost = 0
        cycles = 0
        # No 'penalty' variable needed anymore!

        for h, action in enumerate(actions):
            # 1. Solar fills battery (Capped at capacity)
            gen = self.solar[h]
            battery = min(battery + gen, self.capacity)
            price = self.prices[h]

            # 2. AI Logic (Safe Mode)
            if action == 1:
                # Check how much empty space is left
                empty_space = self.capacity - battery

                # Only charge what fits. If full, charge 0. DO NOT PENALIZE.
                actual_charge = min(2, empty_space)

                battery += actual_charge
                cost += actual_charge * price

            elif action == 2: # DISCHARGE
                # LOGIC CHANGE: Only discharge what we HAVE. Don't punish empty.
                available_energy = battery
                discharge_amount = min(2, available_energy) # If empty, sells 0.

                battery -= discharge_amount
                cost -= discharge_amount * price
                cycles += discharge_amount

        # Objectives: Minimize Cost, Minimize Cycles
        out["F"] = [cost, cycles/10]
# ==========================================
# 2. THE MASTER PIPELINE CLASS
# ==========================================
class MasterEnergyPipeline:
    def __init__(self):
        # Paths (Update if needed)
        self.data_path = "cleaned_solar_data.csv"
        self.model_path = "solar_brain_model.pkl"
        self.scaler_path = "weather_scaler.pkl"

        # Load Components
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("[Init] Model and Scaler loaded successfully.")
        except:
            print("[Error] Model not found. Please run the training script first.")

        self.prices = [0.10 if h < 7 else (0.50 if 17 <= h < 20 else 0.30) for h in range(24)]

    def calculate_stats(self, schedule, solar_gen): # <--- Added solar_gen
        """Helper to calculate Cost and Wear for any schedule, accounting for Solar."""
        battery = 0
        cost = 0
        cycles = 0

        for h, action in enumerate(schedule):
            # 1. Solar Inflow
            gen = solar_gen[h]
            battery = min(battery + gen, 13.5)

            price = self.prices[h]

            # 2. Apply Action
            if action == 1: # CHARGE
                room = 13.5 - battery
                amount = min(2, room)
                battery += amount
                cost += amount * price
                cycles += amount

            elif action == 2: # DISCHARGE
                amount = min(2, battery)
                battery -= amount
                cost -= amount * price
                cycles += amount

        return cost, cycles

    # ---------------------------------------------------------
    # PHASE 1: MODEL DIAGNOSTICS
    # ---------------------------------------------------------
    def run_model_diagnostics(self):
        print("\n--- PHASE 1: MODEL DIAGNOSTICS ---")
        df = pd.read_csv(self.data_path, parse_dates=['Date-Hour(NMT)'], index_col='Date-Hour(NMT)')
        X = df[['AirTemperature', 'RelativeAirHumidity', 'WindSpeed', 'Hour', 'Month']].values
        y = df['SystemProduction'].values

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = np.maximum(self.model.predict(X_test_scaled), 0)

        r2 = r2_score(y_test, y_pred)
        print(f">> Model Accuracy (R2): {r2:.4f}")

        # Graph 1: Real vs Predicted
        plt.figure(figsize=(14, 5))
        limit = 100
        plt.plot(range(limit), y_test[:limit], label='Actual Solar', color='green', linewidth=2)
        plt.plot(range(limit), y_pred[:limit], label='AI Prediction', color='orange', linestyle='--', linewidth=2)
        plt.title(f"Model Performance: Real vs Predicted (R2: {r2:.2f})")
        plt.xlabel("Test Samples (Hours)"); plt.ylabel("Energy (kWh)")
        plt.legend(); plt.grid(True)
        plt.show() #

        # Graph 2: Scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, alpha=0.3, color='purple')
        plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Perfect Fit')
        plt.title("Prediction Accuracy Scatter Plot")
        plt.xlabel("Actual Energy"); plt.ylabel("Predicted Energy")
        plt.legend(); plt.grid(True)
        plt.show() #

    # ---------------------------------------------------------
    # PHASE 2: SIMULATION SETUP
    # ---------------------------------------------------------
    def get_live_weather_data(self, city_name):
        print(f"\n>> Fetching free data from Open-Meteo for: {city_name}...")

        try:
            # 1. Get Lat/Lon (Geocoding)
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
            geo_res = requests.get(geo_url).json()

            if not geo_res.get('results'):
                print("[Error] City not found.")
                return None

            lat = geo_res['results'][0]['latitude']
            lon = geo_res['results'][0]['longitude']

            weather_url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
                "wind_speed_unit": "ms",
                "forecast_days": 2  # <--- CHANGE: Get 2 days to ensure we find a full day
            }

            w_res = requests.get(weather_url, params=params).json()
            hourly = w_res['hourly']

            # 3. Find the first 00:00 hour (Midnight)
            # Open-Meteo returns time in ISO format "2025-12-27T00:00"
            start_index = 0
            for i, time_str in enumerate(hourly['time']):
                if "T00:00" in time_str:
                    start_index = i
                    break

            print(f">> Aligning simulation to start at midnight (Index {start_index})")

            # 4. Format for your AI Model
            final_weather_profile = []
            current_month = datetime.datetime.now().month

            # Grab 24 hours starting from that midnight index
            for h in range(24):
                idx = start_index + h
                t = hourly['temperature_2m'][idx]
                hum = hourly['relative_humidity_2m'][idx]
                wind = hourly['wind_speed_10m'][idx]

                final_weather_profile.append([t, hum, wind, h, current_month])

            return np.array(final_weather_profile)

        except Exception as e:
            print(f"[Error] Open-Meteo fetch failed: {e}")
            return None

    def setup_simulation_day(self):
      print("\n--- PHASE 2: SIMULATION SETUP ---")

      mode = input("Select Mode: [1] Load from CSV (Random Day)  [2] Open Meteo : ")

      if mode == '2':
          city = input("Enter City Name (e.g., Colombo): ").strip()

          weather_array = self.get_live_weather_data(city)

          if weather_array is not None:
              # Scale & Predict
              scaled_weather = self.scaler.transform(weather_array)
              solar_gen = np.maximum(self.model.predict(scaled_weather), 0)

              print(f">> Forecasted Total Solar: {np.sum(solar_gen):.2f} kWh")

              # Visual Check
              plt.figure(figsize=(10,3))
              plt.plot(solar_gen, color='green', label=f'Solar Forecast: {city}')
              plt.title(f"24h Solar Prediction for {city}")
              plt.xlabel("Hour"); plt.ylabel("Energy (kWh)")
              plt.grid(True, alpha=0.3)
              plt.legend()
              plt.show() #

              return solar_gen
          else:
              print("Falling back to CSV...")
      else:
        df = pd.read_csv(self.data_path)
        random_idx = 1026
        weather = df.iloc[random_idx : random_idx+24][['AirTemperature', 'RelativeAirHumidity', 'WindSpeed', 'Hour', 'Month']]
        scaled_weather = self.scaler.transform(weather.values)
        solar_gen = np.maximum(self.model.predict(scaled_weather), 0)
        print(f">> Selected Day Index: {random_idx}")
        print(f">> Forecasted Total Solar: {np.sum(solar_gen):.2f} kWh")
        return solar_gen

    # ---------------------------------------------------------
    # FIXED: EXPLICIT REPORT (NOW SEES THE SUN!)
    # ---------------------------------------------------------
    def print_explicit_schedule(self, schedule, solar_gen, strategy_name):
        print(f"\n{'='*60}")
        print(f" FINAL STRATEGY REPORT: {strategy_name.upper()}")
        print(f"{'='*60}")
        print(f"{'Hour':<6} | {'Price':<8} | {'Solar':<8} | {'Action':<12} | {'Battery':<8} | {'Financial Impact':<15}")
        print("-" * 75)

        battery = 0
        total_cost = 0

        actions_map = {0: "IDLE", 1: "CHARGE (+)", 2: "DISCHARGE (-)"}

        for h, act in enumerate(schedule):
            price = self.prices[h]
            gen = solar_gen[h]
            action_str = actions_map[act]
            cost_change = 0.0

            # 1. CRITICAL FIX: Add Solar First!
            # (Just like the physics engine does)
            battery = min(battery + gen, 13.5)

            # 2. Then Apply Action
            if act == 1: # CHARGE
                # Check space (Smart Guardrail Logic)
                room = 13.5 - battery
                amount = min(2, room)
                battery += amount
                cost_change = amount * price

            elif act == 2: # DISCHARGE
                # Check available energy
                amount = min(2, battery)
                battery -= amount
                cost_change = -(amount * price)

            total_cost += cost_change

            # Formatting
            impact_str = f"£{abs(cost_change):.2f} " + ("(Paid)" if cost_change > 0 else "(Earned)" if cost_change < 0 else "-")
            print(f"{h:02d}:00  | £{price:.2f}   | {gen:>4.1f} kWh | {action_str:<12} | {battery:>4.1f} kWh | {impact_str}")

        print("-" * 75)
        # Note: We display -total_cost because Negative Cost = Profit
        print(f"FINAL RESULT >> Total Profit: £{-total_cost:.2f}")
        print(f"{'='*60}\n")
    # ---------------------------------------------------------
    # PHASE 3: SINGLE OBJECTIVE (GA)
    # ---------------------------------------------------------
    def run_single_objective(self, solar_gen):
        print("\n--- PHASE 3: SINGLE OBJECTIVE OPTIMIZATION ---")

         # Callback to capture Population Diversity (Std Dev of Fitness)
        def on_gen(ga_instance):
            # Diversity: Standard Deviation of the population's fitness
            fitness_std = np.std(ga_instance.last_generation_fitness)
            self.ga_diversity_history.append(fitness_std)
            self.ga_fitness_history.append(ga_instance.best_solution()[1])

        def fitness_func(ga, solution, idx):
            batt=0; cost=0; pen=0
            for h, act in enumerate(solution):
                if act==1:
                    if batt+2<=13.5: batt+=2; cost+=2*self.prices[h]
                    else: pen+=50
                elif act==2:
                    if batt-2>=0: batt-=2; cost-=2*self.prices[h]
                    else: pen+=50
            return 1.0/(cost+pen+1000)

        self.ga_diversity_history = [] # Reset
        self.ga_fitness_history = []

        ga = pygad.GA(num_generations=50, num_parents_mating=5, fitness_func=fitness_func,
                      sol_per_pop=50, num_genes=24, gene_type=int, gene_space=[0,1,2],
                      keep_elitism=2, on_generation=on_gen) # Added Callback
        ga.run()
        best_sol, _, _ = ga.best_solution()

        cost, wear = self.calculate_stats(best_sol, solar_gen)
        self.visualize_operational_report(best_sol, solar_gen, f"Single-Objective (Profit Focused)\nCost: £{cost:.2f} | Wear: {wear:.1f} kWh")
        self.print_explicit_schedule(best_sol, solar_gen,"Single-Objective (Profit Focused)")
        return best_sol

    # ---------------------------------------------------------
    # PHASE 4: MULTI OBJECTIVE (STANDARD - NO RL)
    # ---------------------------------------------------------
    def run_multi_objective(self, solar_gen):
        print("\n--- PHASE 4: MULTI OBJECTIVE OPTIMIZATION (STANDARD) ---")

        # 1. Setup Environment
        problem = BatteryProblem(solar_gen, self.prices)

        # Standard NSGA-II (Static Mutation)
        algorithm = NSGA2(pop_size=100, n_offsprings=50, sampling=IntegerRandomSampling(),
                          crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
                          mutation=PM(prob=0.05, eta=20, repair=RoundingRepair()), # Static Mutation
                          eliminate_duplicates=True)

        # 2. Run Optimization
        # We use 'minimize' here because we don't need the manual RL loop
        res = minimize(problem, algorithm, ('n_gen', 100), seed=1, verbose=False, save_history=True)
        self.nsga_history = res.history # Store for advanced plotting

        # 3. Graph: Pareto Front (Trade-off Visualizer)
        costs = res.F[:, 0]; degradation = res.F[:, 1]
        plt.figure(figsize=(8, 5))
        plt.scatter(costs, degradation, c='red', s=50, label='Solutions')
        plt.title("Pareto Front: Cost vs Battery Health")
        plt.xlabel("Cost (£) [Negative = Profit]"); plt.ylabel("Degradation (Cycles)")
        plt.grid(True); plt.legend()
        plt.show() #

        # 4. Extract Best Solution (SMART SELECTION LOGIC)
        # This now matches your RL version's logic for fair comparison
        all_costs = res.F[:, 0]
        all_wear = res.F[:, 1]

        best_idx = -1
        best_cost = 99999

        # Smart Filter: Real-World Grid Logic
        wear_limit = 40.0 # Aggressive grid limit

        valid_indices = [
            i for i, w in enumerate(all_wear)
            if w <= wear_limit and all_costs[i] < -0.01
        ]

        if len(valid_indices) > 0:
            # Pick the richest survivor
            for idx in valid_indices:
                if all_costs[idx] < best_cost:
                    best_cost = all_costs[idx]
                    best_idx = idx
            print(f">> Success: Found PROFITABLE strategy! Profit: £{-best_cost:.2f} | Wear: {all_wear[best_idx]:.1f} kWh")
        else:
            # Fallback
            print(">> Warning: No profitable strategies found. Picking least worst option.")
            best_idx = np.argmin(all_costs)

        best_sol = np.round(res.X[best_idx]).astype(int)

        # 5. Final Reporting (Solar-Aware)
        # We pass 'solar_gen' to ensure calculations are correct
        cost, wear = self.calculate_stats(best_sol, solar_gen)

        self.visualize_operational_report(best_sol, solar_gen, f"Multi-Objective (Standard)\nCost: £{cost:.2f} | Wear: {wear:.1f} kWh")
        self.print_explicit_schedule(best_sol, solar_gen,"Multi-Objective (Smart Balanced)")
        return best_sol

    # ---------------------------------------------------------
    # VISUALIZATION ENGINE
    # ---------------------------------------------------------
    def visualize_operational_report(self, schedule, solar_gen, title):
        battery_level = [0]; curr = 0
        for act in schedule:
            if act==1: curr = min(curr+2, 13.5)
            elif act==2: curr = max(curr-2, 0)
            battery_level.append(curr)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.1)

        # Top Panel
        ax1.fill_between(range(25), battery_level, color='#1f77b4', alpha=0.4)
        ax1.plot(range(25), battery_level, color='#1f77b4', linewidth=2)
        ax1.set_ylabel('Battery (kWh)', color='blue')
        ax1.set_ylim(0, 15); ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Operational Report: {title}", fontweight='bold')

        ax1_twin = ax1.twinx()
        ax1_twin.step(range(24), self.prices, where='mid', color='red', linestyle='--')
        ax1_twin.set_ylabel('Price', color='red'); ax1_twin.set_ylim(0, 0.6)

        ax1_tri = ax1.twinx()
        ax1_tri.spines["right"].set_position(("axes", 1.1))
        ax1_tri.plot(range(24), solar_gen, color='orange', label='Solar')
        ax1_tri.set_ylabel('Solar', color='orange')

        # Bottom Panel (Traffic Light)
        cmap = mcolors.ListedColormap(['lightgrey', '#2ca02c', '#d62728'])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        ax2.imshow(np.array([schedule]), aspect='auto', cmap=cmap, norm=norm)
        ax2.set_yticks([]); ax2.set_xlabel("Hour of Day")

        legend_elements = [Patch(facecolor='lightgrey', label='Idle'), Patch(facecolor='#2ca02c', label='Charge'), Patch(facecolor='#d62728', label='Discharge')]
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.3))
        plt.show()


    def compare_strategies(self, sched_A, sched_B, solar_gen): # <--- 1. Added solar_gen here
        print("\n--- PHASE 5: HEAD-TO-HEAD COMPARISON ---")

        # 2. Pass solar_gen to the calculator so it knows about free energy
        cost_A, wear_A = self.calculate_stats(sched_A, solar_gen)
        cost_B, wear_B = self.calculate_stats(sched_B, solar_gen)

        # 1. VISUAL COMPARISON STRIPS
        cmap = mcolors.ListedColormap(['lightgrey', '#2ca02c', '#d62728'])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

        ax1.set_title(f"Single Objective (Greedy)\nProfit: £{-cost_A:.2f} | Wear: {wear_A} kWh (High)", fontweight='bold', color='darkred')
        ax1.imshow(np.array([sched_A]), aspect='auto', cmap=cmap, norm=norm)
        ax1.set_yticks([])

        ax2.set_title(f"Multi Objective (Balanced)\nProfit: £{-cost_B:.2f} | Wear: {wear_B} kWh (Low)", fontweight='bold', color='darkgreen')
        ax2.imshow(np.array([sched_B]), aspect='auto', cmap=cmap, norm=norm)
        ax2.set_yticks([]); ax2.set_xlabel("Hour of Day")
        plt.show()

        # 2. BAR CHART COMPARISON
        labels = ['Single Obj (Greedy)', 'Multi Obj (Balanced)']
        costs = [-cost_A, -cost_B] # Profit (Negative cost)
        wears = [wear_A, wear_B]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, costs, width, label='Profit (£)', color='green')

        # Use a twin axis for Wear because the scale is different
        ax2 = ax.twinx()
        rects2 = ax2.bar(x + width/2, wears, width, label='Battery Wear (kWh)', color='grey')

        ax.set_ylabel('Profit (£)', color='green', fontweight='bold')
        ax2.set_ylabel('Battery Wear (kWh)', color='grey', fontweight='bold')
        ax.set_title('Final Verdict: Financial Gain vs. Hardware Damage')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        # Add values on top
        ax.bar_label(rects1, padding=3, fmt='£%.2f')
        ax2.bar_label(rects2, padding=3, fmt='%.1f kWh')

        plt.show()

        # ---------------------------------------------------------
    # PHASE 6: ADVANCED ANALYSIS PLOTS (NEW!)
    # ---------------------------------------------------------
    def plot_advanced_analysis(self):
        print("\n--- PHASE 6: GENERATING SCIENTIFIC ANALYSIS PLOTS ---")

        # --- PLOT 1: CONVERGENCE (GA vs NSGA-II) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # PyGAD Convergence (Fitness over time)
        ax1.plot(self.ga_fitness_history, color='blue', linewidth=2)
        ax1.set_title("Single-Objective Convergence (Fitness)")
        ax1.set_xlabel("Generation"); ax1.set_ylabel("Best Fitness (1/Cost)")
        ax1.grid(True)

        # NSGA-II Convergence (Hypervolume or Min Cost)
        # We track the minimum cost found in the population at each generation
        nsga_min_costs = [np.min(algo.pop.get("F")[:, 0]) for algo in self.nsga_history]
        ax2.plot(nsga_min_costs, color='red', linewidth=2)
        ax2.set_title("Multi-Objective Convergence (Min Cost)")
        ax2.set_xlabel("Generation"); ax2.set_ylabel("Minimum Cost Found")
        ax2.grid(True)
        plt.show() #

        # --- PLOT 2: POPULATION DIVERSITY (Exploration vs Exploitation) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # PyGAD Diversity (Std Dev of Fitness)
        ax1.plot(self.ga_diversity_history, color='green', linestyle='--')
        ax1.set_title("Single-Obj Diversity (Fitness Std Dev)")
        ax1.set_xlabel("Generation"); ax1.set_ylabel("Diversity Metric")
        ax1.fill_between(range(len(self.ga_diversity_history)), self.ga_diversity_history, color='green', alpha=0.1)
        ax1.grid(True)

        # NSGA-II Diversity (Spread of Objective Space)
        # Calculate standard deviation of Costs in the population per generation
        nsga_diversity = [np.std(algo.pop.get("F")[:, 0]) for algo in self.nsga_history]
        ax2.plot(nsga_diversity, color='purple', linestyle='--')
        ax2.set_title("Multi-Obj Diversity (Cost Spread)")
        ax2.set_xlabel("Generation"); ax2.set_ylabel("Diversity Metric")
        ax2.fill_between(range(len(nsga_diversity)), nsga_diversity, color='purple', alpha=0.1)
        ax2.grid(True)
        plt.show() #

        # --- PLOT 3: EVOLUTION OF SOLUTIONS (Objective Space) ---
        # Only relevant for Multi-Objective to show the Pareto Front moving
        print("Generating Evolution Animation Frame...")

        generations_to_plot = [0, 9, 49, 99] # Plot Start, Early, Mid, End
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

        for i, gen_idx in enumerate(generations_to_plot):
            if gen_idx < len(self.nsga_history):
                pop_F = self.nsga_history[gen_idx].pop.get("F")
                ax = axes[i]
                ax.scatter(pop_F[:, 0], pop_F[:, 1], color='red', alpha=0.6)
                ax.set_title(f"Generation {gen_idx+1}")
                ax.set_xlabel("Cost (£)");
                if i == 0: ax.set_ylabel("Battery Wear (Cycles)")
                ax.grid(True)

        plt.suptitle("Evolution of Pareto Front (Movement toward Optimality)", fontsize=14)
        plt.show()
    def run_pipeline(self):
        self.run_model_diagnostics()
        solar = self.setup_simulation_day()
        sched_single = self.run_single_objective(solar)
        sched_multi = self.run_multi_objective(solar)
        self.compare_strategies(sched_single, sched_multi, solar)
        self.plot_advanced_analysis()


# ==========================================
# MAIN RUN
# ==========================================
if __name__ == "__main__":
    pipeline = MasterEnergyPipeline()
    pipeline.run_pipeline()
import random
import numpy as np
from collections import defaultdict, deque
import time
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Literal
from multiprocessing import Pool, cpu_count
import heapq

class MaxFlowTabuSearch:
    def __init__(self, graph: Dict[Tuple[int, int], float], source: int, sink: int,
                 tabu_tenure=20, max_iterations=20000,
                 initialization: Literal['random', 'ek_partial', 'greedy'] = 'ek_partial'):
        self.source = source
        self.sink = sink
        self.graph = graph
        self.edges = list(graph.keys())
        self.capacities = np.array([graph[e] for e in self.edges], dtype=np.float32)
        self.edge_to_idx = {(u, v): i for i, (u, v) in enumerate(self.edges)}
        self.source_edges = [i for i, (u, v) in enumerate(self.edges) if u == self.source]
        self.n_edges = len(self.edges)
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.initialization = initialization
        self.current_flow = np.zeros(len(self.edges), dtype=np.float32)
        self.best_flow = np.zeros(len(self.edges), dtype=np.float32)
        self.best_value = 0.0
        self.convergence_history = []
        self.convergence_iterations = []
        self.evaluations = 0
        self.best_iteration = 0
        self.start_time = time.time()
        self._precompute_adjacency()
        self.optimal_max_flow_value = self._run_edmonds_karp()
        self.source_out_capacity = sum(cap for (u, v), cap in self.graph.items() if u == self.source)
        self.sink_in_capacity = sum(cap for (u, v), cap in self.graph.items() if v == self.sink)
        self._initialize_flow()
        self.tabu_dict = {}
        self.move_frequency = np.zeros(self.n_edges, dtype=int)
        self.elite_solutions = []
        self.elite_size = 5

    def _precompute_adjacency(self):
        self.adj = defaultdict(list)
        for (u, v) in self.graph:
            self.adj[u].append(v)

    def _run_edmonds_karp(self) -> float:
        residual = defaultdict(float)
        for (u, v), cap in self.graph.items():
            residual[(u, v)] = cap
            residual[(v, u)] = 0.0
        flow = 0.0
        while True:
            parent = {}
            visited = {self.source}
            queue = deque([self.source])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in self.adj[u]:
                    if residual[(u, v)] > 1e-6 and v not in visited:
                        visited.add(v)
                        parent[v] = u
                        if v == self.sink:
                            found = True
                            break
                        queue.append(v)
            if not found:
                break
            path_flow = float('inf')
            v = self.sink
            while v != self.source:
                u = parent[v]
                path_flow = min(path_flow, residual[(u, v)])
                v = u
            v = self.sink
            while v != self.source:
                u = parent[v]
                residual[(u, v)] -= path_flow
                residual[(v, u)] += path_flow
                v = u
            flow += path_flow
        return flow

    def _initialize_flow(self):
        if self.initialization == 'random':
            self._random_initialization()
        elif self.initialization == 'ek_partial':
            self._ek_partial_initialization()
        elif self.initialization == 'greedy':
            self._greedy_initialization()
        self.best_value = self._calculate_flow_value(self.current_flow)
        self.best_flow = self.current_flow.copy()
        self.convergence_history = [self.best_value]
        self.convergence_iterations = [0]

    def _random_initialization(self):
        self.current_flow = np.array([random.uniform(0, cap) for cap in self.capacities], dtype=np.float32)

    def _ek_partial_initialization(self):
        optimal_flow = np.zeros(len(self.edges), dtype=np.float32)
        residual = defaultdict(float)
        for idx, (u, v) in enumerate(self.edges):
            residual[(u, v)] = self.capacities[idx]
            residual[(v, u)] = 0.0
        while True:
            parent = {}
            visited = {self.source}
            queue = deque([self.source])
            found = False
            while queue and not found:
                u = queue.popleft()
                for v in self.adj[u]:
                    if residual[(u, v)] > 1e-6 and v not in visited:
                        visited.add(v)
                        parent[v] = u
                        if v == self.sink:
                            found = True
                            break
                        queue.append(v)
            if not found:
                break
            path_flow = float('inf')
            v = self.sink
            while v != self.source:
                u = parent[v]
                path_flow = min(path_flow, residual[(u, v)])
                v = u
            v = self.sink
            while v != self.source:
                u = parent[v]
                residual[(u, v)] -= path_flow
                residual[(v, u)] += path_flow
                optimal_flow[self.edge_to_idx[(u, v)]] += path_flow
                v = u
        factor = random.uniform(0.7, 0.95)
        self.current_flow = optimal_flow * factor

    def _greedy_initialization(self):
        self.current_flow = np.zeros(len(self.edges), dtype=np.float32)
        residual = self.capacities.copy()
        for _ in range(10):
            path = self._find_augmenting_path_greedy(residual)
            if not path:
                break
            min_residual = min(residual[edge_idx] for edge_idx in path)
            for edge_idx in path:
                self.current_flow[edge_idx] += min_residual
                residual[edge_idx] -= min_residual

    def _find_augmenting_path_greedy(self, residual):
        parent = {}
        capacity = {}
        visited = set()
        queue = [self.source]
        visited.add(self.source)
        capacity[self.source] = float('inf')
        while queue:
            u = max(queue, key=lambda x: capacity[x])
            queue.remove(u)
            for v in self.adj[u]:
                edge_idx = self.edge_to_idx[(u, v)]
                if residual[edge_idx] > 1e-6 and v not in visited:
                    visited.add(v)
                    parent[v] = u
                    capacity[v] = min(capacity[u], residual[edge_idx])
                    if v == self.sink:
                        path = []
                        node = self.sink
                        while node != self.source:
                            edge_idx = self.edge_to_idx[(parent[node], node)]
                            path.append(edge_idx)
                            node = parent[node]
                        return path[::-1]
                    queue.append(v)
        return None

    def _calculate_flow_value(self, flow):
        self.evaluations += 1
        return np.sum(flow[self.source_edges])

    def _find_augmenting_path(self, flow):
        parent = {}
        visited = set()
        queue = deque([self.source])
        visited.add(self.source)
        found = False
        while queue and not found:
            u = queue.popleft()
            for v in self.adj[u]:
                residual = self.graph[(u, v)] - flow[self.edge_to_idx[(u, v)]]
                if residual > 1e-6 and v not in visited:
                    visited.add(v)
                    parent[v] = u
                    if v == self.sink:
                        found = True
                        break
                    queue.append(v)
        if not found:
            return None
        path = []
        v = self.sink
        while v != self.source:
            u = parent[v]
            path.append((self.edge_to_idx[(u, v)], True))
            v = u
        return path[::-1]

    def _generate_edge_exchange_moves(self, current_flow, n_exchanges=5):
        moves = []
        saturated = [i for i in range(self.n_edges)
                    if current_flow[i] >= self.capacities[i] - 1e-6]
        unsaturated = [i for i in range(self.n_edges)
                      if current_flow[i] < self.capacities[i] - 1e-6]
        for _ in range(min(n_exchanges, len(saturated), len(unsaturated))):
            src, dest = random.choice(saturated), random.choice(unsaturated)
            delta = min(current_flow[src], self.capacities[dest] - current_flow[dest])
            if delta > 1e-6:
                new_flow = current_flow.copy()
                new_flow[src] -= delta
                new_flow[dest] += delta
                moves.append((new_flow, (src, dest), delta, False))
        return moves

    def _get_neighborhood_moves(self, current_flow, iteration):
        moves = []
        bonus_scores = np.zeros(self.n_edges)
        if np.sum(self.move_frequency) > 0:
            mean_freq = np.mean(self.move_frequency)
            if mean_freq > 0:
                freq_bonus = np.maximum(self.move_frequency - mean_freq, 0) / (mean_freq + 1e-6)
                bonus_scores = 0.05 * freq_bonus
        sample_size = min(30, self.n_edges // 80)
        for edge_idx in random.sample(range(self.n_edges), sample_size):
            curr = current_flow[edge_idx]
            cap = self.capacities[edge_idx]
            if curr < cap - 1e-6:
                step = min(cap - curr, cap * 0.2)
                new_flow = current_flow.copy()
                new_flow[edge_idx] += step
                is_tabu = edge_idx in self.tabu_dict and self.tabu_dict[edge_idx] > iteration
                moves.append((new_flow, edge_idx, step, is_tabu, bonus_scores[edge_idx]))
            if curr > 1e-6:
                step = min(curr, cap * 0.2)
                new_flow = current_flow.copy()
                new_flow[edge_idx] -= step
                is_tabu = edge_idx in self.tabu_dict and self.tabu_dict[edge_idx] > iteration
                moves.append((new_flow, edge_idx, -step, is_tabu, bonus_scores[edge_idx]))
        if iteration % 20 == 0:
            exchange_moves = self._generate_edge_exchange_moves(current_flow)
            for move in exchange_moves:
                flow, idx_tuple, delta, is_tabu = move
                moves.append((flow, idx_tuple, delta, is_tabu, 0.0))
        if iteration % 100 == 0:
            path = self._find_augmenting_path(current_flow)
            if path:
                min_residual = min(
                    self.capacities[edge_idx] - current_flow[edge_idx] if is_forward
                    else current_flow[edge_idx]
                    for edge_idx, is_forward in path
                )
                if min_residual > 1e-6:
                    new_flow = current_flow.copy()
                    for edge_idx, is_forward in path:
                        if is_forward:
                            new_flow[edge_idx] += min_residual
                        else:
                            new_flow[edge_idx] -= min_residual
                    path_bonus = np.sum([bonus_scores[e] for e, _ in path])
                    moves.append((new_flow, tuple(edge_idx for edge_idx, _ in path), min_residual, False, path_bonus))
        return moves

    def _diversify_solution(self, current_flow):
        reset_factor = random.uniform(0.6, 0.8)
        noise_factor = random.uniform(0.1, 0.2)
        new_flow = current_flow * (1 - reset_factor)
        num_perturb = min(len(self.edges) // 20, 5)
        indices = random.sample(range(len(self.edges)), num_perturb)
        for idx in indices:
            cap = self.capacities[idx]
            noise = random.uniform(-noise_factor, noise_factor) * cap
            new_flow[idx] = max(0, min(cap, new_flow[idx] + noise))
        return new_flow

    def _adaptive_parameters(self, iteration, stagnation_counter):
        if stagnation_counter > 100:
            self.tabu_tenure = min(int(self.tabu_tenure * 1.2), 150)
        elif stagnation_counter < 50:
            self.tabu_tenure = max(int(self.tabu_tenure * 0.9), 10)
        if iteration % 2000 == 0 and stagnation_counter > 300:
            self.current_flow = self._diversify_solution(self.current_flow)
            return True
        return False

    def plot_convergence(self, filename=None, title_suffix=""):
        plt.figure(figsize=(12, 7))
        plt.plot(self.convergence_iterations, self.convergence_history, 'b-', label='Best Flow Value', linewidth=2)
        plt.axhline(y=self.optimal_max_flow_value, color='r', linestyle='--',
                   linewidth=1.5, label=f'Optimal (EK): {self.optimal_max_flow_value:.2f}')
        plt.scatter([self.best_iteration], [self.best_value], color='g',
                   s=100, zorder=5, label=f'Best: {self.best_value:.2f} at {self.best_iteration}')
        plt.title(f'Tabu Search Convergence {title_suffix}\n'
                 f'Nodes: {len(set(n for n, _ in self.graph))}, Edges: {len(self.graph)}', pad=20)
        plt.xlabel('Iterations', labelpad=10)
        plt.ylabel('Flow Value', labelpad=10)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        info_text = (f"Optimal: {self.optimal_max_flow_value:.2f}\n"
                     f"Evaluations: {self.evaluations}\n"
                     f"Time: {time.time() - self.start_time:.2f}s")
        plt.gcf().text(0.15, 0.7, info_text, bbox=dict(facecolor='white', alpha=0.8))
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def search(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.tabu_dict = {}
        stagnation_counter = 0
        upper_bound = min(self.source_out_capacity, self.sink_in_capacity)
        long_term_penalties = np.zeros(self.n_edges)
        if self.elite_solutions:
            elite_flows = [flow for _, flow in self.elite_solutions]
            if elite_flows:
                elite_flows_array = np.array(elite_flows)
                is_saturated_in_elite = elite_flows_array >= (self.capacities - 1e-4)
                saturation_frequency = np.mean(is_saturated_in_elite, axis=0)
                threshold = 0.8
                mask = saturation_frequency > threshold
                long_term_penalties[mask] = 5
        for iteration in range(self.max_iterations):
            diversified = self._adaptive_parameters(iteration, stagnation_counter)
            if diversified:
                stagnation_counter = 0
            moves = self._get_neighborhood_moves(self.current_flow, iteration)
            if not moves:
                stagnation_counter += 1
                continue
            best_move = None
            best_val = -np.inf
            base_val_of_best_move = -np.inf
            for move_flow, idx, delta, is_tabu, bonus_score in moves:
                base_val = self._calculate_flow_value(move_flow)
                val_with_bonus = base_val + bonus_score
                is_admissible = (not is_tabu) or (base_val > self.best_value)
                if is_admissible and val_with_bonus > best_val:
                    best_val = val_with_bonus
                    base_val_of_best_move = base_val
                    best_move = (move_flow, idx)
            if best_move is None:
                stagnation_counter += 1
                continue
            new_flow, edge_idx = best_move
            self.current_flow = new_flow
            base_tenure = self.tabu_tenure
            if isinstance(edge_idx, tuple):
                for e in edge_idx:
                    final_tenure = base_tenure + long_term_penalties[e]
                    self.tabu_dict[e] = iteration + int(final_tenure)
                    self.move_frequency[e] += 1
            else:
                final_tenure = base_tenure + long_term_penalties[edge_idx]
                self.tabu_dict[edge_idx] = iteration + int(final_tenure)
                self.move_frequency[edge_idx] += 1
            if iteration % 500 == 0:
                self.tabu_dict = {k: v for k, v in self.tabu_dict.items() if v > iteration}
            current_val = self._calculate_flow_value(self.current_flow)
            if current_val > self.best_value:
                self.best_value = current_val
                self.best_flow = self.current_flow.copy()
                self.best_iteration = iteration
                stagnation_counter = 0
                heapq.heappush(self.elite_solutions, (-current_val, self.current_flow.copy()))
                if len(self.elite_solutions) > self.elite_size:
                    heapq.heappop(self.elite_solutions)
            else:
                stagnation_counter += 1
            if iteration % 100 == 0 or current_val > self.convergence_history[-1]:
                self.convergence_history.append(self.best_value)
                self.convergence_iterations.append(iteration)
            if abs(current_val - upper_bound) < 1e-6:
                break
        elapsed_time = time.time() - self.start_time
        return {
            'best_value': self.best_value,
            'best_flow': self.best_flow.tolist(),
            'best_iteration': self.best_iteration,
            'evaluations': self.evaluations,
            'elapsed_time': elapsed_time,
            'convergence_history': self.convergence_history,
            'convergence_iterations': self.convergence_iterations,
            'optimal_flow': self.optimal_max_flow_value
        }

def read_instance(filename: str) -> Tuple[Dict[Tuple[int, int], float], int, int, int, int]:
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    n_nodes = int(lines[0])
    n_edges = int(lines[1])
    source = int(lines[2])
    sink = int(lines[3])
    graph = {}
    for i in range(4, min(4 + n_edges, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 3:
            u, v, capacity = int(parts[0]), int(parts[1]), float(parts[2])
            graph[(u, v)] = capacity
    return graph, source, sink, n_nodes, n_edges

def run_single_experiment(args):
    graph, source, sink, seed, output_dir, filename_base, run = args
    solver = MaxFlowTabuSearch(graph, source, sink)
    result = solver.search(seed=seed)
    plot_path = os.path.join(output_dir, f"{filename_base}_run{run+1}_convergence.png")
    solver.plot_convergence(plot_path, title_suffix=f"(Run {run+1})")
    print(f"Run {run + 1}: Best={result['best_value']:.2f}, Iter={result['best_iteration']}, "
          f"Eval={result['evaluations']}, Time={result['elapsed_time']:.4f}s")
    return result

def compute_mean_convergence(runs_results, max_iterations):
    common_iterations = np.linspace(0, max_iterations, 1000)
    all_flows = []
    for run in runs_results:
        interp_flow = np.interp(
            common_iterations,
            run['convergence_iterations'],
            run['convergence_history'],
            right=run['convergence_history'][-1]
        )
        all_flows.append(interp_flow)
    mean_flow = np.mean(all_flows, axis=0)
    std_flow = np.std(all_flows, axis=0)
    return common_iterations, mean_flow, std_flow

def get_best_run(runs_results):
    scores = [r['best_value'] - (0.0001 * r['best_iteration']) for r in runs_results]
    best_idx = np.argmax(scores)
    return runs_results[best_idx]

def plot_average_convergence(results, filename, optimal_value):
    plt.figure(figsize=(12, 7))
    max_iter = max([r['convergence_iterations'][-1] for r in results])
    x_vals, mean_conv, std_conv = compute_mean_convergence(results, max_iter)
    plt.plot(x_vals, mean_conv, 'b-', linewidth=2, label='Average Flow')
    plt.fill_between(x_vals, mean_conv - std_conv, mean_conv + std_conv, alpha=0.2)
    plt.axhline(y=optimal_value, color='r', linestyle='--', linewidth=1.5, label=f'Optimal (EK): {optimal_value:.2f}')
    plt.title(f'Average Convergence for {os.path.basename(filename)}\n{len(results)} runs', pad=20)
    plt.xlabel('Iterations', labelpad=10)
    plt.ylabel('Flow Value', labelpad=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def run_experiments(output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    instance_files = [f for f in os.listdir() if f.startswith("network_") and f.endswith(".txt")]
    results = {}
    for filename in sorted(instance_files):
        print(f"\n--- Processing instance: {filename} ---")
        try:
            graph, source, sink, n_nodes, n_edges = read_instance(filename)
            args_list = [(graph, source, sink, run + 42, output_dir,
                         os.path.splitext(os.path.basename(filename))[0], run) for run in range(10)]
            with Pool(max(1, cpu_count() // 2)) as p:
                runs_results = p.map(run_single_experiment, args_list)
            max_iter = max([r['convergence_iterations'][-1] for r in runs_results])
            x_vals, mean_conv, std_conv = compute_mean_convergence(runs_results, max_iter)
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, mean_conv, label='Mean Flow', color='blue')
            plt.axhline(y=runs_results[0]['optimal_flow'], color='red',
                       linestyle='--', label='Optimal (Edmonds-Karp)')
            plt.title(f'Mean Convergence for {filename}')
            plt.xlabel('Iterations')
            plt.ylabel('Flow Value')
            plt.grid(True)
            plt.legend()
            mean_plot_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mean_convergence.png")
            plt.savefig(mean_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            best_run = get_best_run(runs_results)
            plt.figure(figsize=(10, 6))
            plt.plot(best_run['convergence_iterations'], best_run['convergence_history'],
                    label=f'Best Run (Flow={best_run["best_value"]:.2f})', color='blue')
            plt.axhline(y=best_run['optimal_flow'], color='red',
                       linestyle='--', label='Optimal')
            plt.title(f'Best Run Convergence for {filename}\n'
                     f'Final Flow: {best_run["best_value"]:.2f} | '
                     f'Optimal: {best_run["optimal_flow"]:.2f} | '
                     f'Iter: {best_run["best_iteration"]}')
            plt.xlabel('Iterations')
            plt.ylabel('Flow Value')
            plt.grid(True)
            plt.legend()
            best_plot_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_best_convergence.png")
            plt.savefig(best_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            all_flows = [r['best_value'] for r in runs_results]
            result = {
                'best': max(all_flows),
                'mean': np.mean(all_flows),
                'std': np.std(all_flows),
                'avg_iterations': np.mean([r['best_iteration'] for r in runs_results]),
                'avg_evaluations': np.mean([r['evaluations'] for r in runs_results]),
                'avg_time': np.mean([r['elapsed_time'] for r in runs_results]),
                'optimal_flow': runs_results[0]['optimal_flow']
            }
            results[filename] = result
            print(f"\nFinal statistics for {filename}:")
            print(f"Best Flow (TS): {result['best']:.2f} (Optimal: {result['optimal_flow']:.2f})")
            print(f"Mean Flow (TS): {result['mean']:.2f} ± {result['std']:.2f}")
            print(f"Avg Iterations: {result['avg_iterations']:.0f}")
            print(f"Avg Evaluations: {result['avg_evaluations']:.0f}")
            print(f"Avg Time: {result['avg_time']:.4f}s")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    return results

if __name__ == "__main__":
    experiment_results = run_experiments()

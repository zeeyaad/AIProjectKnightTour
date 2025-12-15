import random
from typing import List, Tuple, Optional


class KnightTourSolver:
    def __init__(self, n: int):
        self.n = n
        self.moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]

    def is_valid(self, x: int, y: int, board) -> bool:
        """Check if position is valid and unvisited"""
        return 0 <= x < self.n and 0 <= y < self.n and board[x][y] == -1

    def get_valid_moves(self, x: int, y: int, board) -> List[Tuple[int, int]]:
        """Get all valid knight moves from current position"""
        valid = []
        for dx, dy in self.moves:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny, board):
                valid.append((nx, ny))
        return valid

    def is_valid_knight_move(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions form a valid knight move"""
        x1, y1 = pos1
        x2, y2 = pos2
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        return (dx, dy) in [(1, 2), (2, 1)]

    def count_onward_moves(self, x: int, y: int, board) -> int:
        """Warnsdorff's heuristic: count accessibility of next moves"""
        return len(self.get_valid_moves(x, y, board))


class CulturalAlgorithmSolver(KnightTourSolver):
    def __init__(self, n: int, population_size: int = 100, generations: int = 2000):
        super().__init__(n)
        self.population_size = population_size
        self.generations = generations

        # Adaptive parameters
        self.initial_mutation_rate = 0.4
        self.initial_cultural_influence = 0.3
        self.mutation_rate = self.initial_mutation_rate
        self.cultural_influence = self.initial_cultural_influence

        # Diversity tracking
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0

    def generate_individual_greedy(self, start_x: int, start_y: int, randomness: float = 0.3):
        """Generate individual using Warnsdorff's heuristic with randomness"""
        path = [(start_x, start_y)]
        board = [[-1 for _ in range(self.n)] for _ in range(self.n)]
        board[start_x][start_y] = 0
        x, y = start_x, start_y

        for move_num in range(1, self.n * self.n):
            valid_moves = self.get_valid_moves(x, y, board)
            if not valid_moves:
                break

            # Apply Warnsdorff's heuristic with some randomness
            if random.random() > randomness:
                # Choose move with minimum onward moves (Warnsdorff)
                move_scores = []
                for nx, ny in valid_moves:
                    board[nx][ny] = move_num  # Temporarily mark
                    score = self.count_onward_moves(nx, ny, board)
                    board[nx][ny] = -1  # Unmark
                    move_scores.append((score, (nx, ny)))

                move_scores.sort()
                next_x, next_y = move_scores[0][1]
            else:
                # Random move for diversity
                next_x, next_y = random.choice(valid_moves)

            path.append((next_x, next_y))
            board[next_x][next_y] = move_num
            x, y = next_x, next_y

        return path

    def generate_individual_random(self, start_x: int, start_y: int):
        """Generate a random knight tour attempt"""
        path = [(start_x, start_y)]
        board = [[-1 for _ in range(self.n)] for _ in range(self.n)]
        board[start_x][start_y] = 0
        x, y = start_x, start_y

        for move_num in range(1, self.n * self.n):
            valid_moves = self.get_valid_moves(x, y, board)
            if not valid_moves:
                break
            next_x, next_y = random.choice(valid_moves)
            path.append((next_x, next_y))
            board[next_x][next_y] = move_num
            x, y = next_x, next_y

        return path

    def fitness(self, path: List[Tuple[int, int]]) -> float:
        """Enhanced fitness function considering coverage and move validity"""
        if len(path) == 0:
            return 0.0

        # Component 1: Unique squares visited (70% weight)
        unique_squares = len(set(path))
        coverage_score = unique_squares / (self.n * self.n)

        # Component 2: Valid knight moves (30% weight)
        valid_moves = 0
        for i in range(len(path) - 1):
            if self.is_valid_knight_move(path[i], path[i + 1]):
                valid_moves += 1

        move_validity_score = valid_moves / max(1, len(path) - 1) if len(path) > 1 else 1.0

        # Combined fitness
        fitness = coverage_score * 0.7 + move_validity_score * 0.3

        # Bonus for complete tours
        if unique_squares == self.n * self.n:
            fitness += 0.5

        return fitness

    def crossover(self, p1: List[Tuple[int, int]], p2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Improved crossover maintaining valid knight moves"""
        if len(p1) <= 1 or len(p2) <= 1:
            return p1.copy() if len(p1) > len(p2) else p2.copy()

        # Find a good crossover point (25-75% range)
        min_len = min(len(p1), len(p2))
        cp = random.randint(min_len // 4, 3 * min_len // 4)

        child = p1[:cp]
        visited = set(child)

        # Try to extend from p2 while maintaining valid moves
        for move in p2[cp:]:
            if move not in visited:
                # Check if it's a valid knight move from last position
                if child and self.is_valid_knight_move(child[-1], move):
                    child.append(move)
                    visited.add(move)
                    if len(child) >= self.n * self.n:
                        break

        # If child is too short, try to extend greedily
        if len(child) < min_len * 0.8:
            child = self.extend_path_greedy(child, visited)

        return child

    def extend_path_greedy(self, path: List[Tuple[int, int]], visited: set) -> List[Tuple[int, int]]:
        """Extend a partial path using greedy heuristic"""
        if not path:
            return path

        extended = path.copy()
        board = [[-1 for _ in range(self.n)] for _ in range(self.n)]

        # Mark visited squares
        for i, (x, y) in enumerate(extended):
            board[x][y] = i

        x, y = extended[-1]

        for step in range(len(extended), self.n * self.n):
            valid_moves = self.get_valid_moves(x, y, board)
            if not valid_moves:
                break

            # Use Warnsdorff's heuristic
            move_scores = []
            for nx, ny in valid_moves:
                board[nx][ny] = step
                score = self.count_onward_moves(nx, ny, board)
                board[nx][ny] = -1
                move_scores.append((score, (nx, ny)))

            move_scores.sort()
            next_x, next_y = move_scores[0][1]

            extended.append((next_x, next_y))
            board[next_x][next_y] = step
            x, y = next_x, next_y

        return extended

    def mutate(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Improved mutation with multiple strategies"""
        if len(path) <= 2:
            return path

        mutated = path.copy()
        mutation_type = random.random()

        if mutation_type < 0.4:
            # Strategy 1: Cut and regrow from random point
            mp = random.randint(1, len(path) - 1)
            board = [[-1 for _ in range(self.n)] for _ in range(self.n)]

            for i, (x, y) in enumerate(mutated[:mp]):
                board[x][y] = i

            x, y = mutated[mp - 1]
            valid = self.get_valid_moves(x, y, board)

            if valid:
                new = random.choice(valid)
                mutated = mutated[:mp]
                mutated.append(new)

                cx, cy = new
                board[cx][cy] = mp

                # Extend using Warnsdorff's heuristic
                for step in range(mp + 1, self.n * self.n):
                    valid_moves = self.get_valid_moves(cx, cy, board)
                    if not valid_moves:
                        break

                    # Warnsdorff's rule
                    move_scores = []
                    for nx, ny in valid_moves:
                        board[nx][ny] = step
                        score = self.count_onward_moves(nx, ny, board)
                        board[nx][ny] = -1
                        move_scores.append((score, (nx, ny)))

                    move_scores.sort()
                    nx, ny = move_scores[0][1]

                    mutated.append((nx, ny))
                    board[nx][ny] = step
                    cx, cy = nx, ny

        elif mutation_type < 0.7:
            # Strategy 2: Swap two subsequences
            if len(mutated) > 4:
                i1 = random.randint(1, len(mutated) - 3)
                i2 = random.randint(i1 + 1, len(mutated) - 2)

                # Try swap if moves remain valid
                test = mutated.copy()
                test[i1], test[i2] = test[i2], test[i1]

                # Check validity
                valid = True
                for i in range(len(test) - 1):
                    if not self.is_valid_knight_move(test[i], test[i + 1]):
                        valid = False
                        break

                if valid:
                    mutated = test

        else:
            # Strategy 3: Local optimization - try improving a segment
            if len(mutated) > 3:
                start = random.randint(0, len(mutated) - 3)
                end = min(start + random.randint(2, 5), len(mutated))

                # Rebuild segment
                board = [[-1 for _ in range(self.n)] for _ in range(self.n)]
                for i, (x, y) in enumerate(mutated):
                    if i < start or i >= end:
                        board[x][y] = i

                segment = mutated[:start]
                if segment:
                    x, y = segment[-1]
                    for _ in range(end - start):
                        valid = self.get_valid_moves(x, y, board)
                        if not valid:
                            break
                        nx, ny = random.choice(valid)
                        segment.append((nx, ny))
                        board[nx][ny] = len(segment) - 1
                        x, y = nx, ny

                    segment.extend(mutated[end:])
                    mutated = segment

        return mutated

    def calculate_diversity(self, population: List[List[Tuple[int, int]]]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 1.0

        diversity_sum = 0.0
        comparisons = 0

        for i in range(min(10, len(population))):
            for j in range(i + 1, min(10, len(population))):
                # Calculate difference ratio
                p1, p2 = set(population[i]), set(population[j])
                if len(p1) > 0 and len(p2) > 0:
                    difference = len(p1.symmetric_difference(p2))
                    total = len(p1.union(p2))
                    diversity_sum += difference / total if total > 0 else 0
                    comparisons += 1

        return diversity_sum / comparisons if comparisons > 0 else 0.5

    def adapt_parameters(self, generation: int, best_fitness: float, diversity: float):
        """Adapt mutation rate and cultural influence based on progress"""
        # Track stagnation
        if abs(best_fitness - self.last_best_fitness) < 0.001:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        self.last_best_fitness = best_fitness

        # Increase mutation if stagnating or low diversity
        if self.stagnation_counter > 50 or diversity < 0.3:
            self.mutation_rate = min(0.7, self.mutation_rate + 0.05)
            self.cultural_influence = max(0.1, self.cultural_influence - 0.02)
        else:
            # Gradually decrease mutation rate as we progress
            self.mutation_rate = max(0.2, self.initial_mutation_rate * (1 - generation / self.generations))
            self.cultural_influence = min(0.5, self.initial_cultural_influence + generation / self.generations * 0.2)

    def solve(self, start_x: int, start_y: int, verbose: bool = False):
        """Solve Knight's Tour using improved Cultural Algorithm"""

        # Initialize population with mix of strategies
        population = []
        for i in range(self.population_size):
            if i < self.population_size * 0.6:
                # 60% greedy with varying randomness
                individual = self.generate_individual_greedy(start_x, start_y, randomness=random.uniform(0.1, 0.5))
            else:
                # 40% random for diversity
                individual = self.generate_individual_random(start_x, start_y)
            population.append(individual)

        belief_space = []
        evaluations = 0
        best_ever_fitness = 0.0
        best_ever_individual = None

        for gen in range(self.generations):
            # Evaluate fitness
            scored = [(ind, self.fitness(ind)) for ind in population]
            evaluations += len(population)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Update best ever
            if scored[0][1] > best_ever_fitness:
                best_ever_fitness = scored[0][1]
                best_ever_individual = scored[0][0]

            # Update belief space (keep top performers without duplicates)
            for ind, fit in scored[:5]:
                # Check if already in belief space
                is_duplicate = False
                for belief_ind in belief_space:
                    if set(ind) == set(belief_ind):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    belief_space.append(ind)

            belief_space.sort(key=self.fitness, reverse=True)
            belief_space = belief_space[:15]  # Keep top 15

            # Check for solution
            best_ind, best_fit = scored[0]
            if verbose and gen % 100 == 0:
                diversity = self.calculate_diversity(population)
                print(f"Gen {gen}: Best fitness = {best_fit:.4f}, "
                      f"Best length = {len(best_ind)}, "
                      f"Diversity = {diversity:.4f}, "
                      f"Mutation rate = {self.mutation_rate:.3f}")

            if best_fit >= 1.5:  # Complete tour (1.0 + 0.5 bonus)
                board = [[-1] * self.n for _ in range(self.n)]
                for step, (x, y) in enumerate(best_ind):
                    board[x][y] = step
                if verbose:
                    print(f"\n✓ Solution found in generation {gen}!")
                    print(f"Total evaluations: {evaluations}")
                return True, board, evaluations

            # Adaptive parameter adjustment
            diversity = self.calculate_diversity(population)
            self.adapt_parameters(gen, best_fit, diversity)

            # Generate new population
            elite_count = max(2, self.population_size // 20)
            new_pop = [ind for ind, _ in scored[:elite_count]]

            while len(new_pop) < self.population_size:
                # Tournament selection
                tournament_size = 3
                p1 = max(random.sample(population, tournament_size), key=self.fitness)
                p2 = max(random.sample(population, tournament_size), key=self.fitness)

                # Crossover
                child = self.crossover(p1, p2)

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                # Cultural influence from belief space
                if belief_space and random.random() < self.cultural_influence:
                    elite = random.choice(belief_space)
                    child = self.crossover(child, elite)

                new_pop.append(child)

            population = new_pop

            # Restart mechanism if stuck
            if self.stagnation_counter > 100:
                if verbose:
                    print(f"Gen {gen}: Restarting population due to stagnation...")

                # Keep best individuals
                keep_count = self.population_size // 4
                population = [ind for ind, _ in scored[:keep_count]]

                # Generate new individuals
                for _ in range(self.population_size - keep_count):
                    population.append(
                        self.generate_individual_greedy(start_x, start_y, randomness=random.uniform(0.2, 0.7)))

                self.stagnation_counter = 0
                self.mutation_rate = self.initial_mutation_rate

        # Return best found
        best = best_ever_individual if best_ever_individual else max(population, key=self.fitness)
        board = [[-1] * self.n for _ in range(self.n)]
        for step, (x, y) in enumerate(best):
            board[x][y] = step

        success = self.fitness(best) >= 1.5
        if verbose:
            print(f"\nFinal best fitness: {self.fitness(best):.4f}")
            print(f"Path length: {len(best)}/{self.n * self.n}")
            print(f"Total evaluations: {evaluations}")

        return success, board, evaluations


# # Example usage
# if __name__ == "__main__":
#     # Test on 5x5 board
#     print("Testing on 5x5 board:")
#     print("=" * 50)
#     solver = ImprovedCulturalAlgorithmSolver(n=5, population_size=100, generations=1000)
#     success, board, evals = solver.solve(0, 0, verbose=True)
#
#     if success:
#         print("\n✓ Knight's Tour Solution Found!")
#         print("\nBoard (move numbers):")
#         for row in board:
#             print(" ".join(f"{cell:3}" for cell in row))
#     else:
#         print("\n✗ Could not find complete tour")
#         print("Best solution found:")
#         for row in board:
#             print(" ".join(f"{cell:3}" for cell in row))
#
#     print(f"\n{'=' * 50}\n")
#
#     # Test on 6x6 board
#     print("Testing on 6x6 board:")
#     print("=" * 50)
#     solver = ImprovedCulturalAlgorithmSolver(n=6, population_size=150, generations=1500)
#     success, board, evals = solver.solve(0, 0, verbose=True)
#
#     if success:
#         print("\n✓ Knight's Tour Solution Found!")
#         print("\nBoard (move numbers):")
#         for row in board:
#             print(" ".join(f"{cell:3}" for cell in row))
#     else:
#         print("\n✗ Could not find complete tour")
#         print("Best solution found:")
#         for row in board:
#             print(" ".join(f"{cell:3}" for cell in row))
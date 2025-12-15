# import random
# from typing import List, Tuple, Set
#
#
# class PureCulturalAlgorithm:
#     """
#     Pure Cultural Algorithm for Knight's Tour Problem
#     No domain-specific heuristics - only evolutionary operators and cultural knowledge
#     """
#
#     def __init__(self, n: int, population_size: int = 100, generations: int = 2000):
#         self.n = n
#         self.population_size = population_size
#         self.generations = generations
#
#         # Knight move offsets
#         self.moves = [
#             (2, 1), (1, 2), (-1, 2), (-2, 1),
#             (-2, -1), (-1, -2), (1, -2), (2, -1)
#         ]
#
#         # Cultural Algorithm components
#         self.belief_space = {
#             'normative': [],  # Best solutions found
#             'situational': {},  # Successful patterns/positions
#             'domain': set()  # Visited configurations
#         }
#
#         # Parameters
#         self.mutation_rate = 0.3
#         self.cultural_influence_rate = 0.4
#         self.elite_size = 5
#
#     def is_valid_position(self, x: int, y: int) -> bool:
#         """Check if position is on the board"""
#         return 0 <= x < self.n and 0 <= y < self.n
#
#     def is_valid_knight_move(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
#         """Check if two positions form a valid knight move"""
#         dx = abs(pos2[0] - pos1[0])
#         dy = abs(pos2[1] - pos1[1])
#         return (dx, dy) in [(1, 2), (2, 1)]
#
#     def get_valid_moves_from(self, x: int, y: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """Get all valid unvisited knight moves from position"""
#         valid = []
#         for dx, dy in self.moves:
#             nx, ny = x + dx, y + dy
#             if self.is_valid_position(nx, ny) and (nx, ny) not in visited:
#                 valid.append((nx, ny))
#         return valid
#
#     def generate_random_individual(self, start_x: int, start_y: int) -> List[Tuple[int, int]]:
#         """
#         Generate a random individual (path) using only random selection
#         NO HEURISTICS - purely random walk
#         """
#         path = [(start_x, start_y)]
#         visited = {(start_x, start_y)}
#
#         current = (start_x, start_y)
#
#         # Continue until no valid moves available
#         while len(path) < self.n * self.n:
#             valid_moves = self.get_valid_moves_from(current[0], current[1], visited)
#
#             if not valid_moves:
#                 break
#
#             # Pure random selection - NO HEURISTICS
#             next_move = random.choice(valid_moves)
#             path.append(next_move)
#             visited.add(next_move)
#             current = next_move
#
#         return path
#
#     def fitness(self, path: List[Tuple[int, int]]) -> float:
#         """
#         Fitness function based on:
#         1. Number of unique squares visited
#         2. Validity of knight moves
#         3. Bonus for complete tour
#         """
#         if not path:
#             return 0.0
#
#         # Component 1: Coverage (50%)
#         unique_squares = len(set(path))
#         coverage = unique_squares / (self.n * self.n)
#
#         # Component 2: Move validity (30%)
#         valid_moves = 0
#         for i in range(len(path) - 1):
#             if self.is_valid_knight_move(path[i], path[i + 1]):
#                 valid_moves += 1
#
#         validity = valid_moves / max(1, len(path) - 1) if len(path) > 1 else 1.0
#
#         # Component 3: Path continuity (20%)
#         continuity = len(path) / (self.n * self.n)
#
#         fitness = coverage * 0.5 + validity * 0.3 + continuity * 0.2
#
#         # Bonus for complete tour
#         if unique_squares == self.n * self.n and validity == 1.0:
#             fitness += 1.0
#
#         return fitness
#
#     def crossover(self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Order-based crossover for path representation
#         Combines portions of two parents while maintaining path validity
#         """
#         if len(parent1) <= 1:
#             return parent1.copy()
#         if len(parent2) <= 1:
#             return parent2.copy()
#
#         # Choose crossover point
#         min_len = min(len(parent1), len(parent2))
#         crossover_point = random.randint(1, max(1, min_len - 1))
#
#         # Start with prefix from parent1
#         child = parent1[:crossover_point].copy()
#         visited = set(child)
#
#         # Try to add moves from parent2 that are valid
#         for pos in parent2:
#             if pos not in visited:
#                 # Check if it forms valid knight move from last position
#                 if child and self.is_valid_knight_move(child[-1], pos):
#                     child.append(pos)
#                     visited.add(pos)
#
#         # If child is too short, try adding from parent1
#         if len(child) < min_len * 0.5:
#             for pos in parent1[crossover_point:]:
#                 if pos not in visited:
#                     if child and self.is_valid_knight_move(child[-1], pos):
#                         child.append(pos)
#                         visited.add(pos)
#
#         return child
#
#     def mutate(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Mutation operators (NO HEURISTICS):
#         1. Truncate and regrow randomly
#         2. Swap random segments
#         3. Insert random detour
#         """
#         if len(individual) <= 2:
#             return individual
#
#         mutated = individual.copy()
#         mutation_type = random.random()
#
#         if mutation_type < 0.4:
#             # Mutation 1: Truncate and regrow randomly
#             truncate_point = random.randint(1, len(mutated) - 1)
#             mutated = mutated[:truncate_point]
#             visited = set(mutated)
#             current = mutated[-1]
#
#             # Regrow randomly (NO HEURISTICS)
#             while len(mutated) < self.n * self.n:
#                 valid_moves = self.get_valid_moves_from(current[0], current[1], visited)
#                 if not valid_moves:
#                     break
#
#                 next_move = random.choice(valid_moves)  # Pure random
#                 mutated.append(next_move)
#                 visited.add(next_move)
#                 current = next_move
#
#         elif mutation_type < 0.7:
#             # Mutation 2: Reverse a random segment
#             if len(mutated) > 3:
#                 start = random.randint(1, len(mutated) - 2)
#                 end = random.randint(start + 1, len(mutated))
#                 mutated = mutated[:start] + mutated[start:end][::-1] + mutated[end:]
#
#         else:
#             # Mutation 3: Random restart from middle
#             if len(mutated) > 4:
#                 restart_point = random.randint(2, len(mutated) - 2)
#                 valid_positions = []
#
#                 for i in range(self.n):
#                     for j in range(self.n):
#                         if (i, j) not in set(mutated[:restart_point]):
#                             if self.is_valid_knight_move(mutated[restart_point - 1], (i, j)):
#                                 valid_positions.append((i, j))
#
#                 if valid_positions:
#                     new_pos = random.choice(valid_positions)
#                     mutated = mutated[:restart_point] + [new_pos]
#
#                     # Continue randomly
#                     visited = set(mutated)
#                     current = new_pos
#
#                     while len(mutated) < self.n * self.n:
#                         valid_moves = self.get_valid_moves_from(current[0], current[1], visited)
#                         if not valid_moves:
#                             break
#                         next_move = random.choice(valid_moves)
#                         mutated.append(next_move)
#                         visited.add(next_move)
#                         current = next_move
#
#         return mutated
#
#     def update_belief_space(self, population: List[List[Tuple[int, int]]]):
#         """
#         Update cultural knowledge from population
#         This is the KEY component of Cultural Algorithm
#         """
#         # Sort by fitness
#         scored = [(ind, self.fitness(ind)) for ind in population]
#         scored.sort(key=lambda x: x[1], reverse=True)
#
#         # Update normative knowledge (best solutions)
#         for ind, fit in scored[:self.elite_size]:
#             self.belief_space['normative'].append((ind.copy(), fit))
#
#         # Keep only top solutions
#         self.belief_space['normative'].sort(key=lambda x: x[1], reverse=True)
#         self.belief_space['normative'] = self.belief_space['normative'][:20]
#
#         # Update situational knowledge (good positions/patterns)
#         for ind, fit in scored[:10]:
#             if fit > 0.5:  # Only learn from reasonably good solutions
#                 for i, pos in enumerate(ind):
#                     if pos not in self.belief_space['situational']:
#                         self.belief_space['situational'][pos] = {'count': 0, 'avg_step': 0, 'fitness_sum': 0}
#
#                     self.belief_space['situational'][pos]['count'] += 1
#                     self.belief_space['situational'][pos]['avg_step'] += i
#                     self.belief_space['situational'][pos]['fitness_sum'] += fit
#
#         # Prune situational knowledge (keep most frequent)
#         if len(self.belief_space['situational']) > 100:
#             sorted_pos = sorted(
#                 self.belief_space['situational'].items(),
#                 key=lambda x: x[1]['count'],
#                 reverse=True
#             )
#             self.belief_space['situational'] = dict(sorted_pos[:100])
#
#     def apply_cultural_influence(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Apply knowledge from belief space to individual
#         This is pure cultural learning - NO DOMAIN HEURISTICS
#         """
#         if not self.belief_space['normative']:
#             return individual
#
#         # Select a random elite from normative knowledge
#         elite, _ = random.choice(self.belief_space['normative'])
#
#         # Perform crossover with elite
#         influenced = self.crossover(individual, elite)
#
#         return influenced
#
#     def tournament_selection(self, population: List[List[Tuple[int, int]]],
#                              tournament_size: int = 3) -> List[Tuple[int, int]]:
#         """Select individual using tournament selection"""
#         tournament = random.sample(population, min(tournament_size, len(population)))
#         return max(tournament, key=self.fitness)
#
#     def solve(self, start_x: int, start_y: int, verbose: bool = False):
#         """
#         Main Cultural Algorithm loop
#         """
#         # Initialize population randomly (NO HEURISTICS)
#         population = []
#         for _ in range(self.population_size):
#             individual = self.generate_random_individual(start_x, start_y)
#             population.append(individual)
#
#         best_ever = None
#         best_ever_fitness = 0.0
#         evaluations = 0
#
#         for generation in range(self.generations):
#             # Evaluate population
#             scored = [(ind, self.fitness(ind)) for ind in population]
#             evaluations += len(population)
#             scored.sort(key=lambda x: x[1], reverse=True)
#
#             best_ind, best_fit = scored[0]
#
#             # Track best ever
#             if best_fit > best_ever_fitness:
#                 best_ever_fitness = best_fit
#                 best_ever = best_ind.copy()
#
#             # Update belief space (CULTURAL COMPONENT)
#             self.update_belief_space(population)
#
#             # Check for solution
#             if best_fit >= 1.5:  # Complete valid tour
#                 if verbose:
#                     print(f"\n✓ Solution found in generation {generation}!")
#                     print(f"Fitness: {best_fit:.4f}")
#                     print(f"Path length: {len(best_ind)}/{self.n * self.n}")
#                     print(f"Total evaluations: {evaluations}")
#
#                 board = [[-1] * self.n for _ in range(self.n)]
#                 for step, (x, y) in enumerate(best_ind):
#                     board[x][y] = step
#                 return True, board, evaluations
#
#             if verbose and generation % 100 == 0:
#                 print(f"Gen {generation}: Best fitness = {best_fit:.4f}, "
#                       f"Length = {len(best_ind)}/{self.n * self.n}, "
#                       f"Belief space size = {len(self.belief_space['normative'])}")
#
#             # Generate new population
#             new_population = []
#
#             # Elitism: keep best individuals
#             elite_count = max(2, self.population_size // 20)
#             new_population.extend([ind for ind, _ in scored[:elite_count]])
#
#             # Generate rest through evolution + cultural influence
#             while len(new_population) < self.population_size:
#                 # Selection
#                 parent1 = self.tournament_selection(population)
#                 parent2 = self.tournament_selection(population)
#
#                 # Crossover
#                 child = self.crossover(parent1, parent2)
#
#                 # Mutation
#                 if random.random() < self.mutation_rate:
#                     child = self.mutate(child)
#
#                 # Cultural influence (KEY CULTURAL ALGORITHM COMPONENT)
#                 if random.random() < self.cultural_influence_rate:
#                     child = self.apply_cultural_influence(child)
#
#                 new_population.append(child)
#
#             population = new_population
#
#         # Return best found
#         if verbose:
#             print(f"\nSearch completed. Best fitness: {best_ever_fitness:.4f}")
#             print(f"Best path length: {len(best_ever)}/{self.n * self.n}")
#             print(f"Total evaluations: {evaluations}")
#
#         board = [[-1] * self.n for _ in range(self.n)]
#         for step, (x, y) in enumerate(best_ever):
#             board[x][y] = step
#
#         return best_ever_fitness >= 1.5, board, evaluations
#
#
# # Example usage
# # if __name__ == "__main__":
# #     print("Pure Cultural Algorithm - Knight's Tour")
# #     print("=" * 60)
# #     print("Note: This uses NO domain-specific heuristics")
# #     print("Only evolutionary operators + cultural knowledge learning")
# #     print("=" * 60)
# #
# #     # Test on 5x5 board
# #     print("\nTesting on 5x5 board:")
# #     print("-" * 60)
# #     solver = PureCulturalAlgorithm(n=5, population_size=150, generations=1000)
# #     success, board, evals = solver.solve(0, 0, verbose=True)
# #
# #     if success:
# #         print("\n✓ Complete Knight's Tour Found!")
# #         print("\nBoard (move sequence):")
# #         for row in board:
# #             print(" ".join(f"{cell:3}" for cell in row))
# #     else:
# #         print("\n✗ Complete tour not found")
# #         print("Best solution:")
# #         for row in board:
# #             print(" ".join(f"{cell:3}" for cell in row))
# #
# #     print("\n" + "=" * 60)
# #
# #     # Test on 6x6 board
# #     print("\nTesting on 6x6 board:")
# #     print("-" * 60)
# #     solver = PureCulturalAlgorithm(n=6, population_size=200, generations=2000)
# #     success, board, evals = solver.solve(0, 0, verbose=True)
# #
# #     if success:
# #         print("\n✓ Complete Knight's Tour Found!")
# #         print("\nBoard (move sequence):")
# #         for row in board:
# #             print(" ".join(f"{cell:3}" for cell in row))
# #     else:
# #         print("\n✗ Complete tour not found")
# #         print("Best solution:")
# #         for row in board:
# #             print(" ".join(f"{cell:3}" for cell in row))

# import random
# from typing import List, Tuple, Set
#
#
# class PureCulturalAlgorithm:
#     """
#     Pure Cultural Algorithm for Knight's Tour Problem
#     """
#
#     def __init__(self, board_size: int, population_size: int = 100, max_generations: int = 2000):
#         self.board_size = board_size
#         self.population_size = population_size
#         self.max_generations = max_generations
#
#         # Knight move offsets
#         self.knight_move_offsets = [
#             (2, 1), (1, 2), (-1, 2), (-2, 1),
#             (-2, -1), (-1, -2), (1, -2), (2, -1)
#         ]
#
#         # Cultural Algorithm components
#         self.cultural_knowledge = {
#             'best_solutions': [],  # Best solutions 'Elite' found (normative knowledge)
#             'successful_patterns': {},  # Successful patterns/positions 'squares frequently used in good solutions' (situational knowledge)
#             'explored_configurations': set()  # Visited configurations 'Visited Paths' (domain knowledge)
#         }
#
#         # Evolutionary parameters
#         self.mutation_probability = 0.3
#         self.cultural_influence_probability = 0.4 #'Zb:- New Generation Will Be Effected By Belief Space By 40%'
#         self.elite_preservation_count = 5 # Keep The Best 5 Individuals 'Elite' Unchanged For Next generation
#
#     def is_position_on_board(self, row: int, col: int) -> bool:
#         """Check if position is within board boundaries"""
#         return 0 <= row < self.board_size and 0 <= col < self.board_size
#
#     def is_valid_knight_move(self, position1: Tuple[int, int], position2: Tuple[int, int]) -> bool:
#         """Check if two positions form a legal knight move"""
#         row_distance = abs(position2[0] - position1[0])
#         col_distance = abs(position2[1] - position1[1])
#         return (row_distance, col_distance) in [(1, 2), (2, 1)] # This Mean Move Must Be In L Shape
#
#     def get_unvisited_valid_moves(self, row: int, col: int, visited_squares: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """Get all valid unvisited knight moves from current position By Using Random Path Generation, Mutation Logic"""
#         available_moves = []
#         for row_offset, col_offset in self.knight_move_offsets:
#             """Loop Through All Valid Moves '8' In Each Iteration Pick One Offset (x, y)"""
#             next_row = row + row_offset
#             next_col = col + col_offset
#             """For New Offset 'Mean Form My Position To New One' """
#             if self.is_position_on_board(next_row, next_col) and (next_row, next_col) not in visited_squares:
#                 """If Position Is Valid and Inside Board, Not Already Visited"""
#                 available_moves.append((next_row, next_col))
#         return available_moves # Return All Possible Moves
#
#     def create_random_path(self, start_row: int, start_col: int) -> List[Tuple[int, int]]:
#         """
#         Generate a random individual path using pure random selection
#         Start from Start_row, _Col
#         Then Randomly Choose One Available Unvisited Moves
#         Continue Until No Move Left, Path Complete All N^2 Squares Mean if N=8 Path len =8^2
#         """
#         path = [(start_row, start_col)]
#         visited_squares = {(start_row, start_col)}
#         current_position = (start_row, start_col)
#
#         # Continue until no valid moves available or board is complete
#         while len(path) < self.board_size * self.board_size:
#             available_moves = self.get_unvisited_valid_moves(
#                 current_position[0], current_position[1], visited_squares
#             )
#
#             if not available_moves:
#                 break
#
#             # Pure random selection - NO HEURISTICS
#             next_position = random.choice(available_moves)
#             path.append(next_position)
#             visited_squares.add(next_position)
#             current_position = next_position
#
#         return path
#
#     def calculate_path_fitness(self, path: List[Tuple[int, int]]) -> float:
#         """Take List Of Coordinates Representing Knight's Path return Fittness Score Between 0.0, 2.0"""
#         """
#         Calculate fitness score based on:
#         1. Number of unique squares visited (coverage)
#         2. Validity of knight moves
#         3. Path continuity
#         4. Bonus for complete tour +1
#         """
#         if not path:
#             return 0.0 # If Path Is Empty therefore Fitness is 0
#
#         total_squares = self.board_size * self.board_size
#
#         # Component 1: Coverage (50% Weight)
#         """Ensure that A Knight Visit All Square Exactly Once"""
#         unique_squares_visited = len(set(path))
#         coverage_score = unique_squares_visited / total_squares
#
#         # Component 2: Move validity (30%)
#         valid_moves_count = 0
#         for move_index in range(len(path) - 1):
#             """This checks every pair of consecutive moves."""
#             if self.is_valid_knight_move(path[move_index], path[move_index + 1]):
#                 valid_moves_count += 1
#
#         validity_score = valid_moves_count / max(1, len(path) - 1) if len(path) > 1 else 1.0
#         """
#             If all moves are valid knight jumps → score = 1.0
#
#             If half are valid → score = 0.5
#
#             If none are valid → score = 0
#         """
#         # Component 3: Path continuity (20%)
#         """How long is the path relative to the total board size"""
#         continuity_score = len(path) / total_squares
#
#         fitness_score = coverage_score * 0.5 + validity_score * 0.3 + continuity_score * 0.2
#
#         # Bonus for complete valid tour For Perfect Tour "" This Mean Visits All Squares and All Moves Valid""
#         if unique_squares_visited == total_squares and validity_score == 1.0:
#             fitness_score += 1.0
#
#         return fitness_score
#
#     # From Here
#
#     def combine_parent_paths(self, parent1: List[Tuple[int, int]], parent2: List[Tuple[int, int]]) -> List[
#         Tuple[int, int]]:
#         """
#         Order-based crossover combining portions of two parent paths
#         Maintains path validity while mixing genetic material
#         """
#         if len(parent1) <= 1:
#             return parent1.copy()
#         if len(parent2) <= 1:
#             return parent2.copy()
#
#         # Choose crossover point
#         shorter_path_length = min(len(parent1), len(parent2))
#         crossover_point = random.randint(1, max(1, shorter_path_length - 1))
#
#         # Start with prefix from parent1
#         offspring = parent1[:crossover_point].copy()
#         visited_in_offspring = set(offspring)
#
#         # Try to add moves from parent2 that are valid
#         for position in parent2:
#             if position not in visited_in_offspring:
#                 if self.is_valid_knight_move(offspring[-1], position):  # offspring always has elements
#                     offspring.append(position)
#                     visited_in_offspring.add(position)
#
#         # If offspring is too short, try adding from BOTH parents
#         min_acceptable_length = max(3, shorter_path_length * 0.5)  # At least 3 moves
#         if len(offspring) < min_acceptable_length:
#             # Try parent1 first
#             for position in parent1[crossover_point:]:
#                 if position not in visited_in_offspring:
#                     if self.is_valid_knight_move(offspring[-1], position):
#                         offspring.append(position)
#                         visited_in_offspring.add(position)
#
#             # If still too short, try parent2 again (circular search)
#             if len(offspring) < min_acceptable_length:
#                 for position in parent2:
#                     if position not in visited_in_offspring:
#                         if self.is_valid_knight_move(offspring[-1], position):
#                             offspring.append(position)
#                             visited_in_offspring.add(position)
#
#         return offspring
#
#     def apply_random_mutation(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Apply mutation operators (NO HEURISTICS):
#         1. Truncate and regrow randomly
#         2. Reverse random segment
#         3. Random restart from middle point
#         """
#         if len(individual) <= 2:
#             return individual
#
#         mutated_path = individual.copy()
#         mutation_type = random.random()
#
#         if mutation_type < 0.4:
#             # Mutation 1: Truncate and regrow randomly
#             truncation_point = random.randint(1, len(mutated_path) - 1)
#             mutated_path = mutated_path[:truncation_point]
#             visited_squares = set(mutated_path)
#             current_position = mutated_path[-1]
#
#             # Regrow randomly (NO HEURISTICS)
#             while len(mutated_path) < self.board_size * self.board_size:
#                 available_moves = self.get_unvisited_valid_moves(
#                     current_position[0], current_position[1], visited_squares
#                 )
#                 if not available_moves:
#                     break
#
#                 next_position = random.choice(available_moves)  # Pure random
#                 mutated_path.append(next_position)
#                 visited_squares.add(next_position)
#                 current_position = next_position
#
#         elif mutation_type < 0.7:
#             # Mutation 2: Reverse a random segment
#             if len(mutated_path) > 3:
#                 segment_start = random.randint(1, len(mutated_path) - 2)
#                 segment_end = random.randint(segment_start + 1, len(mutated_path))
#                 mutated_path = (mutated_path[:segment_start] +
#                                mutated_path[segment_start:segment_end][::-1] +
#                                mutated_path[segment_end:])
#
#         else:
#             # Mutation 3: Random restart from middle point
#             if len(mutated_path) > 4:
#                 restart_point = random.randint(2, len(mutated_path) - 2)
#                 unvisited_valid_positions = []
#
#                 for row in range(self.board_size):
#                     for col in range(self.board_size):
#                         if (row, col) not in set(mutated_path[:restart_point]):
#                             if self.is_valid_knight_move(mutated_path[restart_point - 1], (row, col)):
#                                 unvisited_valid_positions.append((row, col))
#
#                 if unvisited_valid_positions:
#                     new_position = random.choice(unvisited_valid_positions)
#                     mutated_path = mutated_path[:restart_point] + [new_position]
#
#                     # Continue randomly
#                     visited_squares = set(mutated_path)
#                     current_position = new_position
#
#                     while len(mutated_path) < self.board_size * self.board_size:
#                         available_moves = self.get_unvisited_valid_moves(
#                             current_position[0], current_position[1], visited_squares
#                         )
#                         if not available_moves:
#                             break
#                         next_position = random.choice(available_moves)
#                         mutated_path.append(next_position)
#                         visited_squares.add(next_position)
#                         current_position = next_position
#
#         return mutated_path
#
#     def update_cultural_knowledge_base(self, population: List[List[Tuple[int, int]]]):
#         """
#         Update cultural knowledge from current population
#         This is the KEY component of Cultural Algorithm
#         """
#         # Evaluate and sort population by fitness
#         population_with_fitness = [(individual, self.calculate_path_fitness(individual))
#                                    for individual in population]
#         population_with_fitness.sort(key=lambda x: x[1], reverse=True)
#
#         # Update normative knowledge (best solutions)
#         for individual, fitness in population_with_fitness[:self.elite_preservation_count]:
#             self.cultural_knowledge['best_solutions'].append((individual.copy(), fitness))
#
#         # Keep only top solutions in belief space
#         self.cultural_knowledge['best_solutions'].sort(key=lambda x: x[1], reverse=True)
#         self.cultural_knowledge['best_solutions'] = self.cultural_knowledge['best_solutions'][:20]
#
#         # Update situational knowledge (good positions/patterns)
#         for individual, fitness in population_with_fitness[:10]:
#             if fitness > 0.5:  # Only learn from reasonably good solutions
#                 for step_index, position in enumerate(individual):
#                     if position not in self.cultural_knowledge['successful_patterns']:
#                         self.cultural_knowledge['successful_patterns'][position] = {
#                             'occurrence_count': 0,
#                             'average_step': 0,
#                             'total_fitness': 0
#                         }
#
#                     self.cultural_knowledge['successful_patterns'][position]['occurrence_count'] += 1
#                     self.cultural_knowledge['successful_patterns'][position]['average_step'] += step_index
#                     self.cultural_knowledge['successful_patterns'][position]['total_fitness'] += fitness
#
#         # Prune situational knowledge (keep most frequent patterns)
#         if len(self.cultural_knowledge['successful_patterns']) > 100:
#             sorted_patterns = sorted(
#                 self.cultural_knowledge['successful_patterns'].items(),
#                 key=lambda x: x[1]['occurrence_count'],
#                 reverse=True
#             )
#             self.cultural_knowledge['successful_patterns'] = dict(sorted_patterns[:100])
#
#     def apply_cultural_learning(self, individual: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
#         """
#         Apply knowledge from cultural belief space to individual
#         This is pure cultural learning - NO DOMAIN HEURISTICS
#         """
#         if not self.cultural_knowledge['best_solutions']:
#             return individual
#
#         # Select a random elite solution from normative knowledge
#         elite_solution, _ = random.choice(self.cultural_knowledge['best_solutions'])
#
#         # Perform crossover with elite solution
#         culturally_influenced = self.combine_parent_paths(individual, elite_solution)
#
#         return culturally_influenced
#
#     def select_parent_via_tournament(self, population: List[List[Tuple[int, int]]],
#                                      tournament_size: int = 3) -> List[Tuple[int, int]]:
#         """Select individual using tournament selection"""
#         tournament_competitors = random.sample(population, min(tournament_size, len(population)))
#         return max(tournament_competitors, key=self.calculate_path_fitness)
#
#     def solve(self, start_row: int, start_col: int, verbose: bool = False):
#         """
#         Main Cultural Algorithm evolutionary loop
#         """
#         # Initialize population randomly (NO HEURISTICS)
#         population = []
#         for _ in range(self.population_size):
#             random_individual = self.create_random_path(start_row, start_col)
#             population.append(random_individual)
#
#         best_solution_ever = None
#         best_fitness_ever = 0.0
#         total_fitness_evaluations = 0
#
#         for generation_number in range(self.max_generations):
#             # Evaluate entire population
#             population_with_fitness = [(ind, self.calculate_path_fitness(ind)) for ind in population]
#             total_fitness_evaluations += len(population)
#             population_with_fitness.sort(key=lambda x: x[1], reverse=True)
#
#             current_best_individual, current_best_fitness = population_with_fitness[0]
#
#             # Track best solution ever found
#             if current_best_fitness > best_fitness_ever:
#                 best_fitness_ever = current_best_fitness
#                 best_solution_ever = current_best_individual.copy()
#
#             # Update cultural belief space (CULTURAL COMPONENT)
#             self.update_cultural_knowledge_base(population)
#
#             # Check for complete solution
#             if current_best_fitness >= 1.5:  # Complete valid tour
#                 if verbose:
#                     print(f"\n✓ Solution found in generation {generation_number}!")
#                     print(f"Fitness: {current_best_fitness:.4f}")
#                     print(f"Path length: {len(current_best_individual)}/{self.board_size * self.board_size}")
#                     print(f"Total evaluations: {total_fitness_evaluations}")
#
#                 board = [[-1] * self.board_size for _ in range(self.board_size)]
#                 for step, (row, col) in enumerate(current_best_individual):
#                     board[row][col] = step
#                 return True, board, total_fitness_evaluations
#
#             if verbose and generation_number % 100 == 0:
#                 print(f"Gen {generation_number}: Best fitness = {current_best_fitness:.4f}, "
#                       f"Length = {len(current_best_individual)}/{self.board_size * self.board_size}, "
#                       f"Belief space size = {len(self.cultural_knowledge['best_solutions'])}")
#
#             # Generate new population through evolution
#             new_population = []
#
#             # Elitism: preserve best individuals
#             elite_count = max(2, self.population_size // 20)
#             new_population.extend([ind for ind, _ in population_with_fitness[:elite_count]])
#
#             # Generate remaining individuals through evolution + cultural influence
#             while len(new_population) < self.population_size:
#                 # Parent selection
#                 parent1 = self.select_parent_via_tournament(population)
#                 parent2 = self.select_parent_via_tournament(population)
#
#                 # Crossover
#                 offspring = self.combine_parent_paths(parent1, parent2)
#
#                 # Mutation
#                 if random.random() < self.mutation_probability:
#                     offspring = self.apply_random_mutation(offspring)
#
#                 # Cultural influence (KEY CULTURAL ALGORITHM COMPONENT)
#                 if random.random() < self.cultural_influence_probability:
#                     offspring = self.apply_cultural_learning(offspring)
#
#                 new_population.append(offspring)
#
#             population = new_population
#
#         # Return best solution found
#         if verbose:
#             print(f"\nSearch completed. Best fitness: {best_fitness_ever:.4f}")
#             print(f"Best path length: {len(best_solution_ever)}/{self.board_size * self.board_size}")
#             print(f"Total evaluations: {total_fitness_evaluations}")
#
#         board = [[-1] * self.board_size for _ in range(self.board_size)]
#         for step, (row, col) in enumerate(best_solution_ever):
#             board[row][col] = step
#
#         return best_fitness_ever >= 1.5, board, total_fitness_evaluations


import random
from typing import List, Tuple, Set, Dict
from collections import deque


class PureCulturalAlgorithm:
    """
    Optimized Cultural Algorithm for Knight's Tour Problem
    Key improvements:
    1. Cached fitness evaluations
    2. More efficient data structures
    3. Adaptive parameters
    4. Better memory management
    5. Improved crossover and mutation
    """

    def __init__(self, board_size: int, population_size: int = 100, max_generations: int = 2000):
        self.board_size = board_size
        self.population_size = population_size
        self.max_generations = max_generations
        self.total_squares = board_size * board_size

        # Pre-computed knight moves
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]

        # Cultural knowledge
        self.cultural_knowledge = {
            'best_solutions': [],
            'successful_patterns': {},
            'position_frequency': {}  # Track position usage frequency
        }

        # Adaptive parameters
        self.mutation_probability = 0.3
        self.cultural_influence_probability = 0.4
        self.elite_count = max(2, population_size // 20)

        # Fitness cache (optimization)
        self.fitness_cache: Dict[tuple, float] = {}

        # Pre-compute valid positions for each board cell
        self._precompute_valid_moves()

    def _precompute_valid_moves(self):
        """Pre-compute valid moves from each position"""
        self.valid_moves_map = {}
        for row in range(self.board_size):
            for col in range(self.board_size):
                valid = []
                for dr, dc in self.knight_moves:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        valid.append((nr, nc))
                self.valid_moves_map[(row, col)] = valid

    def get_valid_moves(self, row: int, col: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Optimized: Use pre-computed moves"""
        return [pos for pos in self.valid_moves_map[(row, col)] if pos not in visited]

    def create_random_path(self, start_row: int, start_col: int) -> Tuple[Tuple[int, int], ...]:
        """Generate random path - returns tuple for hashability"""
        path = [(start_row, start_col)]
        visited = {(start_row, start_col)}
        current = (start_row, start_col)

        while len(path) < self.total_squares:
            available = self.get_valid_moves(current[0], current[1], visited)
            if not available:
                break

            current = random.choice(available)
            path.append(current)
            visited.add(current)

        return tuple(path)

    def calculate_fitness(self, path: Tuple[Tuple[int, int], ...]) -> float:
        """Optimized fitness calculation with caching"""
        # Check cache first
        if path in self.fitness_cache:
            return self.fitness_cache[path]

        if not path:
            return 0.0

        path_len = len(path)
        unique_count = len(set(path))

        # Fast path for complete tours
        if unique_count == self.total_squares and path_len == self.total_squares:
            # Quick validity check
            valid = all(
                abs(path[i][0] - path[i + 1][0]) in (1, 2) and
                abs(path[i][1] - path[i + 1][1]) in (1, 2) and
                abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1]) == 3
                for i in range(path_len - 1)
            )
            if valid:
                self.fitness_cache[path] = 2.0
                return 2.0

        # Regular fitness calculation
        coverage = unique_count / self.total_squares

        # Optimized validity check
        valid_moves = sum(
            1 for i in range(path_len - 1)
            if abs(path[i][0] - path[i + 1][0]) in (1, 2) and
            abs(path[i][1] - path[i + 1][1]) in (1, 2) and
            abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1]) == 3
        )

        validity = valid_moves / max(1, path_len - 1) if path_len > 1 else 1.0
        continuity = path_len / self.total_squares

        fitness = coverage * 0.5 + validity * 0.3 + continuity * 0.2

        if unique_count == self.total_squares and validity == 1.0:
            fitness += 1.0

        # Cache result
        self.fitness_cache[path] = fitness
        return fitness

    def improved_crossover(self, parent1: Tuple, parent2: Tuple) -> Tuple:
        """Improved crossover with better path repair"""
        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1 if len(parent1) > len(parent2) else parent2

        # Try multiple crossover points
        best_child = None
        best_len = 0

        for _ in range(2):  # Try 2 random crossover points
            cut = random.randint(1, min(len(parent1), len(parent2)) - 1)

            child = list(parent1[:cut])
            visited = set(child)

            # Add from parent2
            for pos in parent2:
                if pos not in visited and len(child) < self.total_squares:
                    if not child or self._is_valid_move(child[-1], pos):
                        child.append(pos)
                        visited.add(pos)

            # Try to extend with random moves
            if len(child) < self.total_squares:
                child = self._extend_path(child, visited)

            if len(child) > best_len:
                best_len = len(child)
                best_child = child

        return tuple(best_child if best_child else parent1)

    def _is_valid_move(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Fast knight move validation"""
        dr, dc = abs(pos2[0] - pos1[0]), abs(pos2[1] - pos1[1])
        return (dr == 1 and dc == 2) or (dr == 2 and dc == 1)

    def _extend_path(self, path: List, visited: Set, max_steps: int = 20) -> List:
        """Try to extend path with random moves"""
        path = path.copy()
        visited = visited.copy()
        current = path[-1]

        for _ in range(max_steps):
            if len(path) >= self.total_squares:
                break

            available = self.get_valid_moves(current[0], current[1], visited)
            if not available:
                break

            current = random.choice(available)
            path.append(current)
            visited.add(current)

        return path

    def smart_mutation(self, individual: Tuple) -> Tuple:
        """Improved mutation with adaptive strategies"""
        if len(individual) <= 2:
            return individual

        mutation_type = random.random()
        path = list(individual)

        if mutation_type < 0.5:
            # Truncate and regrow (most effective)
            cut = random.randint(max(1, len(path) // 3), len(path) - 1)
            path = path[:cut]
            path = self._extend_path(path, set(path))

        elif mutation_type < 0.8:
            # Segment reversal
            if len(path) > 4:
                start = random.randint(1, len(path) - 3)
                end = random.randint(start + 2, len(path))
                path = path[:start] + path[start:end][::-1] + path[end:]
        else:
            # Smart restart from good position
            if len(path) > 5:
                # Find a position with many available moves
                restart = random.randint(2, len(path) - 2)
                path = path[:restart]
                path = self._extend_path(path, set(path))

        return tuple(path)

    def update_belief_space(self, population: List[Tuple]):
        """Optimized belief space update"""
        # Evaluate population
        scored = [(ind, self.calculate_fitness(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update best solutions (elite)
        for ind, fit in scored[:self.elite_count]:
            self.cultural_knowledge['best_solutions'].append((ind, fit))

        # Keep top 15 solutions
        self.cultural_knowledge['best_solutions'].sort(key=lambda x: x[1], reverse=True)
        self.cultural_knowledge['best_solutions'] = self.cultural_knowledge['best_solutions'][:15]

        # Update position frequency (lightweight)
        for ind, fit in scored[:5]:
            if fit > 0.6:
                for pos in ind:
                    self.cultural_knowledge['position_frequency'][pos] = \
                        self.cultural_knowledge['position_frequency'].get(pos, 0) + 1

        # Prune frequency map
        if len(self.cultural_knowledge['position_frequency']) > 80:
            items = sorted(
                self.cultural_knowledge['position_frequency'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.cultural_knowledge['position_frequency'] = dict(items[:80])

    def apply_cultural_influence(self, individual: Tuple) -> Tuple:
        """Apply cultural knowledge"""
        if not self.cultural_knowledge['best_solutions']:
            return individual

        elite, _ = random.choice(self.cultural_knowledge['best_solutions'][:5])
        return self.improved_crossover(individual, elite)

    def tournament_select(self, population: List[Tuple], k: int = 3) -> Tuple:
        """Tournament selection"""
        competitors = random.sample(population, min(k, len(population)))
        return max(competitors, key=self.calculate_fitness)

    def solve(self, start_row: int, start_col: int, verbose: bool = False):
        """Main evolutionary loop"""
        # Initialize population
        population = [self.create_random_path(start_row, start_col)
                      for _ in range(self.population_size)]

        best_ever = None
        best_fitness = 0.0
        evaluations = 0
        stagnation = 0

        for gen in range(self.max_generations):
            # Evaluate
            scored = [(ind, self.calculate_fitness(ind)) for ind in population]
            evaluations += len(population)
            scored.sort(key=lambda x: x[1], reverse=True)

            current_best, current_fitness = scored[0]

            # Track best
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_ever = current_best
                stagnation = 0
            else:
                stagnation += 1

            # Update belief space
            self.update_belief_space(population)

            # Solution found
            if current_fitness >= 1.5:
                if verbose:
                    print(f"\n✓ Solution found at generation {gen}!")
                    print(f"Fitness: {current_fitness:.4f}")
                    print(f"Path: {len(current_best)}/{self.total_squares}")
                    print(f"Evaluations: {evaluations}")

                board = [[-1] * self.board_size for _ in range(self.board_size)]
                for step, (r, c) in enumerate(current_best):
                    board[r][c] = step
                return True, board, evaluations

            if verbose and gen % 100 == 0:
                print(f"Gen {gen}: Fitness={current_fitness:.4f}, "
                      f"Len={len(current_best)}/{self.total_squares}, "
                      f"Cache={len(self.fitness_cache)}")

            # Adaptive mutation rate
            if stagnation > 50:
                self.mutation_probability = min(0.5, self.mutation_probability + 0.05)
            elif stagnation < 10:
                self.mutation_probability = max(0.2, self.mutation_probability - 0.02)

            # Generate new population
            new_pop = [ind for ind, _ in scored[:self.elite_count]]

            while len(new_pop) < self.population_size:
                p1 = self.tournament_select(population)
                p2 = self.tournament_select(population)

                child = self.improved_crossover(p1, p2)

                if random.random() < self.mutation_probability:
                    child = self.smart_mutation(child)

                if random.random() < self.cultural_influence_probability:
                    child = self.apply_cultural_influence(child)

                new_pop.append(child)

            population = new_pop

            # Clear cache periodically
            if gen % 200 == 0:
                self.fitness_cache.clear()

        if verbose:
            print(f"\nBest fitness: {best_fitness:.4f}")
            print(f"Path: {len(best_ever)}/{self.total_squares}")

        board = [[-1] * self.board_size for _ in range(self.board_size)]
        if best_ever:
            for step, (r, c) in enumerate(best_ever):
                board[r][c] = step

        return best_fitness >= 1.5, board, evaluations

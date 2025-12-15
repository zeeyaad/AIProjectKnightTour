# # All 8 Possible Knight Moves On Chessboard (x, y) Like (i = Row, j = Col)
# dx = [2, 1, -1, -2, -2, -1, 1, 2] # Row
# dy = [1, 2, 2, 1, -1, -2, -2, -1] # Col
#
# class PureBacktracking:
#     def __init__(self, n):
#         self.n = n # Constructor Store Size
#
#     def isSafe(self, x, y, board):
#         # Used to Ensure (x, y) Is inside the Board, board[x][y] == -1 Mean This Square is not Visited Yet
#         return 0 <= x < self.n and 0 <= y < self.n and board[x][y] == -1 # Return True if Legal Is Unvisited and Inside, Otherwise False
#
#     def solve(self, start_x, start_y):
#         # Create Board As List Of List "Matrix" With Size N Each Cell Initialized By -1 This Mean UnVised
#         board = [[-1 for _ in range(self.n)] for _ in range(self.n)]
#         board[start_x][start_y] = 0 # Visited With Move Index 0 "1st Move"
#         attempts = [0]
#
#         def dfs(x, y, step):
#             attempts[0] += 1
#             if step == self.n * self.n:
#                 return True
#
#             for i in range(8):
#                 nx = x + dx[i]
#                 ny = y + dy[i]
#
#                 if self.isSafe(nx, ny, board):
#                     board[nx][ny] = step
#
#                     if dfs(nx, ny, step + 1):
#                         return True
#
#                     board[nx][ny] = -1  # backtrack
#
#             return False
#
#         success = dfs(start_x, start_y, 1)
#         return success, board, attempts[0]


# All 8 Possible Knight Moves On Chessboard (x, y) Like (i = Row, j = Col)
dx = [2, 1, -1, -2, -2, -1, 1, 2]  # Row
dy = [1, 2, 2, 1, -1, -2, -2, -1]  # Col


class PureBacktracking:
    def __init__(self, board_size):
        self.board_size = board_size

    def is_valid_unvisited_square(self, row, col, board):
        """Check if the square is inside the board and hasn't been visited yet"""
        return (0 <= row < self.board_size and
                0 <= col < self.board_size and
                board[row][col] == -1)

    def solve(self, start_row, start_col): # find_knight_tour
        """Find a complete knight's tour starting from the given position"""
        # Create board matrix with all cells initialized to -1 (unvisited)
        board = [[-1 for _ in range(self.board_size)] for _ in range(self.board_size)]
        board[start_row][start_col] = 0  # Mark starting position with move 0
        attempts = [0]  # Track number of recursive calls

        def explore_knight_moves(current_row, current_col, move_number):
            """Recursively explore all possible knight moves using backtracking"""
            attempts[0] += 1

            # Base case: all squares visited successfully
            if move_number == self.board_size * self.board_size:
                return True

            # Try all 8 possible knight moves
            for direction in range(8):
                """ If All Directions Were Tested This Mean None Of Them Is Right Move and in this state 'DeadEnd' """
                next_row = current_row + dx[direction]
                next_col = current_col + dy[direction]

                if self.is_valid_unvisited_square(next_row, next_col, board):
                    """Loop Until Getting Valid Move And if This False This Not Exit The Loop It Just Continue A Valid Sol. Finish All 8 Directions"""
                    board[next_row][next_col] = move_number  # Mark as visited

                    if explore_knight_moves(next_row, next_col, move_number + 1):
                        """ IF This Return True This Mean You Found a Complete Solution 'No Backtracking After this Points' """
                        return True

                    board[next_row][next_col] = -1  # Backtrack: unmark square

            return False

        tour_found = explore_knight_moves(start_row, start_col, 1)
        return tour_found, board, attempts[0]
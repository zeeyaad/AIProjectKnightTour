class KnightTourSolver:
    def __init__(self, n):
        self.n = n
        self.moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]

    def is_valid(self, x, y, board):
        return 0 <= x < self.n and 0 <= y < self.n and board[x][y] == -1

    def get_moves(self, x, y, board):
        res = []
        for dx, dy in self.moves:
            nx, ny = x+dx, y+dy
            if self.is_valid(nx, ny, board):
                res.append((nx, ny))
        return res

    def warn_order(self, x, y, board):
        moves = self.get_moves(x, y, board)
        moves.sort(key=lambda m: len(self.get_moves(m[0], m[1], board)))
        return moves


class BacktrackingSolver(KnightTourSolver):
    def solve(self, x0, y0):
        board = [[-1]*self.n for _ in range(self.n)]
        board[x0][y0] = 0
        attempts = [0]

        def dfs(x, y, step):
            attempts[0] += 1
            if step == self.n*self.n:
                return True

            for nx, ny in self.warn_order(x, y, board):
                board[nx][ny] = step
                if dfs(nx, ny, step+1):
                    return True
                board[nx][ny] = -1
            return False

        ok = dfs(x0, y0, 1)
        return ok, board, attempts[0]

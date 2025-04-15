import copy
from constants import Constants


class Board:
    """盤面の状態と操作を管理するクラス"""

    def __init__(self):
        """盤面の初期化"""
        self.reset()

    def reset(self):
        """盤面を初期状態にリセット"""
        self.cells = [
            [None] * Constants.BOARD_SIZE for _ in range(Constants.BOARD_SIZE)
        ]
        mid = Constants.BOARD_SIZE // 2
        self.cells[mid - 1][mid - 1] = Constants.WHITE
        self.cells[mid - 1][mid] = Constants.BLACK
        self.cells[mid][mid - 1] = Constants.BLACK
        self.cells[mid][mid] = Constants.WHITE

    def get_cell(self, x, y):
        """(x,y)の石の色を取得"""
        if 0 <= x < Constants.BOARD_SIZE and 0 <= y < Constants.BOARD_SIZE:
            return self.cells[x][y]
        return None

    def set_cell(self, x, y, color):
        """(x,y)に色をセット"""
        if 0 <= x < Constants.BOARD_SIZE and 0 <= y < Constants.BOARD_SIZE:
            self.cells[x][y] = color

    def copy(self):
        """盤面のディープコピーを返す"""
        board_copy = Board()
        board_copy.cells = copy.deepcopy(self.cells)
        return board_copy

    def count_stones(self):
        """黒石と白石の数をカウント"""
        black_count = sum(row.count(Constants.BLACK) for row in self.cells)
        white_count = sum(row.count(Constants.WHITE) for row in self.cells)
        return black_count, white_count

    def __getitem__(self, key):
        """配列形式でのアクセスをサポート（board[x]の形式）"""
        return self.cells[key]
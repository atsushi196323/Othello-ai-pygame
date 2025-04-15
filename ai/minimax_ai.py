import math
import threading
from constants import Constants
from ai.ai_strategy import AIStrategy


class MinimaxAI(AIStrategy):
    """ミニマックスアルゴリズムを使用するAI"""

    def __init__(self, game_logic, depth=3):
        """ミニマックスAIの初期化"""
        super().__init__(game_logic)
        self.game_logic = game_logic  # 明示的に設定
        self.depth = depth
        self.thinking = False  # thinkingフラグを追加

    def evaluate_board(self, board):
        """盤面の評価値を算出する（ホワイト視点）"""
        black_count, white_count = self.game_logic.count_stones(board)

        # 角の評価
        corners = [
            (0, 0),
            (0, Constants.BOARD_SIZE - 1),
            (Constants.BOARD_SIZE - 1, 0),
            (Constants.BOARD_SIZE - 1, Constants.BOARD_SIZE - 1),
        ]
        white_corners = sum(1 for x, y in corners if board[x][y] == Constants.WHITE)
        black_corners = sum(1 for x, y in corners if board[x][y] == Constants.BLACK)

        # 辺の評価
        edges = []
        for i in range(1, Constants.BOARD_SIZE - 1):
            edges.extend(
                [
                    (0, i),
                    (Constants.BOARD_SIZE - 1, i),
                    (i, 0),
                    (i, Constants.BOARD_SIZE - 1),
                ]
            )
        white_edges = sum(1 for x, y in edges if board[x][y] == Constants.WHITE)
        black_edges = sum(1 for x, y in edges if board[x][y] == Constants.BLACK)

        # 安定石（角と同じ）
        white_stable = white_corners
        black_stable = black_corners

        # モビリティ
        white_moves = len(self.game_logic.get_valid_moves(Constants.WHITE, board))
        black_moves = len(self.game_logic.get_valid_moves(Constants.BLACK, board))
        mobility = white_moves - black_moves

        return (
            (white_count - black_count)
            + (white_corners * 10 - black_corners * 10)
            + (white_edges * 2 - black_edges * 2)
            + (white_stable * 5 - black_stable * 5)
            + mobility
        )

    def is_terminal_board(self, board):
        """盤面が終局状態かどうかを判定する"""
        return not self.game_logic.has_valid_move(
            Constants.BLACK, board
        ) and not self.game_logic.has_valid_move(Constants.WHITE, board)

    def minimax(self, board, depth, maximizing_player, alpha=-math.inf, beta=math.inf):
        """ミニマックス探索（α-βカットオフ付き）で盤面の評価値を求める"""
        # 深さが0になったか終局状態なら評価値を返す
        if depth == 0 or self.is_terminal_board(board):
            return self.evaluate_board(board)

        if maximizing_player:
            max_eval = -math.inf
            for move in self.game_logic.get_valid_moves(Constants.WHITE, board):
                new_board = self.game_logic.make_move_for_board(
                    board, move[0], move[1], Constants.WHITE
                )
                eval_value = self.minimax(new_board, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in self.game_logic.get_valid_moves(Constants.BLACK, board):
                new_board = self.game_logic.make_move_for_board(
                    board, move[0], move[1], Constants.BLACK
                )
                eval_value = self.minimax(new_board, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            return min_eval

    def get_move(self):
        """ミニマックスで最適手を選択"""
        valid_moves = self.game_logic.get_valid_moves(Constants.WHITE)
        if not valid_moves:
            return None

        best_score = -math.inf
        move_selected = None

        for move in valid_moves:
            new_board = self.game_logic.make_move_for_board(
                self.game_logic.state.board.cells, move[0], move[1], Constants.WHITE
            )
            score = self.minimax(new_board, self.depth - 1, False)
            if score > best_score:
                best_score = score
                move_selected = move

        return move_selected

    def think_and_move(self, board, player, time_limit):
        """AIが思考して手を選ぶ処理"""
        try:
            # 引数なしでget_moveを呼び出す
            move = self.get_move()
            if move is not None:
                # game_logicの形式に合わせて手を変換して適用
                grid_x, grid_y = move
                self.game_logic.place_stone(grid_x, grid_y)
            else:
                # 有効な手がない場合（パス）
                self.game_logic.pass_turn()
        except Exception as e:
            print(f"AIの思考中にエラーが発生しました: {e}")
        finally:
            self.thinking = False

    def start_thinking(self):
        """AIの思考プロセスを開始する"""
        self.thinking = True

        # ゲームロジックから現在のボード状態とプレイヤーを取得
        board = self.game_logic.state.board
        player = Constants.WHITE  # AIは白石として定義されている

        # 適切な思考時間に基づいて手を選択
        difficulty = getattr(self.game_logic.state, "difficulty", 1)  # 安全に取得
        time_limit = 1
        if difficulty == 2:
            time_limit = 3
        elif difficulty == 3:
            time_limit = 8

        # 別スレッドで思考処理を実行し、UIをブロックしないように
        thinking_thread = threading.Thread(
            target=self.think_and_move, args=(board, player, time_limit)
        )
        thinking_thread.daemon = True  # メインプログラム終了時にスレッドも終了
        thinking_thread.start()
import random
import time
import math
import threading
from constants import Constants
from typing import List, Tuple, Dict, Optional, Set
from ai.ai_strategy import AIStrategy
from board import Board

# 定数の定義 - AIロジック内部で使用
AI_BLACK = 1
AI_WHITE = -1
AI_EMPTY = 0
DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

# ボードの評価マトリックス（略）
POSITION_WEIGHTS = [
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120],
]

# その他の定数（略）
STABILITY_WEIGHT = 10
MOBILITY_WEIGHT = 5
PARITY_WEIGHT = 2
DEFAULT_DEPTH = 6
ENDGAME_THRESHOLD = 12
# トランスポジションテーブルをインスタンス変数に変更（メモリ効率化）
OPENING_BOOK = {
    "................OX......XO................": [(2, 4), (3, 5), (4, 2), (5, 3)],
    "................OX......XO.......O...X....": [(2, 2), (2, 3), (2, 5)],
    "................OX......XO....X..O........": [(2, 4), (4, 2), (4, 6)],
}

# コーナー位置を定数として定義
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]


class WorldAI(AIStrategy):
    def __init__(self, game_logic, board_size=8):
        super().__init__(game_logic)
        self.board_size = board_size
        self.game_logic = game_logic
        self.max_time = 10
        self.start_time = 0
        self.time_limit_reached = False
        self.nodes_expanded = 0
        self.cutoffs = 0
        self.show_thinking_indicator = True
        self.thinking = False
        self.difficulty = 2
        # インスタンス変数として初期化（メモリ効率のため）
        self.transposition_table = {}
        # 方向ごとの有効性を事前計算するキャッシュ
        self.valid_cache = {}
        # 位置評価のキャッシュ
        self.position_score_cache = {}

    # 定数変換ヘルパーメソッド
    def _convert_to_ai_player(self, game_player):
        """ゲームの定数をAI内部の定数に変換"""
        if game_player == Constants.BLACK:
            return AI_BLACK
        elif game_player == Constants.WHITE:
            return AI_WHITE
        return None

    def _convert_to_game_player(self, ai_player):
        """AI内部の定数をゲームの定数に変換"""
        if ai_player == AI_BLACK:
            return Constants.BLACK
        elif ai_player == AI_WHITE:
            return Constants.WHITE
        return None

    def _convert_board(self, game_board):
        """ゲームボードをAI用の2次元配列に変換"""
        ai_board = [
            [AI_EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)
        ]

        # Board オブジェクトの場合
        if isinstance(game_board, Board):
            for i in range(self.board_size):
                for j in range(self.board_size):
                    cell = game_board.get_cell(i, j)
                    if cell == Constants.BLACK:
                        ai_board[i][j] = AI_BLACK
                    elif cell == Constants.WHITE:
                        ai_board[i][j] = AI_WHITE
            return ai_board

        # すでに2次元配列の場合（AIが内部計算用に作成した配列）
        if hasattr(game_board, "__getitem__"):
            try:
                # すでにAI形式かチェック
                if game_board[0][0] in [AI_BLACK, AI_WHITE, AI_EMPTY, None]:
                    return game_board

                # 変換が必要な場合
                for i in range(self.board_size):
                    for j in range(self.board_size):
                        cell = game_board[i][j]
                        if cell == Constants.BLACK:
                            ai_board[i][j] = AI_BLACK
                        elif cell == Constants.WHITE:
                            ai_board[i][j] = AI_WHITE
            except:
                pass

        return ai_board

    def start_thinking(self):
        """AIの思考プロセスを開始する"""
        self.thinking = True

        # ゲームロジックから現在のボード状態とプレイヤーを取得
        game_board = self.game_logic.state.board
        game_player = Constants.WHITE  # AIは白石として定義されている

        # AI内部表現に変換
        board = self._convert_board(game_board)
        player = self._convert_to_ai_player(game_player)

        # 思考時間設定
        time_limit = 1
        if self.difficulty == 2:
            time_limit = 3
        elif self.difficulty == 3:
            time_limit = 8

        # 別スレッドで思考処理を実行し、UIをブロックしないように
        def think_and_move(board, player, time_limit):
            try:
                move = self.get_move(board, player, time_limit)
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

        # 思考スレッドを開始
        thinking_thread = threading.Thread(
            target=think_and_move, args=(board, player, time_limit)
        )
        thinking_thread.daemon = True  # メインプログラム終了時にスレッドも終了
        thinking_thread.start()

    def get_move(
        self, board: List[List[int]], player: int, time_limit: int = 10
    ) -> Tuple[int, int]:
        """
        現在のボード状態とプレイヤーに基づいて最適な手を選択する
        """
        # 思考開始フラグをセット
        self.thinking = True

        self.max_time = time_limit
        self.start_time = time.time()
        self.time_limit_reached = False
        self.nodes_expanded = 0
        # 探索ごとにキャッシュをクリア
        self.valid_cache = {}
        self.position_score_cache = {}

        # 有効な手をすべて取得
        valid_moves = self.get_valid_moves(board, player)

        if not valid_moves:
            self.thinking = False  # 思考終了
            return None

        # 1手しかない場合はすぐに返す（高速化）
        if len(valid_moves) == 1:
            self.thinking = False
            return valid_moves[0]

        # 残りの空きマスを数える
        empty_count = sum(row.count(AI_EMPTY) for row in board)

        # オープニングブックをチェック（ゲーム開始時）
        if empty_count > 50:
            book_move = self.get_opening_book_move(board, valid_moves)
            if book_move:
                self.thinking = False  # 思考終了
                return book_move

        # コーナーが取れる場合はすぐに取る（高速化）
        for move in valid_moves:
            if move in CORNERS:
                self.thinking = False
                return move

        # エンドゲームに近づいたら完全解析を試みる
        if empty_count <= ENDGAME_THRESHOLD:
            move = self.endgame_solver(board, player, valid_moves)
            self.thinking = False  # 思考終了
            return move

        # 通常の探索（反復深化）
        best_move = random.choice(valid_moves)  # デフォルト
        current_depth = 2

        # 反復深化で探索し、時間内に可能な限り深く探索する
        while current_depth <= DEFAULT_DEPTH and not self.is_time_up():
            best_score = float("-inf")
            best_move = None
            alpha = float("-inf")
            beta = float("inf")

            # 各可能な手を評価（ムーブオーダリングで最適化）
            ordered_moves = self.order_moves(board, valid_moves, player)

            for move in ordered_moves:
                # 盤面を仮想的に更新
                new_board = self.make_move(board, move, player)

                # α-β法によるミニマックス探索
                opponent = AI_WHITE if player == AI_BLACK else AI_BLACK
                score = self.minimax(
                    new_board, current_depth - 1, alpha, beta, opponent, False
                )

                if self.is_time_up():
                    # 時間切れなら前の深さでの最善手を使用
                    break

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, best_score)

            if not self.is_time_up():
                # この深さでの探索が正常に完了した場合、結果を更新
                current_depth += 1
            else:
                # 時間切れなら探索を終了
                break

        # 思考終了フラグをリセット
        self.thinking = False
        return best_move

    def minimax(
        self,
        board: List[List[int]],
        depth: int,
        alpha: float,
        beta: float,
        player: int,
        maximizing_player: bool,
    ) -> float:
        """
        α-β枝刈りを使用したミニマックスアルゴリズム
        """
        self.nodes_expanded += 1

        # 時間切れチェック（定期的に実行）
        if self.nodes_expanded % 1000 == 0 and self.is_time_up():
            self.time_limit_reached = True
            return 0

        # 終端ノードの評価
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(
                board, AI_BLACK if maximizing_player else AI_WHITE
            )

        # トランスポジションテーブルをチェック
        board_hash = self.hash_board(board)
        if (
            board_hash in self.transposition_table
            and self.transposition_table[board_hash]["depth"] >= depth
        ):
            # 保存された結果を使用
            entry = self.transposition_table[board_hash]
            if entry["flag"] == "exact":
                return entry["value"]
            elif entry["flag"] == "lower" and entry["value"] > alpha:
                alpha = entry["value"]
            elif entry["flag"] == "upper" and entry["value"] < beta:
                beta = entry["value"]

            if alpha >= beta:
                self.cutoffs += 1
                return entry["value"]

        # 有効な手を取得
        valid_moves = self.get_valid_moves(board, player)

        if not valid_moves:
            # パスの場合、相手のターンで再帰
            opponent = AI_WHITE if player == AI_BLACK else AI_BLACK
            return self.minimax(
                board, depth - 1, alpha, beta, opponent, not maximizing_player
            )

        # ムーブオーダリング（より良い手を先に評価）
        ordered_moves = self.order_moves(board, valid_moves, player)

        value = float("-inf") if maximizing_player else float("inf")
        best_val = value
        flag = "exact"  # トランスポジションテーブル用のフラグ
        opponent = AI_WHITE if player == AI_BLACK else AI_BLACK

        for move in ordered_moves:
            new_board = self.make_move(board, move, player)

            if maximizing_player:
                value = max(
                    value,
                    self.minimax(new_board, depth - 1, alpha, beta, opponent, False),
                )
                if value > beta:
                    flag = "lower"  # 下限カット
                    self.cutoffs += 1
                    break
                alpha = max(alpha, value)
            else:
                value = min(
                    value,
                    self.minimax(new_board, depth - 1, alpha, beta, opponent, True),
                )
                if value < alpha:
                    flag = "upper"  # 上限カット
                    self.cutoffs += 1
                    break
                beta = min(beta, value)

        # 結果をトランスポジションテーブルに保存
        self.transposition_table[board_hash] = {
            "value": value,
            "depth": depth,
            "flag": flag,
        }

        return value

    def evaluate_board(self, board: List[List[int]], player: int) -> float:
        """
        ボードの状態を評価する総合的な関数（最適化版）
        """
        # ゲーム終了チェック
        if self.is_game_over(board):
            black_count = sum(row.count(AI_BLACK) for row in board)
            white_count = sum(row.count(AI_WHITE) for row in board)

            if black_count > white_count:
                return 10000 if player == AI_BLACK else -10000
            elif white_count > black_count:
                return 10000 if player == AI_WHITE else -10000
            else:
                return 0

        # 盤面の評価を複数の要素から計算
        score = 0
        opponent = AI_WHITE if player == AI_BLACK else AI_BLACK

        # 1. 駒の配置評価（位置の重み）- キャッシュ利用
        board_hash = self.hash_board(board)
        if board_hash in self.position_score_cache:
            position_score = self.position_score_cache[board_hash]
        else:
            position_score = 0
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == player:
                        position_score += POSITION_WEIGHTS[i][j]
                    elif board[i][j] == opponent:
                        position_score -= POSITION_WEIGHTS[i][j]
            self.position_score_cache[board_hash] = position_score

        score += position_score

        # 2. モビリティ評価（着手可能箇所）
        player_mobility = len(self.get_valid_moves(board, player))
        opponent_mobility = len(self.get_valid_moves(board, opponent))

        if player_mobility + opponent_mobility != 0:
            mobility_score = (
                100
                * (player_mobility - opponent_mobility)
                / (player_mobility + opponent_mobility)
            )
            score += MOBILITY_WEIGHT * mobility_score

        # 3. 簡略化した安定石の評価 (高速化)
        player_stability = self.calculate_stability_fast(board, player)
        opponent_stability = self.calculate_stability_fast(board, opponent)
        score += STABILITY_WEIGHT * (player_stability - opponent_stability)

        # 4. パリティ評価（最後に打つ権利）
        empty_count = sum(row.count(AI_EMPTY) for row in board)
        if empty_count % 2 == 0:
            # 偶数なら現在のプレイヤーが最後に打つ
            score += PARITY_WEIGHT
        else:
            # 奇数なら相手が最後に打つ
            score -= PARITY_WEIGHT

        # 5. フロンティア評価（相手の着手可能箇所を増やす石を減らす）
        # - 計算コストが高いため、中盤以降のみ計算
        if empty_count < 40:
            player_frontier = self.count_frontier_discs(board, player)
            opponent_frontier = self.count_frontier_discs(board, opponent)
            score -= player_frontier - opponent_frontier

        # 6. コーナーに特別なボーナスを与える
        for row, col in CORNERS:
            if board[row][col] == player:
                score += 200  # コーナーボーナスを大きくする
            elif board[row][col] == opponent:
                score -= 200  # 相手のコーナーにはペナルティ

        return score

    def calculate_stability_fast(self, board: List[List[int]], player: int) -> int:
        """
        安定石（もう二度とひっくり返らない石）の数を計算する高速版
        完全な計算の代わりに近似値を返す
        """
        stable_count = 0

        # 1. コーナーは常に安定
        for corner in CORNERS:
            r, c = corner
            if board[r][c] == player:
                stable_count += 1

                # コーナーに隣接する同じ色の石も比較的安定
                for dr, dc in (
                    [(0, 1), (1, 0), (1, 1)]
                    if (r, c) == (0, 0)
                    else (
                        [(0, -1), (1, 0), (1, -1)]
                        if (r, c) == (0, 7)
                        else (
                            [(-1, 0), (0, 1), (-1, 1)]
                            if (r, c) == (7, 0)
                            else [(-1, 0), (0, -1), (-1, -1)]
                        )
                    )
                ):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and board[nr][nc] == player:
                        stable_count += 0.7  # 完全に安定ではないので部分的に加算

        # 2. 端の石も比較的安定
        for i in range(1, 7):
            # 上端
            if board[0][i] == player:
                stable_count += 0.5
            # 下端
            if board[7][i] == player:
                stable_count += 0.5
            # 左端
            if board[i][0] == player:
                stable_count += 0.5
            # 右端
            if board[i][7] == player:
                stable_count += 0.5

        return stable_count

    def count_frontier_discs(self, board: List[List[int]], player: int) -> int:
        """
        フロンティア（空きマスに隣接している石）の数を数える
        """
        frontier = 0

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == player:
                    # 隣接する空きマスをチェック
                    for dr, dc in DIRECTIONS:
                        r, c = i + dr, j + dc
                        if (
                            0 <= r < self.board_size
                            and 0 <= c < self.board_size
                            and board[r][c] == AI_EMPTY
                        ):
                            frontier += 1
                            break

        return frontier

    def endgame_solver(
        self, board: List[List[int]], player: int, valid_moves: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        エンドゲームソルバー: 残りの手数が少ない場合に最適解を探索
        """
        best_move = None
        best_score = float("-inf")
        opponent = AI_WHITE if player == AI_BLACK else AI_BLACK

        # 時間制限を考慮した探索深さの調整
        remaining_moves = sum(row.count(AI_EMPTY) for row in board)
        max_depth = min(remaining_moves, 12)  # 探索深さを制限（高速化）

        # より深い探索で完全解析を試みる
        for move in valid_moves:
            new_board = self.make_move(board, move, player)

            # エンドゲームでは制限された深さで探索
            score = self.minimax(
                new_board, max_depth, float("-inf"), float("inf"), opponent, False
            )

            if self.is_time_up():
                # 時間切れなら通常の評価に戻る
                return self.get_move(board, player, self.max_time)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def order_moves(
        self, board: List[List[int]], moves: List[Tuple[int, int]], player: int
    ) -> List[Tuple[int, int]]:
        """
        ムーブオーダリング: 最も有望な手を先に評価するために並べ替え
        """
        move_scores = []

        # コーナーを最優先（高速化）
        for move in moves:
            if move in CORNERS:
                return [move] + [m for m in moves if m != move]

        for move in moves:
            # 簡易評価でスコア計算
            score = self.simple_evaluate_move(board, move, player)
            move_scores.append((move, score))

        # スコアの高い順に並べ替え
        move_scores.sort(key=lambda x: x[1], reverse=True)

        return [move for move, _ in move_scores]

    def simple_evaluate_move(
        self, board: List[List[int]], move: Tuple[int, int], player: int
    ) -> int:
        """
        手の簡易評価（ムーブオーダリング用）
        実際にボードを作らず高速に評価
        """
        r, c = move

        # 位置の重みを基本スコアとする
        score = POSITION_WEIGHTS[r][c] * 2

        # コーナー近くの位置は危険なことが多い
        if (r, c) in [(0, 1), (1, 0), (1, 1)] and (0, 0) not in CORNERS:
            score -= 100
        elif (r, c) in [(0, 6), (1, 7), (1, 6)] and (0, 7) not in CORNERS:
            score -= 100
        elif (r, c) in [(6, 0), (7, 1), (6, 1)] and (7, 0) not in CORNERS:
            score -= 100
        elif (r, c) in [(6, 7), (7, 6), (6, 6)] and (7, 7) not in CORNERS:
            score -= 100

        # ひっくり返す数が多いほど良い（簡易計算）
        flip_count = 0
        for dr, dc in DIRECTIONS:
            line_flips = 0
            curr_r, curr_c = r + dr, c + dc
            # ボード上にあり、相手の石がある間ループ
            while (
                0 <= curr_r < 8
                and 0 <= curr_c < 8
                and board[curr_r][curr_c]
                == (AI_WHITE if player == AI_BLACK else AI_BLACK)
            ):
                line_flips += 1
                curr_r += dr
                curr_c += dc

                # ボード外または空白に到達した場合、このラインのflipは無効
                if (
                    not (0 <= curr_r < 8 and 0 <= curr_c < 8)
                    or board[curr_r][curr_c] == AI_EMPTY
                ):
                    line_flips = 0
                    break

                # 自分の石に到達した場合、このラインのflipは有効
                if board[curr_r][curr_c] == player:
                    flip_count += line_flips
                    break

        # ひっくり返す数も評価に加える
        score += flip_count * 5

        return score

    def get_opening_book_move(
        self, board: List[List[int]], valid_moves: List[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        """
        オープニングブックから手を選択
        """
        board_str = "".join(
            [
                "X" if cell == AI_BLACK else "O" if cell == AI_WHITE else "."
                for row in board
                for cell in row
            ]
        )

        if board_str in OPENING_BOOK:
            book_moves = OPENING_BOOK[board_str]
            # オープニングブックの手が有効な手なら返す
            for move in book_moves:
                if move in valid_moves:
                    return move

        return None

    def hash_board(self, board: List[List[int]]) -> str:
        """
        ボードをハッシュ文字列に変換（トランスポジションテーブル用）
        """
        return "".join(str(cell) for row in board for cell in row)

    def is_time_up(self) -> bool:
        """
        思考時間が制限を超えているかチェック
        """
        return time.time() - self.start_time > self.max_time

    def is_game_over(self, board: List[List[int]]) -> bool:
        """
        ゲームが終了しているかチェック
        """
        # 両プレイヤーが打てるかチェック
        return not self.get_valid_moves(board, AI_BLACK) and not self.get_valid_moves(
            board, AI_WHITE
        )

    def get_valid_moves(self, board, player):
        """
        指定したプレイヤーの有効な手をすべて取得
        """
        # キャッシュをチェック（高速化）
        board_hash = self.hash_board(board)
        cache_key = f"{board_hash}_{player}"
        if cache_key in self.valid_cache:
            return self.valid_cache[cache_key]

        # Boardクラスのインスタンスの場合はgame_logicを使用
        if isinstance(board, Board):
            game_player = self._convert_to_game_player(player)
            valid_moves = self.game_logic.get_valid_moves(game_player)
            self.valid_cache[cache_key] = valid_moves
            return valid_moves

        # boardが適切な型かチェック
        if not hasattr(board, "__getitem__"):
            raise TypeError(
                "Board must be a subscriptable object (like a list or array)"
            )

        valid_moves = []

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == AI_EMPTY and self.is_valid_move(
                    board, (i, j), player
                ):
                    valid_moves.append((i, j))

        # 結果をキャッシュに保存
        self.valid_cache[cache_key] = valid_moves
        return valid_moves

    def is_valid_move(
        self, board: List[List[int]], move: Tuple[int, int], player: int
    ) -> bool:
        """
        指定した手が有効かチェック
        """
        if board[move[0]][move[1]] != AI_EMPTY:
            return False

        opponent = AI_WHITE if player == AI_BLACK else AI_BLACK

        # すべての方向をチェック
        for dr, dc in DIRECTIONS:
            r, c = move[0] + dr, move[1] + dc
            # 最初のステップが盤内かつ相手の石である必要がある
            if (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and board[r][c] == opponent
            ):
                # 相手の石を見つけた
                r += dr
                c += dc
                # 盤内である間ループ
                while 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if board[r][c] == AI_EMPTY:
                        # 空きマスがあるとダメ
                        break
                    if board[r][c] == player:
                        # 自分の石で挟めればOK
                        return True
                    r += dr
                    c += dc

        return False

    def make_move(
        self, board: List[List[int]], move: Tuple[int, int], player: int
    ) -> List[List[int]]:
        """
        指定した手を適用した新しいボードを生成
        """
        opponent = AI_WHITE if player == AI_BLACK else AI_BLACK

        if move is None:
            # パスの場合は盤面をそのまま返す
            return [row[:] for row in board]

        # ボードのコピーを作成
        new_board = [row[:] for row in board]

        # 石を置く
        new_board[move[0]][move[1]] = player

        # ひっくり返す処理
        for dr, dc in DIRECTIONS:
            r, c = move[0] + dr, move[1] + dc
            to_flip = []

            # 相手の石を見つけたらto_flipに追加
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and new_board[r][c] == opponent
            ):
                to_flip.append((r, c))
                r += dr
                c += dc

                # 盤外に出たら反転なし
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    to_flip = []
                    break

                # 空きマスがあったら反転なし
                if new_board[r][c] == AI_EMPTY:
                    to_flip = []
                    break

                # 自分の石で挟めたら反転確定
                if new_board[r][c] == player:
                    break

            # 挟まれた石をひっくり返す
            for flip_r, flip_c in to_flip:
                new_board[flip_r][flip_c] = player

        return new_board


# Pygameとの連携用インターフェース
def get_ai_move(board, player, difficulty=3):
    """
    Pygameから呼び出す関数
    """
    # 注意: このインターフェース関数はgame_logicがないので機能しません
    # 適切に実装するにはgame_logicを渡す必要があります
    ai = WorldAI(None)  # ここでエラーが発生します

    # 難易度に応じて思考時間を調整
    time_limit = 1
    if difficulty == 2:
        time_limit = 3
    elif difficulty == 3:
        time_limit = 8

    return ai.get_move(board, player, time_limit)
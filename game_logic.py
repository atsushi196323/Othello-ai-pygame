import copy
import pygame
from constants import Constants
from game_state import GameState
from board import Board


class GameLogic:
    """オセロのゲームロジックを管理するクラス"""

    # GameLogicクラスに以下のメソッドを追加
    def get_current_board(self):
        # 既存のボード取得メソッドまたはプロパティを使用
        return self.state  # または適切な方法でボード状態を返す

    def __init__(self):
        """ゲームロジックの初期化"""
        self.state = GameState()

    def is_valid_position(self, x, y):
        """(x, y)が有効な盤面座標かどうかを確認する"""
        return 0 <= x < Constants.BOARD_SIZE and 0 <= y < Constants.BOARD_SIZE

    def is_valid_move(self, x, y, color=None, board=None):
        """指定の盤面上で、(x,y) に color の石を置けるか判定する"""
        if not self.is_valid_position(x, y):
            return False

        if color is None:
            color = self.state.turn
        if board is None:
            board = self.state.board.cells

        # 既に石が置かれている場合は無効
        if board[x][y] is not None:
            return False

        opponent = Constants.WHITE if color == Constants.BLACK else Constants.BLACK

        # 全方向を探索
        for dx, dy in Constants.DIRECTIONS:
            nx, ny = x + dx, y + dy
            has_opponent_between = False

            # この方向に対して、反転できる石があるか確認
            while self.is_valid_position(nx, ny):
                if board[nx][ny] == opponent:
                    has_opponent_between = True
                elif board[nx][ny] == color:
                    if has_opponent_between:
                        return True
                    break
                else:
                    break
                nx += dx
                ny += dy

        return False

    def get_stones_to_flip(self, x, y, color=None, board=None):
        """(x,y) に石を置いた際、ひっくり返すべき石の座標リストを返す"""
        if not self.is_valid_position(x, y):
            return []

        if color is None:
            color = self.state.turn
        if board is None:
            board = self.state.board.cells

        stones_to_flip = []
        opponent = Constants.WHITE if color == Constants.BLACK else Constants.BLACK

        for dx, dy in Constants.DIRECTIONS:
            nx, ny = x + dx, y + dy
            temp_flips = []

            while self.is_valid_position(nx, ny):
                if board[nx][ny] == opponent:
                    temp_flips.append((nx, ny))
                elif board[nx][ny] == color:
                    if temp_flips:  # 少なくとも1つの相手の石がある場合
                        stones_to_flip.extend(temp_flips)
                    break
                else:
                    break
                nx += dx
                ny += dy

        return stones_to_flip

    def place_stone(self, x, y):
        """石を配置し、アニメーションキューを作成する"""
        # 既にゲームが終了している場合や無効な手の場合は何もしない
        if self.state.game_over or not self.is_valid_move(x, y):
            return False

        # アニメーション中は操作を受け付けない
        if self.state.is_animating or self.state.paused:
            return False

        # 移動履歴に追加
        self.state.move_history.append((x, y, self.state.turn))

        # 石を配置
        self.state.board.set_cell(x, y, self.state.turn)
        stones_to_flip = self.get_stones_to_flip(x, y)

        # 配置する石のアニメーションを追加
        self.state.animation_queue.append(
            {
                "type": "place",
                "position": (x, y),
                "color": self.state.turn,
                "progress": 0,
            }
        )

        # 反転する石のアニメーションを追加
        for fx, fy in stones_to_flip:
            self.state.animation_queue.append(
                {
                    "type": "flip",
                    "position": (fx, fy),
                    "from_color": (
                        Constants.WHITE
                        if self.state.turn == Constants.BLACK
                        else Constants.BLACK
                    ),
                    "to_color": self.state.turn,
                    "progress": 0,
                }
            )

        # ターン交代
        self.state.switch_turn()
        self.state.is_animating = True
        self.state.pass_occurred = False

        return True

    def get_valid_moves(self, color=None, board=None):
        """指定の盤面上で、color の着手可能な手のリストを返す"""
        if color is None:
            color = self.state.turn
        if board is None:
            board = self.state.board.cells

        valid_moves = []
        for x in range(Constants.BOARD_SIZE):
            for y in range(Constants.BOARD_SIZE):
                if self.is_valid_move(x, y, color, board):
                    valid_moves.append((x, y))

        return valid_moves

    def has_valid_move(self, color=None, board=None):
        """現在の盤面で、color の有効な着手が存在するかを判定する"""
        if color is None:
            color = self.state.turn
        if board is None:
            board = self.state.board.cells

        for x in range(Constants.BOARD_SIZE):
            for y in range(Constants.BOARD_SIZE):
                if board[x][y] is None and self.is_valid_move(x, y, color, board):
                    return True
        return False

    def make_move_for_board(self, board, x, y, color):
        """与えられた盤面のコピーに対して、(x,y) に color の石を置き反転処理を行い新盤面を返す"""
        if not self.is_valid_position(x, y) or not self.is_valid_move(
            x, y, color, board
        ):
            return copy.deepcopy(board)  # 無効な手の場合は盤面をそのままコピーして返す

        new_board = copy.deepcopy(board)
        new_board[x][y] = color

        stones_to_flip = self.get_stones_to_flip(x, y, color, board)
        for fx, fy in stones_to_flip:
            new_board[fx][fy] = color

        return new_board

    def count_stones(self, board=None):
        """盤面上の黒石と白石の数をカウントして返す"""
        if board is None:
            return self.state.board.count_stones()

        black_count = sum(row.count(Constants.BLACK) for row in board)
        white_count = sum(row.count(Constants.WHITE) for row in board)

        return black_count, white_count

    def game_result(self):
        """ゲーム終了時の勝者または引き分けを返す"""
        black_count, white_count = self.count_stones()

        if black_count > white_count:
            return "Winner: Black"
        elif white_count > black_count:
            return "Winner: White"
        else:
            return "Draw"

    def is_game_over(self):
        """ゲームが終了したかどうかを判定する"""
        return not self.has_valid_move(Constants.BLACK) and not self.has_valid_move(
            Constants.WHITE
        )

    def update_animation(self):
        """アニメーションの更新とゲーム状態の管理"""
        # デルタタイムを計算（前回フレームからの経過秒数）
        delta_time = self.state.calculate_delta_time()

        if self.state.animation_queue:
            # アニメーションの進行（フレームレート非依存）
            animation = self.state.animation_queue[0]
            animation["progress"] += (
                Constants.ANIMATION_SPEED * delta_time * 60
            )  # 60FPSを基準に調整

            if animation["progress"] >= 1.0:
                if animation["type"] == "flip":
                    x, y = animation["position"]
                    self.state.board.set_cell(x, y, animation["to_color"])
                self.state.animation_queue.pop(0)

            self.state.is_animating = bool(self.state.animation_queue)

        else:
            self.state.is_animating = False

            # ゲーム終了チェック
            if self.is_game_over():
                self.state.game_over = True
                return

            # 一時停止中は処理しない
            if self.state.paused:
                return

            # 現在のプレイヤーが着手できない場合、パスする
            if not self.has_valid_move():
                opponent = (
                    Constants.WHITE
                    if self.state.turn == Constants.BLACK
                    else Constants.BLACK
                )
                if self.has_valid_move(opponent):
                    # パスの表示
                    if self.state.turn == Constants.BLACK:
                        self.state.set_message("黒がパスします")
                    else:
                        self.state.set_message("白がパスします")
                    self.state.pass_occurred = True
                    self.state.switch_turn()
                else:
                    self.state.game_over = True

    def toggle_pause(self):
        """ゲームの一時停止を切り替える"""
        self.state.paused = not self.state.paused
        message = (
            "ゲームを一時停止しました" if self.state.paused else "ゲームを再開しました"
        )
        self.state.set_message(message)

    def undo_move(self):
        """最後の手を元に戻す（デモ用）"""
        if not self.state.move_history or self.state.is_animating:
            return False

        # ゲームが終了していたら解除
        if self.state.game_over:
            self.state.game_over = False

        # 盤面を初期状態に戻す
        self.state.board.reset()

        # 最後の1手を除いて履歴を再生
        history_to_replay = self.state.move_history[:-1]
        self.state.move_history = []
        self.state.turn = Constants.BLACK  # 初期ターンに戻す

        # 履歴を再生
        for x, y, color in history_to_replay:
            self.state.turn = color  # 色を設定
            self.place_stone(x, y)

        return True

    def pass_turn(self):
        """現在のプレイヤーのターンをパスする"""
        # アニメーション中や一時停止中、またはゲーム終了時は処理しない
        if self.state.is_animating or self.state.paused or self.state.game_over:
            return False

        # 現在のプレイヤーが有効な手を持っている場合はパスできない
        if self.has_valid_move():
            return False

        # パスの表示
        if self.state.turn == Constants.BLACK:
            self.state.set_message("黒がパスします")
        else:
            self.state.set_message("白がパスします")

        self.state.pass_occurred = True
        self.state.switch_turn()

        #  次のプレイヤーも有効な手がない場合はゲーム終了
        if not self.has_valid_move():
            self.state.game_over = True

        return True

import threading
import time
from constants import Constants
from game_logic import GameLogic


class AIStrategy:
    """AIの抽象基底クラス"""

    def __init__(self, game_logic):
        """AIの初期化"""
        self.game_logic = game_logic
        self.thinking = False
        self.show_thinking_indicator = False
        self.thinking_indicator_time = 0
        self.thread = None  # スレッド参照を保持

    def get_move(self, board, player):
        """次の一手を返す（サブクラスでオーバーライド）"""
        raise NotImplementedError

    def start_thinking(self):
        """AIの思考を別スレッドで開始する"""
        if (
            not self.thinking
            and not self.game_logic.state.is_animating
            and not self.game_logic.state.game_over
        ):

            # 前のスレッドが存在し、まだ実行中なら終了を待つ
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=0.1)

            self.thread = threading.Thread(target=self.think)
            self.thread.daemon = True
            self.thread.start()

    def think(self):
        """AIの思考プロセス（スレッドで実行）"""
        if self.game_logic.state.game_over:
            return

        # 思考中フラグを設定
        self.thinking = True
        self.show_thinking_indicator = True

        try:
            # ゲーム状態を取得
            game_state = self.game_logic.get_current_board()
            self.board = game_state.board  # 実際のボード配列を取得
            self.player = self.game_logic.state.turn

            # AI思考の少し遅延を入れる（UI応答性のため）
            time.sleep(0.5)

            # 実際の着手処理
            move = self.get_move(self.board, self.player)
            if move:
                self.game_logic.place_stone(*move)
            elif not self.game_logic.has_valid_move(self.player):
                # 現在のプレイヤー（AI）が有効な手を持っていない場合はパス
                self.game_logic.state.switch_turn()
                self.game_logic.state.set_message("AIがパスしました")
                self.game_logic.state.pass_occurred = True
            else:
                # エラー：AIは手を見つけられなかったが、有効な手はあるはず
                # この状態は通常発生しないはず
                self.game_logic.state.set_message("AIエラー: 有効な手が見つかりません")
        finally:
            # 例外が発生しても確実に思考終了状態にする
            self.thinking = False
            self.show_thinking_indicator = False
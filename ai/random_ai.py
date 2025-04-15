import random
import threading
from constants import Constants
from ai.ai_strategy import AIStrategy


class RandomAI(AIStrategy):
    """ランダムに手を選ぶAI"""

    def __init__(self, game_logic):
        super().__init__(game_logic)
        self.game_logic = game_logic
        self.show_thinking_indicator = True
        self.thinking = False
        self.difficulty = 1  # デフォルト難易度を追加

    def get_move(self):
        """有効な手からランダムに選択"""
        valid_moves = self.game_logic.get_valid_moves(Constants.WHITE)
        if valid_moves:
            return random.choice(valid_moves)
        return None

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
        difficulty = self.difficulty
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
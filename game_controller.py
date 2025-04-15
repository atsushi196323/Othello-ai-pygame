import pygame
import sys
import math
import threading
from constants import Constants
from game_logic import GameLogic
from renderer import Renderer
from ai.random_ai import RandomAI
from ai.minimax_ai import MinimaxAI
from ai.stronger_ai import StrongerAI
from ai.world_class_ai import WorldAI


class GameController:
    """ゲームの実行と制御を管理するクラス"""

    def __init__(self, ai_type=Constants.AI_TYPE_MINIMAX, screen=None):
        """ゲームコントローラの初期化"""
        self.game_logic = GameLogic()
        self.ai = self.create_ai(ai_type)
        self.screen = screen
        self.renderer = Renderer(self.screen, self.game_logic, self.ai)

    def create_ai(self, ai_type):
        """指定されたタイプのAIを作成"""
        if ai_type == Constants.AI_TYPE_RANDOM:
            return RandomAI(self.game_logic)
        elif ai_type == Constants.AI_TYPE_MINIMAX:
            return MinimaxAI(self.game_logic)
        elif ai_type == Constants.AI_TYPE_STRONGER:
            return StrongerAI(self.game_logic)
        elif ai_type == Constants.AI_TYPE_WORLD:
            return WorldAI(self.game_logic)
        else:  # デフォルトはminimax
            return MinimaxAI(self.game_logic)

    def handle_event(self, event):
        """イベント処理"""
        if event.type == pygame.QUIT:
            return False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if (
                self.game_logic.state.is_player_turn()
                and not self.game_logic.state.is_animating
                and not self.game_logic.state.game_over
                and not self.game_logic.state.paused
            ):
                x, y = event.pos
                if y < Constants.SIZE:  # ボード内をクリックした場合
                    grid_x = x // Constants.GRID_SIZE
                    grid_y = y // Constants.GRID_SIZE
                    if self.game_logic.is_valid_move(grid_x, grid_y, Constants.BLACK):
                        self.game_logic.place_stone(grid_x, grid_y)
                    else:
                        # 打てない場所をクリックした場合のフィードバック
                        self.game_logic.state.set_message("そこには置けません")

        elif event.type == pygame.KEYDOWN:
            # 'u'キーで手を戻す（デバッグ用）
            if event.key == pygame.K_u:
                self.game_logic.undo_move()
            # スペースキーでゲームの一時停止/再開
            elif event.key == pygame.K_SPACE:
                self.game_logic.toggle_pause()
            # ESCキーでゲーム終了
            elif event.key == pygame.K_ESCAPE:
                return False

        return True

    def update(self):
        """ゲーム状態の更新"""
        self.game_logic.update_animation()

        # アニメーション中でなくAIの手番ならAIに着手を依頼
        if (
            not self.game_logic.state.is_animating
            and not self.game_logic.state.game_over
            and not self.game_logic.state.is_player_turn()
            and not self.game_logic.state.paused
            and not self.ai.thinking
        ):
            self.ai.start_thinking()

    def run(self):
        """ゲームループの実行"""
        clock = pygame.time.Clock()
        running = True

        while running:
            clock.tick(60)  # 60FPSを目標に

            # イベント処理
            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
                    break

            # ゲーム状態の更新
            self.update()

            # 描画
            self.renderer.draw_board()
            self.renderer.draw_valid_moves()
            self.renderer.draw_animations()
            pygame.display.flip()

        # ゲーム終了時の処理
        if self.game_logic.state.game_over:
            result = self.game_logic.game_result()
            self.animate_end(result)

    def animate_end(self, message):
        """ゲーム終了時の結果表示アニメーション"""
        clock = pygame.time.Clock()
        duration = 3000  # 3秒間のアニメーション
        start_time = pygame.time.get_ticks()

        while True:
            # イベント処理（ウィンドウを閉じる操作とキー入力に対応）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        pygame.quit()
                        sys.exit()

            elapsed = pygame.time.get_ticks() - start_time
            if elapsed >= duration:
                break

            scale = 1 + 0.1 * math.sin(elapsed / 200 * math.pi)
            font_size = int(74 * scale)
            font = pygame.font.Font(None, font_size)

            if message == "CLEAR":
                text_color = Constants.YELLOW
            elif message == "GAME OVER":
                text_color = (255, 0, 0)
            else:
                text_color = Constants.WHITE

            text = font.render(message, True, text_color)
            text_rect = text.get_rect(center=(Constants.SIZE // 2, Constants.SIZE // 2))

            # 少し小さな文字でプレイヤーに終了方法を知らせる
            small_font = pygame.font.Font(None, 30)
            exit_text = small_font.render(
                "Enterキーまたはエスケープキーで終了", True, Constants.WHITE
            )
            exit_rect = exit_text.get_rect(
                center=(Constants.SIZE // 2, Constants.SIZE // 2 + 50)
            )

            self.screen.fill(Constants.BLACK)
            self.screen.blit(text, text_rect)
            self.screen.blit(exit_text, exit_rect)
            pygame.display.flip()
            clock.tick(60)

        pygame.time.wait(500)  # 500msの待機
        pygame.quit()
        sys.exit()

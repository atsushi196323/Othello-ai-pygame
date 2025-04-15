import math
import pygame
from constants import Constants


class Renderer:
    """画面描画処理を管理するクラス"""

    def lighten_color(self, color, amount=50):
        """色を明るくする"""
        r = min(255, color[0] + amount)
        g = min(255, color[1] + amount)
        b = min(255, color[2] + amount)
        return (r, g, b)

    def darken_color(self, color, amount=50):
        """色を暗くする"""
        r = max(0, color[0] - amount)
        g = max(0, color[1] - amount)
        b = max(0, color[2] - amount)
        return (r, g, b)

    def __init__(self, screen, game_logic, ai=None):
        """描画処理の初期化"""
        self.screen = screen
        self.game_logic = game_logic
        self.ai = ai
        # フォントの初期化
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def draw_board(self):
        """盤面を描画"""
        self.screen.fill(Constants.GREEN)
        self._draw_grid()
        self._draw_stones()
        self.draw_scores()
        self.draw_message()
        if self.ai and self.ai.show_thinking_indicator:
            self.draw_thinking_indicator()

        if self.game_logic.state.paused:
            self.draw_pause_overlay()

    def _draw_grid(self):
        """盤面のグリッドを描画"""
        for x in range(Constants.BOARD_SIZE):
            for y in range(Constants.BOARD_SIZE):
                rect = pygame.Rect(
                    x * Constants.GRID_SIZE,
                    y * Constants.GRID_SIZE,
                    Constants.GRID_SIZE,
                    Constants.GRID_SIZE,
                )
                pygame.draw.rect(self.screen, Constants.BLACK, rect, 1)

    def _draw_stones(self):
        """盤面上の全ての石を描画"""
        for x in range(Constants.BOARD_SIZE):
            for y in range(Constants.BOARD_SIZE):
                cell = self.game_logic.state.board.cells[x][y]
                if cell is not None:
                    self.draw_stone(x, y, cell)

    def draw_stone(self, x, y, color, progress=1.0, flipping=False, flip_progress=0.0):
        """石を描画（アニメーション対応）"""
        if progress <= 0.0:
            return
        center = (
            x * Constants.GRID_SIZE + Constants.GRID_SIZE // 2,
            y * Constants.GRID_SIZE + Constants.GRID_SIZE // 2,
        )

        radius = Constants.GRID_SIZE // 2 - 4

        if flipping:
            # ひっくり返るアニメーションの場合、石が回転しているように見せる
            # 0→0.5: 横に縮む、0.5→1.0: 横に広がる
            flip_stage = (
                flip_progress * 2 if flip_progress < 0.5 else (1.0 - flip_progress) * 2
            )

            # 縦と横の半径を計算（楕円形にする）
            horizontal_radius = int(
                radius * (0.2 + 0.8 * flip_stage)
            )  # 最小20%まで縮小
            vertical_radius = radius

            # 楕円を描画
            rect = pygame.Rect(
                center[0] - horizontal_radius,
                center[1] - vertical_radius,
                horizontal_radius * 2,
                vertical_radius * 2,
            )
            pygame.draw.ellipse(self.screen, color, rect)

            # 立体感を出すためのハイライトと影
            highlight_color = self.lighten_color(color, 50)
            shadow_color = self.darken_color(color, 50)

            # ハイライト（右上部分）
            if flip_progress < 0.5:
                pygame.draw.arc(
                    self.screen,
                    highlight_color,
                    rect,
                    math.pi * 1.25,
                    math.pi * 1.75,
                    max(1, int(horizontal_radius * 0.2)),
                )
            else:
                # 裏返った後は反対側にハイライト
                pygame.draw.arc(
                    self.screen,
                    highlight_color,
                    rect,
                    math.pi * 0.25,
                    math.pi * 0.75,
                    max(1, int(horizontal_radius * 0.2)),
                )

            # 影（左下部分）
            if flip_progress < 0.5:
                pygame.draw.arc(
                    self.screen,
                    shadow_color,
                    rect,
                    math.pi * 0.25,
                    math.pi * 0.75,
                    max(1, int(horizontal_radius * 0.2)),
                )
            else:
                # 裏返った後は反対側に影
                pygame.draw.arc(
                    self.screen,
                    shadow_color,
                    rect,
                    math.pi * 1.25,
                    math.pi * 1.75,
                    max(1, int(horizontal_radius * 0.2)),
                )

        elif progress >= 1.0:
            # 通常の石を描画
            pygame.draw.circle(self.screen, color, center, radius)

            # 立体感を出すためのハイライトと影
            highlight_color = self.lighten_color(color, 50)
            shadow_color = self.darken_color(color, 50)

            # ハイライト（右上部分）
            pygame.draw.arc(
                self.screen,
                highlight_color,
                pygame.Rect(
                    center[0] - radius, center[1] - radius, radius * 2, radius * 2
                ),
                math.pi * 1.25,
                math.pi * 1.75,
                max(1, int(radius * 0.15)),
            )

            # 影（左下部分）
            pygame.draw.arc(
                self.screen,
                shadow_color,
                pygame.Rect(
                    center[0] - radius, center[1] - radius, radius * 2, radius * 2
                ),
                math.pi * 0.25,
                math.pi * 0.75,
                max(1, int(radius * 0.15)),
            )

        else:
            current_radius = int(radius * progress)
            if current_radius > 0:
                # アルファブレンディングを使用するためのSurface
                stone_surf = pygame.Surface(
                    (Constants.GRID_SIZE, Constants.GRID_SIZE), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    stone_surf,
                    color + (int(255 * progress),),  # RGBAカラー
                    (Constants.GRID_SIZE // 2, Constants.GRID_SIZE // 2),
                    current_radius,
                )
                self.screen.blit(
                    stone_surf,
                    (x * Constants.GRID_SIZE, y * Constants.GRID_SIZE),
                )

    def draw_valid_moves(self):
        """有効な手を描画"""
        if (
            self.game_logic.state.turn == Constants.BLACK
            and not self.game_logic.state.game_over
            and not self.game_logic.state.paused
        ):
            for x, y in self.game_logic.get_valid_moves(Constants.BLACK):
                center = (
                    x * Constants.GRID_SIZE + Constants.GRID_SIZE // 2,
                    y * Constants.GRID_SIZE + Constants.GRID_SIZE // 2,
                )
                pygame.draw.circle(self.screen, Constants.YELLOW, center, 5)

    def draw_scores(self):
        """スコアを描画"""
        black_count, white_count = self.game_logic.count_stones()

        # ゲーム終了時は結果も表示
        if self.game_logic.state.game_over:
            result = self.game_logic.game_result()
            score_text = f"黒: {black_count}  白: {white_count}  {result}"
        else:
            turn_text = "黒" if self.game_logic.state.turn == Constants.BLACK else "白"
            score_text = f"黒: {black_count}  白: {white_count}  手番: {turn_text}"

        text = self.font.render(score_text, True, Constants.YELLOW)
        self.screen.blit(text, (10, Constants.SIZE + 10))

        # パスが発生した場合、フラグを表示
        if self.game_logic.state.pass_occurred and not self.game_logic.state.game_over:
            pass_text = self.small_font.render("（パス発生中）", True, Constants.RED)
            self.screen.blit(pass_text, (Constants.SIZE - 100, Constants.SIZE + 15))

    def draw_animations(self):
        """アニメーションを描画"""
        for animation in self.game_logic.state.animation_queue:
            if animation["type"] == "place":
                x, y = animation["position"]
                color = animation["color"]
                progress = animation["progress"]
                self.draw_stone(x, y, color, progress)
            elif animation["type"] == "flip":
                x, y = animation["position"]
                from_color = animation["from_color"]
                to_color = animation["to_color"]
                progress = animation["progress"]

                # 進行度0.5を境に色を切り替え
                if progress < 0.5:
                    current_color = from_color
                else:
                    current_color = to_color

                # ひっくり返るアニメーション効果で描画
                self.draw_stone(x, y, current_color, 1.0, True, progress)

                # 中間色を計算（アニメーション進行度に応じて色を変化）
                intermediate_color = (
                    int(from_color[0] + (to_color[0] - from_color[0]) * progress),
                    int(from_color[1] + (to_color[1] - from_color[1]) * progress),
                    int(from_color[2] + (to_color[2] - from_color[2]) * progress),
                )
                self.draw_stone(x, y, intermediate_color, progress)

    def draw_message(self):
        """一時的なメッセージを表示"""
        if self.game_logic.state.message:
            message_surf = self.font.render(
                self.game_logic.state.message, True, Constants.YELLOW
            )
            message_rect = message_surf.get_rect(
                center=(Constants.SIZE // 2, Constants.SIZE // 2)
            )

            # 背景を追加して読みやすく
            bg_rect = message_rect.inflate(20, 10)
            bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 180))  # 半透明の黒

            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(message_surf, message_rect)

            # メッセージの表示時間を更新
            self.game_logic.state.update_message()

    def draw_thinking_indicator(self):
        """AIの思考中インジケータを描画"""
        if (
            self.ai.show_thinking_indicator
            and self.game_logic.state.turn == Constants.WHITE
        ):
            # 思考インジケータのアニメーション
            self.ai.thinking_indicator_time += 1
            dots = "." * (self.ai.thinking_indicator_time // 10 % 4)

            thinking_text = self.font.render(
                f"AIが考え中{dots}", True, Constants.LIGHT_BLUE
            )
            thinking_rect = thinking_text.get_rect(center=(Constants.SIZE // 2, 30))

            # 背景を追加
            bg_rect = thinking_rect.inflate(20, 10)
            bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))  # 半透明の黒

            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(thinking_text, thinking_rect)

    def draw_pause_overlay(self):
        """一時停止中の半透明オーバーレイを描画"""
        overlay = pygame.Surface((Constants.SIZE, Constants.SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # 半透明の黒
        self.screen.blit(overlay, (0, 0))

        pause_text = self.font.render(
            "一時停止中 - スペースキーで再開", True, Constants.WHITE
        )
        text_rect = pause_text.get_rect(
            center=(Constants.SIZE // 2, Constants.SIZE // 2)
        )
        self.screen.blit(pause_text, text_rect)

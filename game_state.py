import pygame
from constants import Constants
from board import Board


class GameState:
    """ゲームの状態を管理するクラス"""

    def __init__(self):
        """ゲーム状態の初期化"""
        self.board = Board()
        self.turn = Constants.BLACK  # 初期ターンは黒
        self.game_over = False
        self.animation_queue = []
        self.is_animating = False
        self.message = None
        self.message_time = 0
        self.pass_occurred = False
        self.move_history = []
        self.paused = False
        self.last_frame_time = pygame.time.get_ticks()  # フレーム時間管理用

    def switch_turn(self):
        """ターンを交代"""
        self.turn = Constants.WHITE if self.turn == Constants.BLACK else Constants.BLACK

    def set_message(self, text):
        """一時的なメッセージをセット"""
        self.message = text
        self.message_time = pygame.time.get_ticks()

    def update_message(self):
        """メッセージの表示時間を管理"""
        if self.message:
            current_time = pygame.time.get_ticks()
            if current_time - self.message_time > Constants.MESSAGE_DURATION:
                self.message = None

    def is_player_turn(self):
        """現在のターンがプレイヤー(黒)かどうかを返す"""
        return self.turn == Constants.BLACK

    def calculate_delta_time(self):
        """前回フレームからの経過時間を計算（秒単位）"""
        current_time = pygame.time.get_ticks()
        delta_time = (
            current_time - self.last_frame_time
        ) / 1000.0  # ミリ秒から秒に変換
        self.last_frame_time = current_time
        return delta_time

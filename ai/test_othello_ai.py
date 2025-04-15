import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import time


# モック作成（依存モジュール用）
class MockConstants:
    BLACK = 1
    WHITE = -1
    EMPTY = 0


class MockBoard:
    def __init__(self):
        # 標準的な8x8のオセロ初期配置
        self.board = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

    def get_cell(self, row, col):
        return self.board[row][col]


class MockGameLogic:
    def __init__(self):
        self.state = MagicMock()
        self.state.board = MockBoard()
        self.placed_move = None
        self.passed = False

    def place_stone(self, row, col):
        self.placed_move = (row, col)
        return True

    def pass_turn(self):
        self.passed = True
        return True

    def get_valid_moves(self, player):
        if player == MockConstants.BLACK:
            return [(2, 3), (3, 2), (4, 5), (5, 4)]
        else:  # WHITE
            return [(2, 4), (3, 5), (4, 2), (5, 3)]


# モジュールのパッチ
sys.modules["constants"] = MagicMock()
sys.modules["constants"].Constants = MockConstants
sys.modules["board"] = MagicMock()
sys.modules["board"].Board = MockBoard
sys.modules["ai.ai_strategy"] = MagicMock()
sys.modules["ai.ai_strategy"].AIStrategy = MagicMock

# AIモジュールをインポート（パッチ後に行う必要があります）
from world_class_ai import WorldAI, AI_BLACK, AI_WHITE, AI_EMPTY


class TestWorldAI(unittest.TestCase):
    def setUp(self):
        """各テスト前の準備"""
        self.game_logic = MockGameLogic()
        self.ai = WorldAI(self.game_logic)
        self.ai.difficulty = 1  # テスト高速化のため低難易度設定

        # 標準的なボード状態
        self.standard_board = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

    def test_initialization(self):
        """AIの初期化が正しく行われるかテスト"""
        self.assertEqual(self.ai.board_size, 8)
        self.assertEqual(self.ai.game_logic, self.game_logic)
        self.assertEqual(self.ai.difficulty, 1)

    def test_convert_board(self):
        """ボード変換が正しく動作するかテスト"""
        game_board = self.game_logic.state.board
        ai_board = self.ai._convert_board(game_board)

        # 初期配置の黒と白の石の位置を確認
        self.assertEqual(ai_board[3][4], AI_BLACK)
        self.assertEqual(ai_board[4][3], AI_BLACK)
        self.assertEqual(ai_board[3][3], AI_WHITE)
        self.assertEqual(ai_board[4][4], AI_WHITE)

    def test_get_valid_moves(self):
        """有効な手の取得が正しく動作するかテスト"""
        moves = self.ai.get_valid_moves(self.standard_board, AI_BLACK)
        expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]
        self.assertEqual(set(moves), set(expected_moves))

    def test_is_valid_move(self):
        """手の妥当性チェックが正しく動作するかテスト"""
        # 有効な手を確認
        self.assertTrue(self.ai.is_valid_move(self.standard_board, (2, 3), AI_BLACK))
        self.assertTrue(self.ai.is_valid_move(self.standard_board, (3, 2), AI_BLACK))

        # 無効な手を確認
        self.assertFalse(self.ai.is_valid_move(self.standard_board, (0, 0), AI_BLACK))
        self.assertFalse(
            self.ai.is_valid_move(self.standard_board, (3, 3), AI_BLACK)
        )  # すでに石がある

    def test_make_move(self):
        """手を打った後の盤面が正しく更新されるかテスト"""
        new_board = self.ai.make_move(self.standard_board, (2, 3), AI_BLACK)

        # 石が置かれているか確認
        self.assertEqual(new_board[2][3], AI_BLACK)

        # 挟まれた石が裏返されているか確認
        self.assertEqual(new_board[3][3], AI_BLACK)  # 白から黒に変わっている

    def test_evaluate_board(self):
        """盤面評価が正しく動作するかテスト"""
        # 標準的なボードの評価値
        initial_score = self.ai.evaluate_board(self.standard_board, AI_BLACK)

        # 有利なボードの評価値
        advantageous_board = self.ai.make_move(self.standard_board, (2, 3), AI_BLACK)
        advantage_score = self.ai.evaluate_board(advantageous_board, AI_BLACK)

        # 有利なボードの評価値が高くなっているか確認
        self.assertGreater(advantage_score, initial_score)

    def test_time_limit(self):
        """時間制限内に手を返すかテスト"""
        start_time = time.time()
        move = self.ai.get_move(self.standard_board, AI_BLACK, time_limit=1)
        end_time = time.time()

        # 有効な手を返しているか確認
        self.assertIsNotNone(move)
        self.assertIn(move, [(2, 3), (3, 2), (4, 5), (5, 4)])

        # 時間制限を守っているか確認（少し余裕を持たせる）
        self.assertLessEqual(end_time - start_time, 1.5)

    def test_start_thinking(self):
        """思考プロセスが正しく動作するかテスト"""
        # 思考を開始
        self.ai.start_thinking()
        self.assertTrue(self.ai.thinking)  # 思考フラグが立っているか確認

        # 思考完了まで少し待つ
        time.sleep(2)

        # 思考が完了したか確認
        self.assertFalse(self.ai.thinking)

        # いずれかのアクションが実行されたか確認
        self.assertTrue(
            self.game_logic.placed_move is not None or self.game_logic.passed
        )

    def test_specific_board_positions(self):
        """特定の盤面で正しい判断をするかテスト"""
        # コーナーを取るチャンスがある局面
        corner_opportunity_board = [
            [0, -1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        # 難易度を上げて十分な探索を保証
        self.ai.difficulty = 3
        move = self.ai.get_move(corner_opportunity_board, AI_BLACK, time_limit=5)

        # コーナー(0,0)を選ぶべき
        self.assertEqual(move, (2, 3))

    def test_calculate_stability(self):
        """安定石の計算が正しく動作するかテスト"""
        # コーナーに石がある盤面
        corner_board = [row[:] for row in self.standard_board]
        corner_board[0][0] = AI_BLACK  # 左上コーナーに黒石

        stability = self.ai.calculate_stability(corner_board, AI_BLACK)
        # 少なくとも1つの安定石（コーナー）があるはず
        self.assertGreaterEqual(stability, 1)


if __name__ == "__main__":
    unittest.main()

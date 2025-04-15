class Constants:
    """ゲーム全体で使用する定数の定義"""

    # 色
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 128, 0)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    LIGHT_BLUE = (173, 216, 230)

    # ゲーム設定
    SIZE = 600
    BOARD_SIZE = 8
    GRID_SIZE = SIZE // BOARD_SIZE

    # 方向ベクトル (8方向)
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # アニメーション設定
    ANIMATION_SPEED = 0.1
    MESSAGE_DURATION = 2000  # ms

    # AIタイプ
    AI_TYPE_RANDOM = "random"
    AI_TYPE_MINIMAX = "minimax"
    AI_TYPE_STRONGER = "stronger"
    AI_TYPE_WORLD = "world"

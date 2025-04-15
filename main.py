import pygame
import asyncio
import sys
from constants import Constants
from game_controller import GameController


async def main_async():

    pygame.init()
    screen = pygame.display.set_mode((Constants.SIZE, Constants.SIZE + 50))
    pygame.display.set_caption("オセロゲーム")
    clock = pygame.time.Clock()

    # ゲームコントローラーの初期化
    game_controller = GameController(Constants.AI_TYPE_MEDIUM, screen)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            # イベント処理をコントローラーに委譲
            game_controller.handle_event(event)

        # ゲーム状態の更新
        game_controller.update()

        # 描画処理
        screen.fill((0, 128, 0))  # 緑の背景
        game_controller.draw()
        pygame.display.flip()

        # フレームレート制御とブラウザへの制御返却（重要）
        clock.tick(60)
        await asyncio.sleep(0)

    pygame.quit()


def main():
    """メイン関数"""
    # Pygameの初期化
    pygame.init()
    screen = pygame.display.set_mode(
        (Constants.SIZE, Constants.SIZE + 50)
    )  # スコア表示用に高さを追加
    pygame.display.set_caption("オセロゲーム")

    # AIレベルの選択
    # 初級 AI_TYPE_RANDOM = "random"
    # 中級 AI_TYPE_MINIMAX = "minimax"
    # 上級 AI_TYPE_STRONGER = "stronger"
    # 最上級 AI_TYPE_WORLD = "world"
    controller = GameController(Constants.AI_TYPE_WORLD, screen)
    controller.run()


if __name__ == "__main__":
    # Pygbag環境で実行する場合はasyncio.run(main_async())を使用
    # それ以外の環境では通常のmain()を使用
    import platform

    if platform.system() == "Emscripten":  # Pygbag環境の判定
        asyncio.run(main_async())
    else:
        main()

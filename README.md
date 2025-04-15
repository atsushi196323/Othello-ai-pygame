# Othello-ai-pygame

A Python-developed Pygame-based Othello (Reversi) game with AI opponent. You can enjoy playing against the computer AI.

## Features

- Intuitive and user-friendly graphical interface
- Multiple difficulty levels of AI opponents

## Technical Overview

### Architecture

Othello-ai-pygame consists of the following components:

- **Game Logic**: Implements Othello rules and manages valid moves and board states
- **AI Engine**: Decision-making system using MiniMax algorithm and alpha-beta pruning
- **Graphics Engine**: UI rendering and input processing using Pygame
- **Data Management**: Saving and loading game states

### AI Implementation

This game offers multiple AI levels:

1. **Beginner Level**: Random selection and basic strategy
2. **Intermediate Level**: MiniMax algorithm with depth 3
3. **Advanced Level**: MiniMax algorithm with depth 5 and alpha-beta pruning
4. **Expert Level**: MiniMax algorithm with depth 7, alpha-beta pruning, and opening book utilization

## Requirements

- Python 3.7 or higher
- Pygame 2.0.0 or higher
- NumPy (used for AI calculations)

## Installation

### Method: Clone from Git

```bash
# Clone the repository
git clone https://github.com/yourusername/Othello-ai-pygame.git

# Navigate to the project directory
cd Othello-ai-pygame

# Install required packages
pip install -r requirements.txt
```

## How to Run

After installation, launch the game with the following command:

```bash
python main.py
```

or

```bash
python3 main.py
```

## AI Details

The AI in Othello-ai-pygame is implemented using a combination of the following techniques:

1. **Evaluation Function**: Quantifies the board state to evaluate advantage
   - Number of pieces
   - Corner control
   - Mobility
   - Pattern recognition
2. **Search Algorithms**:
   - MiniMax algorithm: Searches for the optimal move
   - Alpha-beta pruning: Reduces unnecessary searches for efficiency
   - Depth-limited search: Limits calculation time
3. **Strategic Elements**:
   - Opening book: Common opening sequences
   - Endgame perfect analysis: Optimal moves in the endgame

## Future Improvements

- Online multiplayer functionality
- More advanced AI algorithms (integration of neural networks)
- Customizable themes and boards
- Mobile platform support

## License

This project is released under the MIT License. For details, please refer to the LICENSE file.

Bug reports and feature requests can be submitted through the GitHub Issues page.

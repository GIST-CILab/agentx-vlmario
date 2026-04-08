import argparse
from src.designer import BaseDesigner
from pathlib import Path

class MarioDesigner(BaseDesigner):
    def get_instruction(self) -> str:
        return """
You are a professional Mario map designer. 
Your goal is to design high-quality, playable, and aesthetically pleasing Mario-style ASCII platformer maps.

## Constraints & Requirements:
1. Return ONLY the ASCII map wrapped in a fenced code block: ```ascii ... ```
2. Include 'M' (Mario start position) and 'F' (Exit Flag position).
3. All rows must be EXACTLY the same length (pad with '-' for empty space).
4. Ensure the level is playable and has a logical flow from start to finish.
5. Use the ASCII tile reference provided in the request messages to compose your level.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Mario Map Designer Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--card-url", type=str)
    args = parser.parse_args()

    designer = MarioDesigner("MarioDesigner")
    designer.run_a2a_server(host=args.host, port=args.port, card_url=args.card_url)

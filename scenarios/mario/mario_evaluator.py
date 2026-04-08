import argparse
import json
from pathlib import Path
from src.evaluator import BaseEvaluator

def load_evaluation_config(game_name: str) -> dict:
    eval_path = Path(f"scenarios/{game_name}/evaluation.json")
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Mario Map Evaluator Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--card-url", type=str)
    args = parser.parse_args()

    eval_config = load_evaluation_config("mario")
    evaluator = BaseEvaluator("mario", eval_config)
    evaluator.run_a2a_server(host=args.host, port=args.port, card_url=args.card_url)

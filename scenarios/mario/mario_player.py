import argparse
import subprocess
from pathlib import Path
from typing import Optional
from src.player import BasePlayer

class MarioPlayer(BasePlayer):
    def play(self, text_map: str, output_dir: str, output_name: str, idx: int) -> Optional[str]:
        jar_path = Path("scenarios/mario/PlayAstar.jar")
        if not jar_path.exists():
            print(f"PlayAstar.jar not found at {jar_path.absolute()}")
            return None

        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        map_filename = output_name.replace("_video.mp4", "_map.txt")
        map_path = out_path / map_filename
        map_path.write_text(text_map, encoding="utf-8")

        video_file = out_path / output_name

        assets_path = Path("scenarios/mario/img/")
        assets_arg = str(assets_path.resolve()).rstrip("/\\") + "/"

        cmd = [
            "java", "-Djava.awt.headless=true", "-jar", jar_path.name,
            str(map_path), "human", assets_arg, str(out_path), output_name
        ]

        try:
            subprocess.run(cmd, check=True, timeout=120, cwd=jar_path.parent)
            if video_file.exists() and video_file.stat().st_size > 0:
                # Return the absolute or relative path to the generated map video
                return str(video_file.absolute())
        except Exception as e:
            print(f"PlayAstar execution failed: {e}")
            
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Mario Map Player Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9120)
    parser.add_argument("--card-url", type=str)
    args = parser.parse_args()

    player = MarioPlayer("MarioPlayer")
    player.run_a2a_server(host=args.host, port=args.port, card_url=args.card_url)

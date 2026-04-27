"""Game-specific modules (map loading + rendering + defaults).

Each module must expose:
    KEY: str                                    # CLI name (e.g. "mario")
    DEFAULTS: dict                              # maps_file, criteria_file, output_dir, profile_key
    add_arguments(parser: argparse.ArgumentParser) -> None      # optional
    load_maps(path: str) -> list[tuple[str, str]]
    normalize_map(raw: str) -> str
    render_map(map_text, map_id, output_dir, args) -> (map_path, video_path)
"""

from procedural_pipeline.games import mario, sokoban


GAMES = {
    mario.KEY: mario,
    sokoban.KEY: sokoban,
}

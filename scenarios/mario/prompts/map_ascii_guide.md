# Mario AI Framework Map ASCII Guide

This document explains the ASCII characters used to represent different elements in the Mario AI Framework level files.

## Level Boundaries

- `M`: **Mario Start** - The starting position of Mario. If not specified, defaults to (0, first floor).
- `F`: **Mario Exit** - The end point (flag) of the level. If not specified, defaults to (rightmost, first floor).
- `-` or space: **Empty** - Empty space (air).

## Terrain & Blocks

### Solid Blocks

- `X`: **Ground** - Solid ground block (floor).
- `#`: **Pyramid Block** - Used for stairs or decorative pyramid blocks.
- `S`: **Normal Brick** - Breakable brick block.
- `C`: **Coin Brick** - Brick block containing a coin (counts toward total coins).
- `L`: **1-Up Brick** - Brick block containing a 1-up mushroom.
- `U`: **Mushroom Brick** - Brick block containing a power-up mushroom.
- `D`: **Used Block** - A block that has already been hit (empty block).

### Platforms

- `%`: **Platform** - Jump-through platform (can jump up through it, blocks from above).
- `|`: **Platform Background** - Background decoration for jump-through platforms.

## Interactive Blocks

### Question Blocks

- `?` or `@`: **Mushroom Question Block** - Question block containing a power-up mushroom.
- `Q` or `!`: **Coin Question Block** - Question block containing a coin (counts toward total coins).

### Hidden Blocks

- `1`: **Invisible 1-Up Block** - Invisible block containing a 1-up mushroom (appears when hit from below).
- `2`: **Invisible Coin Block** - Invisible block containing a coin (counts toward total coins, appears when hit from below).

## Items

- `o`: **Coin** - Collectible coin floating in the air (counts toward total coins).

## Pipes

### Pipe Types

- `t`: **Empty Pipe** - Warp pipe without enemies. Automatically connects adjacent pipe parts.
- `T`: **Flower Pipe** - Pipe with a Piranha Plant (ENEMY_FLOWER) at the top.

### Pipe Parts (Manual Construction)

- `<`: **Pipe Top Left** - Left side of pipe opening.
- `>`: **Pipe Top Right** - Right side of pipe opening.
- `[`: **Pipe Body Left** - Left side of pipe body.
- `]`: **Pipe Body Right** - Right side of pipe body.

**Note**: Using `t` or `T` automatically handles pipe connections. Use `<`, `>`, `[`, `]` for manual pipe construction.

## Bullet Bill Cannons

- `*`: **Bullet Bill Cannon** - Creates a Bullet Bill cannon. Multiple `*` characters vertically create taller cannons (auto-detects height).
- `B`: **Bullet Bill Head** - Top of Bullet Bill cannon (tile index 3).
- `b`: **Bullet Bill Body** - Neck and body parts of Bullet Bill cannon (connects to head).

## Enemies

### Goombas

- `E` or `g`: **Goomba** - Basic ground enemy.
- `G`: **Goomba Winged** - Flying Goomba with wings.

### Koopa Troopas

- `k`: **Green Koopa** - Green Koopa Troopa (turns into shell when stomped).
- `K`: **Green Koopa Winged** - Flying Green Koopa Troopa.
- `r`: **Red Koopa** - Red Koopa Troopa (moves in a pattern).
- `R`: **Red Koopa Winged** - Flying Red Koopa Troopa.

### Spiny Enemies

- `y`: **Spiky** - Spiny enemy (cannot be stomped).
- `Y`: **Spiky Winged** - Flying Spiny enemy.

## Notes

- **Coin Counting**: Characters `o`, `C`, `Q`, `!`, and `2` all count toward the level's `totalCoins`.
- **Pipe Auto-Connection**: Using `t` or `T` automatically detects and connects adjacent pipe parts. The system checks for:
  - Single pipe (isolated) vs. multi-tile pipes
  - Connections to left/right pipe parts
  - Vertical connections to pipe parts above
- **Bullet Bill Height**: Multiple `*` characters stacked vertically automatically create taller cannons (up to 3 tiles high).
- **Platform Connections**: `%` characters automatically connect horizontally to form longer platforms.

"""
tile_generator.py — Equirectangular → Marzipano cubemap tile generator

Usage:
  py -3 tools/tile_generator.py --verify        # generate level 2 for comparison
  py -3 tools/tile_generator.py --level 3        # generate level 3 (2048px, 4×4 tiles)
  py -3 tools/tile_generator.py --level 3 --scene 0-reception-1  # single scene

Face naming matches Marzipano convention: f, b, l, r, u, d
Tile grid: row 0..N-1 top→bottom, col 0..N-1 left→right
"""

import argparse
import os
import math
import numpy as np
from PIL import Image

# ─── Source / output paths ────────────────────────────────────────────────────

SRC_DIR  = r'D:\Marsh\src'
TILE_DIR = r'D:\Design Tech\Claude\marsh-update\tiles'

SCENES = [
    '0-reception-1',
    '1-recruitment-zone-1',
    '2-recruitment-zone-2',
    '3-reception-2',
    '4-visitor-lounge-1',
    '5-visitor-lounge-2',
]

# Tile size is always 512px; face size = tile_size * tiles_per_side
TILE_SIZE = 512

LEVELS = {
    # level: (face_size, tiles_per_side)
    1: (256,  1),
    2: (1024, 2),
    3: (2048, 4),
}

# ─── Face direction mappings ──────────────────────────────────────────────────
# Each face maps (u, v) ∈ [-1,1]² to a 3D direction vector (before normalisation).
# Convention matches Marzipano / standard OpenGL cubemap:
#   +Z = front, -Z = back, +X = right, -X = left, +Y = up, -Y = down
# u increases left→right, v increases top→bottom in image space.

def face_to_xyz(face, u_grid, v_grid):
    """Return (x, y, z) arrays for the given face, unnormalised."""
    if face == 'f':   # front  (+Z)
        return  u_grid,  -v_grid,  np.ones_like(u_grid)
    elif face == 'b': # back   (-Z)
        return -u_grid,  -v_grid, -np.ones_like(u_grid)
    elif face == 'r': # right  (+X)
        return  np.ones_like(u_grid), -v_grid, -u_grid
    elif face == 'l': # left   (-X)
        return -np.ones_like(u_grid), -v_grid,  u_grid
    elif face == 'u': # up     (+Y)
        return  u_grid,  np.ones_like(u_grid),  v_grid
    elif face == 'd': # down   (-Y)
        return  u_grid, -np.ones_like(u_grid), -v_grid
    else:
        raise ValueError(f'Unknown face: {face}')


def equirect_to_face(equirect_img, face, face_size):
    """
    Sample an equirectangular image to produce a single cubemap face.
    Returns a PIL Image of size (face_size, face_size).
    """
    eq_w, eq_h = equirect_img.size
    eq = np.array(equirect_img)  # shape: (H, W, C)

    # Build a (face_size × face_size) grid of u,v ∈ [-1, 1]
    # Pixel centres: (0.5 / face_size) to (1 - 0.5/face_size) mapped to [-1, 1]
    lin = np.linspace(-1 + 1/face_size, 1 - 1/face_size, face_size)
    u_grid, v_grid = np.meshgrid(lin, lin)  # shape: (face_size, face_size)

    x, y, z = face_to_xyz(face, u_grid, v_grid)

    # Normalise
    r = np.sqrt(x*x + y*y + z*z)
    x, y, z = x/r, y/r, z/r

    # Convert to equirectangular pixel coordinates
    # yaw ∈ [-π, π]:  atan2(x, z) → left edge = -π, centre = 0, right edge = π
    # pitch ∈ [-π/2, π/2]: arcsin(y) → top = π/2, bottom = -π/2
    yaw   = np.arctan2(x, z)                       # [-π, π]
    pitch = np.arcsin(np.clip(y, -1.0, 1.0))       # [-π/2, π/2]

    px = (yaw   / (2 * math.pi) + 0.5) * eq_w     # [0, W]
    py = (0.5   - pitch / math.pi)      * eq_h     # [0, H]  (top = 0)

    # Clamp to valid range
    px = np.clip(px, 0, eq_w - 1).astype(np.float32)
    py = np.clip(py, 0, eq_h - 1).astype(np.float32)

    # Bilinear sampling
    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = np.clip(x0 + 1, 0, eq_w - 1)
    y1 = np.clip(y0 + 1, 0, eq_h - 1)
    fx = (px - x0)[..., np.newaxis]
    fy = (py - y0)[..., np.newaxis]

    tl = eq[y0, x0].astype(np.float32)
    tr = eq[y0, x1].astype(np.float32)
    bl = eq[y1, x0].astype(np.float32)
    br = eq[y1, x1].astype(np.float32)

    sampled = (tl*(1-fx)*(1-fy) + tr*fx*(1-fy) +
               bl*(1-fx)*fy     + br*fx*fy).astype(np.uint8)

    return Image.fromarray(sampled)


def generate_tiles(scene_id, level, verify=False):
    face_size, tiles_per_side = LEVELS[level]
    src_path = os.path.join(SRC_DIR, f'{scene_id}.jpg')
    out_base = os.path.join(TILE_DIR, '_verify') if verify else TILE_DIR
    print(f'\n[{scene_id}] Level {level} — {face_size}px face, {tiles_per_side}x{tiles_per_side} tiles')
    if verify:
        print(f'  (verify mode — output to tiles/_verify/)')

    equirect = Image.open(src_path).convert('RGB')
    print(f'  Source: {equirect.size[0]}x{equirect.size[1]}')

    for face in ['f', 'b', 'l', 'r', 'u', 'd']:
        face_img = equirect_to_face(equirect, face, face_size)

        for row in range(tiles_per_side):
            for col in range(tiles_per_side):
                tile_out_dir = os.path.join(out_base, scene_id, str(level), face, str(row))
                os.makedirs(tile_out_dir, exist_ok=True)

                x = col * TILE_SIZE
                y = row * TILE_SIZE
                tile = face_img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

                out_path = os.path.join(tile_out_dir, f'{col}.jpg')
                tile.save(out_path, 'JPEG', quality=92)

        print(f'  {face} done')


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level',  type=int, default=3, choices=[1, 2, 3],
                        help='Tile level to generate (default: 3 = 2048px)')
    parser.add_argument('--scene',  type=str, default=None,
                        help='Generate a single scene only (default: all)')
    parser.add_argument('--verify', action='store_true',
                        help='Shorthand: generate level 2 for orientation verification')
    args = parser.parse_args()

    level  = 2 if args.verify else args.level
    scenes = [args.scene] if args.scene else SCENES

    for scene in scenes:
        generate_tiles(scene, level)

    print('\nDone.')

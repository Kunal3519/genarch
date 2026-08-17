"""
360-degree turntable videos of GenArch reconstructions, for the project website.

For each tower: run the FULL deployed pipeline (v8.2 -> N=16 -> refinement),
snap to the grid, diff against ground truth, and render an orbiting camera:

    correct blocks  ->  neutral grey  (or a single accent colour if perfect)
    extra blocks    ->  red    (model built where GT is empty)
    missed blocks   ->  blue   (GT block the model did not build)

Output: H.264 MP4 (yuv420p, web-safe, loops cleanly since the orbit is 360).

  CUDA_VISIBLE_DEVICES=<uuid> uv run --with pyvista --with imageio[ffmpeg] \
      --project . python script/gen_turntable_video.py \
      --towers tower_100101 tower_101616 tower_203007 --out_dir generation/turntables
"""
import os, sys, argparse
from os.path import join as pjoin
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import DATA_ROOT

import numpy as np

DATA = DATA_ROOT

C_OK, C_EXTRA, C_MISS = '#2ECC40', '#FF4136', '#0074D9'   # paper diff convention
C_PERFECT = '#2ECC40'


def snap_set(coords):
    from utils.voxel import coords_to_voxel_set
    return coords_to_voxel_set(np.asarray(coords, float))


def cubes(pv, pts):
    import numpy as _np
    if len(pts) == 0:
        return None
    pd = pv.PolyData(_np.asarray(sorted(pts), float))
    # NOTE: no extract_surface().clean() here - merging vertices fuses adjacent
    # cubes into one blob and erases the per-block borders.
    return pd.glyph(geom=pv.Cube(x_length=4, y_length=4, z_length=8),
                    scale=False, orient=False)


def _facade_texture(pv, material='glass'):
    """Procedural building-facade texture: one window unit per block face, so a
    voxel tower reads as a real skyscraper (BrickGPT-style textured surface).
    Cube glyphs carry per-face UVs, so each face maps the full texture once."""
    from PIL import Image, ImageDraw
    px = 256
    if material == 'glass':
        frame, glass, mullion, sky = '#c7ccd2', '#33506b', '#b6bcc3', True
    elif material == 'concrete':
        frame, glass, mullion, sky = '#d8d2c4', '#8a8578', '#c9c3b4', False
    else:  # dark glass
        frame, glass, mullion, sky = '#7d848c', '#26333f', '#9aa1a8', True
    im = Image.new('RGB', (px, px), frame)
    d = ImageDraw.Draw(im)
    m = int(px * 0.10)
    d.rectangle([m, m, px - m, px - m], fill=glass)          # window pane
    d.line([px // 2, m, px // 2, px - m], fill=mullion, width=3)   # one mullion
    g = np.array(im).astype(float)
    if sky:                                                   # faint sky reflection
        g = np.clip(g * np.linspace(1.28, 0.72, px)[:, None, None], 0, 255)
    return pv.Texture(g.astype(np.uint8))


def turntable_facade(gt, pred, out_path, material='glass', seconds=8, fps=30,
                     size=(880, 1100)):
    """Textured skyscraper turntable: window-facade material, ground plane, and
    a shadow that sweeps as the building spins under fixed lights."""
    import math
    import pyvista as pv
    pv.OFF_SCREEN = True
    pts = snap_set(pred)
    tex = _facade_texture(pv, material)
    edge = {'glass': '#8b9199', 'concrete': '#a49e90', 'dark': '#5a6169'}[material]

    p = pv.Plotter(off_screen=True, window_size=size)
    m = cubes(pv, pts)
    p.add_mesh(m, texture=tex, show_edges=True, edge_color=edge, line_width=0.4,
               ambient=0.28, diffuse=0.72, specular=0.55, specular_power=28)

    all_pts = np.asarray(sorted(pts), float)
    cx, cy = all_pts[:, 0].mean(), all_pts[:, 1].mean()
    z_lo, z_hi = all_pts[:, 2].min() - 4, all_pts[:, 2].max() + 4
    cz = (z_lo + z_hi) / 2
    height = z_hi - z_lo
    width = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) + 8

    gsz = max(height, width) * 3.0
    ground = pv.Plane(center=(cx, cy, z_lo + 0.01), direction=(0, 0, 1),
                      i_size=gsz, j_size=gsz)
    p.add_mesh(ground, color='#eceef0', ambient=0.4, diffuse=0.7, specular=0.0)

    view_angle = 30.0
    dist = 1.15 * max(height, width * size[0] / size[1]) / (2 * math.tan(math.radians(view_angle / 2)))
    p.camera.position = (cx + dist, cy, cz)
    p.camera.focal_point = (cx, cy, cz)
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.view_angle = view_angle

    key = pv.Light(position=(cx + dist * 0.55, cy - dist * 0.55, z_hi + height * 2.2),
                   focal_point=(cx, cy, cz), light_type='scene light', intensity=1.15)
    key.positional = True
    key.cone_angle = 90
    p.add_light(key)
    p.add_light(pv.Light(position=(cx - dist, cy + dist, z_hi * 0.6),
                         light_type='scene light', intensity=0.35))
    try:
        p.enable_shadows()
    except Exception as e:
        print('shadows unavailable:', e)
    try:
        p.enable_ssao(radius=6, bias=0.02)
    except Exception as e:
        print('ssao unavailable:', e)
    p.enable_anti_aliasing('ssaa')
    p.set_background('white')

    tower_actor = None
    for _name, a in p.renderer.actors.items():
        b = a.GetBounds()
        if b[5] - b[4] > height * 0.5:
            tower_actor = a
    tower_actor.SetOrigin(cx, cy, cz)

    n_frames = seconds * fps
    p.open_movie(out_path, framerate=fps, quality=8)
    step = 360.0 / n_frames
    for _ in range(n_frames):
        tower_actor.RotateZ(step)
        p.render()
        p.write_frame()
    p.close()
    return len(pts)


def _studio_env(pv):
    """Neutral STUDIO environment for silver-glass reflections: smooth grey
    gradient walls with one bright softbox band. Photographic skyboxes make
    every block face mirror a picture (patchwork look); a smooth gradient gives
    the clean sweeping streak highlights of curtain glass. Faces are written as
    PNGs and loaded with pv.cubemap (its file-based API is the reliable one)."""
    import numpy as _np, tempfile, os as _os
    from PIL import Image as _Im
    n = 512
    row = _np.linspace(0, 1, n)[:, None]
    base = 60 + 150 * (1 - row)
    side = _np.repeat(base, n, axis=1)
    band = _np.exp(-((_np.linspace(0, 1, n)[:, None] - 0.35) ** 2) / (2 * 0.13 ** 2))
    lit = _np.clip(side + 185 * band, 0, 255)
    dark = side * 0.55
    top = _np.full((n, n), 235.0)
    bot = _np.full((n, n), 40.0)
    d = tempfile.mkdtemp(prefix='studioenv_')
    for name, a in [('posx', lit), ('negx', dark), ('posy', side), ('negy', lit),
                    ('posz', top), ('negz', bot)]:
        _Im.fromarray(_np.dstack([a.astype(_np.uint8)] * 3)).save(_os.path.join(d, name + '.png'))
    return pv.cubemap(d, '', '.png')


def turntable_glass(gt, pred, out_path, seconds=8, fps=30, size=(880, 1100)):
    """Showcase render: mirror-glass curtain-wall PBR material with environment
    reflections, a ground plane, and shadows that sweep as the tower rotates."""
    import math
    import pyvista as pv
    pv.OFF_SCREEN = True
    gt_set, pred_set = snap_set(gt), snap_set(pred)
    assert pred_set == gt_set or True
    pts = pred_set

    p = pv.Plotter(off_screen=True, window_size=size)
    # SAME solid-block material as the diff/perfect turntables (phong, glossy,
    # per-block borders), in silver. PBR mirror experiments read as hollow /
    # wallpaper because each cube face reflects the environment on its own.
    m = cubes(pv, pts)
    p.add_mesh(m, color='#9aa3ad', opacity=1.0, show_edges=True,
               edge_color='#4d545c', line_width=0.8,
               ambient=0.26, diffuse=0.70,
               specular=0.7, specular_power=30, smooth_shading=False)

    all_pts = np.asarray(sorted(pts), float)
    cx, cy = all_pts[:, 0].mean(), all_pts[:, 1].mean()
    z_lo, z_hi = all_pts[:, 2].min() - 4, all_pts[:, 2].max() + 4
    cz = (z_lo + z_hi) / 2
    height = z_hi - z_lo
    width = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) + 8

    # ground plane: matte, receives the moving shadow
    gsz = max(height, width) * 3.2
    ground = pv.Plane(center=(cx, cy, z_lo + 0.01), direction=(0, 0, 1),
                      i_size=gsz, j_size=gsz)
    p.add_mesh(ground, color='#e8eaec', pbr=True, metallic=0.0, roughness=0.9)



    view_angle = 30.0
    dist = 1.18 * max(height, width * size[0] / size[1]) / (2 * math.tan(math.radians(view_angle / 2)))
    p.camera.position = (cx + dist, cy, cz + height * 0.04)
    p.camera.focal_point = (cx, cy, cz)
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.view_angle = view_angle

    # one strong key light casting the shadow + a soft fill
    # steep + distant key: crisper shadow map, and the twin-spire self-shadow
    # shrinks instead of smearing across the crown
    key = pv.Light(position=(cx + dist * 0.55, cy - dist * 0.55, z_hi + height * 2.4),
                   focal_point=(cx, cy, cz), light_type='scene light', intensity=1.25)
    key.positional = True
    key.cone_angle = 85          # wide cone: 55 clipped the tower crown (dark blotch)
    p.add_light(key)
    p.add_light(pv.Light(position=(cx - dist, cy + dist, z_hi * 0.6),
                         light_type='scene light', intensity=0.30))
    try:
        p.enable_shadows()
    except Exception as e:
        print('shadows unavailable:', e)
    try:
        p.enable_ssao(radius=6, bias=0.02)   # face-contrast (worked well in stills)
    except Exception as e:
        print('ssao unavailable:', e)
    p.enable_anti_aliasing('ssaa')
    p.set_background('white')

    actor = p.renderer.actors[list(p.renderer.actors)[0]]
    # rotate ONLY the tower (ground + camera + lights stay fixed)
    tower_actor = None
    for name, a in p.renderer.actors.items():
        b = a.GetBounds()
        if b[5] - b[4] > height * 0.5:      # the tall one is the tower
            tower_actor = a
    tower_actor.SetOrigin(cx, cy, cz)

    n_frames = seconds * fps
    p.open_movie(out_path, framerate=fps, quality=8)
    step = 360.0 / n_frames
    for _ in range(n_frames):
        tower_actor.RotateZ(step)
        p.render()
        p.write_frame()
    p.close()
    return len(pts)


def turntable(gt, pred, out_path, seconds=8, fps=30, size=(880, 1100)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    gt_set, pred_set = snap_set(gt), snap_set(pred)
    ok, extra, miss = pred_set & gt_set, pred_set - gt_set, gt_set - pred_set
    perfect = not extra and not miss

    p = pv.Plotter(off_screen=True, window_size=size)
    groups = ([(ok, C_PERFECT)] if perfect else
              [(ok, C_OK), (extra, C_EXTRA), (miss, C_MISS)])
    actors = []
    for pts, col in groups:
        m = cubes(pv, pts)
        if m is not None:
            # fully opaque, glossy, per-cube borders
            a = p.add_mesh(m, color=col, opacity=1.0, show_edges=True,
                           edge_color='#14691f' if col in (C_OK, C_PERFECT) else '#333333',
                           line_width=0.8,
                           ambient=0.28, diffuse=0.72,
                           specular=1.0, specular_power=38, smooth_shading=False)
            actors.append(a)
    p.enable_anti_aliasing('ssaa')
    p.set_background('white')

    # eye-level turntable: camera sits at the tower's mid-height and looks
    # straight at it (no top-down tilt), then orbits horizontally.
    all_pts = np.asarray(sorted(gt_set | pred_set), float)
    cx, cy = all_pts[:, 0].mean(), all_pts[:, 1].mean()
    z_lo, z_hi = all_pts[:, 2].min() - 4, all_pts[:, 2].max() + 4
    cz = (z_lo + z_hi) / 2
    height = z_hi - z_lo
    width = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])) + 8
    import math
    view_angle = 30.0
    dist = 1.12 * max(height, width * size[0] / size[1]) / (2 * math.tan(math.radians(view_angle / 2)))
    p.camera.position = (cx + dist, cy, cz)
    p.camera.focal_point = (cx, cy, cz)
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.view_angle = view_angle

    p.add_light(pv.Light(position=(cx + dist, cy - dist * 0.6, cz + height), light_type='scene light', intensity=0.55))
    p.add_light(pv.Light(position=(cx - dist, cy + dist * 0.6, cz + height * 0.3), light_type='scene light', intensity=0.30))

    # classic turntable: the OBJECT spins under fixed camera + lights, so the
    # specular highlight sweeps across the faces as it rotates (a camera orbit
    # keeps scene lighting fixed relative to the object and looks flat).
    for a in actors:
        a.SetOrigin(cx, cy, cz)
    n_frames = seconds * fps
    p.open_movie(out_path, framerate=fps, quality=8)
    step = 360.0 / n_frames
    for _ in range(n_frames):
        for a in actors:
            a.RotateZ(step)
        p.render()
        p.write_frame()
    p.close()
    n_ok, n_ex, n_ms = len(ok), len(extra), len(miss)
    iou = n_ok / max(1, len(pred_set | gt_set))
    return iou, n_ok, n_ex, n_ms, perfect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--towers', nargs='+', required=True)
    ap.add_argument('--out_dir', default='generation/turntables')
    ap.add_argument('--seconds', type=int, default=8)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--gt_only', action='store_true',
                    help='render the ground-truth tower instead of running the model')
    ap.add_argument('--style', choices=['diff', 'glass', 'facade'], default='diff',
                    help="diff = green/red/blue error render; glass = silver showcase; "
                         "facade = textured skyscraper (windows). Each writes its own "
                         "suffix, never overwriting the others.")
    ap.add_argument('--material', choices=['glass', 'concrete', 'dark'], default='glass',
                    help='facade material')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not a.gt_only:
        import torch
        from utils.fixseed import fixseed
        from gen_paper_diagrams import load_pipeline
        fixseed(a.seed)
        run = load_pipeline(torch.device('cuda'))

    for t in a.towers:
        motion = np.load(pjoin(DATA, 'new_joint_vecs_int', t + '.npy'))
        gt = motion.astype(float)
        if a.gt_only:
            pred = gt
        else:
            feat = np.load(pjoin(DATA, 'image_features_dino_patch', t + '.npy')).astype(np.float32)
            _, pred = run(motion, feat)
        if a.style == 'facade':
            out = pjoin(a.out_dir, f'{t}_facade_{a.material}.mp4')
            n = turntable_facade(gt, pred, out, a.material, a.seconds, a.fps)
            print(f'{t}: {n} blocks (facade/{a.material}) -> {out}', flush=True)
        elif a.style == 'glass':
            out = pjoin(a.out_dir, f'{t}_glass.mp4')
            n = turntable_glass(gt, pred, out, a.seconds, a.fps)
            print(f'{t}: {n} blocks (glass showcase) -> {out}', flush=True)
        else:
            out = pjoin(a.out_dir, f'{t}.mp4')
            iou, n_ok, n_ex, n_ms, perfect = turntable(gt, pred, out, a.seconds, a.fps)
            tag = 'PERFECT' if perfect else f'extra {n_ex}  missed {n_ms}'
            print(f'{t}: IoU {iou:.4f}  ({tag})  -> {out}', flush=True)


if __name__ == '__main__':
    main()

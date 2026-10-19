"""
Door-hold clip for a wall-mounted magnetic door stopper.

Single printed part that:
  * sits on the stopper's 17 mm rod right behind the 22 mm magnet bulb
    (open-bottom channel, drops on from above, gravity + the bulb keep it there),
  * runs an arm along the wall-side face of the door,
  * wraps the door's free edge with a C-hook so the door can't swing closed.

When wind pulls the door, the room-side jaw of the C pulls the clip forward and
the clip's spherical seat bears on the back of the bulb, which is fixed to the wall.

Coordinates (installed):
  X  along the door face, towards the door's free edge
  Y  from the wall towards the door (y = 0 at the clip's front face)
  Z  up (z = 0 on the rod axis)

Requires: pip install manifold3d numpy
Run:      python doorstop_clip.py      -> writes the STLs next to this file
"""

import math
import os
import struct

import numpy as np
import manifold3d as m3d

# ---------------------------------------------------------------- measurements
ROD_D = 17.0              # stopper rod diameter
BALL_D = 22.0             # magnet bulb diameter at the rod tip
DOOR_T = 35.0             # door thickness
EDGE_D = 130.0            # rod/bulb centre -> door's free edge, along the door face
BALL_BACK_TO_DOOR = 40.0  # door surface -> back of bulb (where rod enters it)
ROD_FREE_LEN = 40.0       # bare rod between wall collar and bulb (min of 40-50)

# ---------------------------------------------------------------- clearances
BORE_CLR = 1.0            # rod channel = ROD_D + this
DOOR_T_CLR = 1.5          # C opening = DOOR_T + this
DOOR_FACE_EXTRA = 2.0     # arm face sits this far past the measured door face
EDGE_CLR = 4.0            # gap between door edge and the C's bridge

# ---------------------------------------------------------------- part sizes
H = 28.0                  # part height (Z)
Z_BOT = -13.0             # bottom of part relative to rod axis
CLIP_LEN = 28.0           # length of rod channel (Y)
CLIP_HALF_W = 19.0        # clip block half width (X); side walls ~10 mm
DOME_CLEAR_X = 26.0       # arm starts here so it clears the bulb + door dome
ARM_DEPTH = 20.0          # arm thickness (Y)
BRIDGE_T = 14.0           # C bridge thickness (X)
JAW_T = 12.0              # room-side jaw thickness (Y)
JAW_LEN = 45.0            # how far the room-side jaw reaches back from the edge
SEAT_DEPTH = 1.2          # how deep the spherical bulb seat is cut at the bore edge
RECESS_DEPTH = 3.0        # upper-front relief so only the lower seat touches the bulb

SEGMENTS = 128

# ---------------------------------------------------------------- derived
BORE_R = (ROD_D + BORE_CLR) / 2
BALL_R = BALL_D / 2
SEAT_Y = -SEAT_DEPTH + math.sqrt(BALL_R**2 - BORE_R**2)          # bulb centre when seated
JUNCTION_Y = SEAT_Y - math.sqrt(BALL_R**2 - (ROD_D / 2) ** 2)    # where rod enters bulb
DOOR_IN_Y = JUNCTION_Y + BALL_BACK_TO_DOOR                       # door wall-side face
ARM_FRONT_Y = DOOR_IN_Y + DOOR_FACE_EXTRA
JAW_IN_Y = ARM_FRONT_Y + DOOR_T + DOOR_T_CLR
JAW_OUT_Y = JAW_IN_Y + JAW_T
ARM_REAR_Y = ARM_FRONT_Y - ARM_DEPTH
BRIDGE_IN_X = EDGE_D + EDGE_CLR
BRIDGE_OUT_X = BRIDGE_IN_X + BRIDGE_T
JAW_TIP_X = EDGE_D - JAW_LEN
GUSSET_X = DOME_CLEAR_X + 34.0

assert CLIP_LEN <= ROD_FREE_LEN - 6, "clip longer than the bare rod"


def fillet_polygon(pts, radii, seg=SEGMENTS):
    """Round each vertex of a CCW polygon with its own radius (convex or concave)."""
    pts = [np.asarray(p, float) for p in pts]
    n = len(pts)
    out = []
    for i in range(n):
        p0, p1, p2, r = pts[i - 1], pts[i], pts[(i + 1) % n], radii[i]
        if r <= 0:
            out.append(p1)
            continue
        v1 = (p0 - p1) / np.linalg.norm(p0 - p1)
        v2 = (p2 - p1) / np.linalg.norm(p2 - p1)
        ang = math.acos(np.clip(np.dot(v1, v2), -1, 1))
        t = r / math.tan(ang / 2)
        assert t < 0.5 * min(np.linalg.norm(p0 - p1), np.linalg.norm(p2 - p1)), f"fillet {i} too big"
        a, b = p1 + v1 * t, p1 + v2 * t
        bis = (v1 + v2) / np.linalg.norm(v1 + v2)
        c = p1 + bis * (r / math.sin(ang / 2))
        th_a = math.atan2(*(a - c)[::-1])
        th_b = math.atan2(*(b - c)[::-1])
        d = (th_b - th_a + math.pi) % (2 * math.pi) - math.pi
        k = max(2, int(abs(d) / (2 * math.pi) * seg))
        out += [c + r * np.array([math.cos(th_a + d * j / k), math.sin(th_a + d * j / k)]) for j in range(k + 1)]
    return [tuple(p) for p in out]


def plan_profile():
    """Top-view outline of the part (CCW) with per-corner fillet radii."""
    corners = [
        ((-CLIP_HALF_W, -CLIP_LEN), 2),            # clip rear-left
        ((CLIP_HALF_W, -CLIP_LEN), 2),             # clip rear-right
        ((GUSSET_X, ARM_REAR_Y), 10),              # gusset meets arm (concave)
        ((BRIDGE_OUT_X, ARM_REAR_Y), 3),           # bridge outer, wall side
        ((BRIDGE_OUT_X, JAW_OUT_Y), 3),            # bridge outer, room side
        ((JAW_TIP_X, JAW_OUT_Y), 3),               # jaw tip outer
        ((JAW_TIP_X, JAW_IN_Y), 2),                # jaw tip inner (touches door)
        ((BRIDGE_IN_X, JAW_IN_Y), 3),              # inside of C (concave)
        ((BRIDGE_IN_X, ARM_FRONT_Y), 3),           # inside of C (concave)
        ((DOME_CLEAR_X, ARM_FRONT_Y), 2),          # arm start, door side
        ((DOME_CLEAR_X, 0.0), 6),                  # clip front meets column (concave)
        ((-CLIP_HALF_W, 0.0), 2),                  # clip front-left
    ]
    return fillet_polygon([c for c, _ in corners], [r for _, r in corners])


def prism_xz(poly_xz, y0, y1):
    """Extrude a polygon drawn in the XZ plane along +Y from y0 to y1."""
    cs = m3d.CrossSection([poly_xz])
    return cs.extrude(y1 - y0).rotate((90, 0, 0)).translate((0, y1, 0))


def cyl_y(r, y0, y1):
    return m3d.Manifold.cylinder(y1 - y0, r, circular_segments=SEGMENTS).rotate((90, 0, 0)).translate((0, y1, 0))


def build():
    body = m3d.CrossSection([plan_profile()]).extrude(H).translate((0, 0, Z_BOT))

    y0, y1 = -CLIP_LEN - 1, 1.0
    # rod channel + open slot underneath (drops onto the rod from above), with lead-in
    bore = cyl_y(BORE_R, y0, y1)
    lead = 1.5
    slot = prism_xz([(-BORE_R, 0.0), (-BORE_R, Z_BOT + lead), (-BORE_R - lead, Z_BOT - 0.01),
                     (-BORE_R - lead, Z_BOT - 1), (BORE_R + lead, Z_BOT - 1),
                     (BORE_R + lead, Z_BOT - 0.01), (BORE_R, Z_BOT + lead), (BORE_R, 0.0)], y0, y1)

    # spherical seat for the bulb (only the lower half ends up touching)
    seat = m3d.Manifold.sphere(BALL_R, SEGMENTS).translate((0, SEAT_Y, 0))

    # relieve the upper half of the front so the bulb can't cam the clip upwards;
    # 45-degree roof keeps it printable upside-down without supports
    v = 40.0
    vee = prism_xz([(-BORE_R, 0.0), (BORE_R, 0.0), (BORE_R + v, v), (-BORE_R - v, v)],
                   -RECESS_DEPTH, 1.0)
    recess = cyl_y(BALL_R + 0.8, -RECESS_DEPTH, 1.0) ^ vee

    return body - bore - slot - seat - recess


def write_stl(manifold, path):
    mesh = manifold.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    with open(path, "wb") as fh:
        fh.write(b"doorstop clip".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        for nn, t in zip(n, tri):
            fh.write(struct.pack("<12fH", *nn, *t.ravel(), 0))


def for_print(part):
    """Flip upside-down (rod channel opening up) and drop onto the bed: no supports needed."""
    p = part.rotate((180, 0, 0))
    lo = p.bounding_box()[:3]
    return p.translate((-lo[0], -lo[1], -lo[2]))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    part = build()
    write_stl(for_print(part), os.path.join(here, "doorstop_clip.stl"))
    write_stl(for_print(part.mirror((1, 0, 0))), os.path.join(here, "doorstop_clip_mirrored.stl"))
    bb = part.bounding_box()
    print(f"volume {part.volume() / 1000:.1f} cm3, genus {part.genus()}, status {part.status()}")
    print(f"size X {bb[3] - bb[0]:.1f}  Y {bb[4] - bb[1]:.1f}  Z {bb[5] - bb[2]:.1f} mm")
    print(f"bulb seated centre y={SEAT_Y:.2f}, door faces y={DOOR_IN_Y:.2f}..{DOOR_IN_Y + DOOR_T:.2f}, "
          f"arm face y={ARM_FRONT_Y:.2f}, jaw y={JAW_IN_Y:.2f}")

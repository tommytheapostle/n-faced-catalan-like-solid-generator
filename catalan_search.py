#!/usr/bin/env python3
"""Exhaustive search for the optimal Catalan-like polyhedra C(n).

For each admissible hexagon count n this program enumerates *every* isolated-pentagon-
rule (IPR) fullerene C_{2n+20}, forms the polyhedron C(n) of Definition 3 by pole
contraction, discards those possessing an isolated hexagon, realizes the remainder in
the Koebe-Andreev-Thurston midsphere (canonical) form, and reports the isomer of least
face-area ratio rho = A_max / A_min. No isomer is pruned by any heuristic, so the
reported rho_min(n) are exhaustive minima over the admissible family; this is the
computation behind Table 1.

Dependencies
------------
  * numpy, networkx
  * buckygen -- the fullerene generator of Brinkmann, Goedgebeur and McKay. Build it
    from source (e.g. the mirror at https://github.com/evanberkowitz/buckygen) with
    `make`, then point the module at the binary via the BUCKYGEN environment variable
    or the --buckygen option (default: ./buckygen/buckygen).
  * make_best_offs.py (shipped alongside) -- shared construction and canonicalization.

Method
------
  1. Generation.  `buckygen (n+12) -d -I` emits, in planar_code, every IPR fullerene
     with n+12 faces (i.e. C_{2n+20}) as a cubic plane graph.  Its pentagonal and
     hexagonal faces are recovered from the rotation system by face tracing.
  2. Construction.  Each fullerene is mapped to C(n) by contracting its twelve
     pentagons to poles (make_best_offs.construct_polytope).  An isomer yielding a
     hexagonal face (an isolated hexagon, p=0) is inadmissible and discarded.
  3. Canonicalization.  Every admissible isomer is realized in the midsphere-canonical
     form; rho and the insphere ratio iota are read off.  The optimizer's coordinates
     are then refined to machine precision by a Newton step on the exact tangency and
     planarity constraints (complex-step Jacobian).
  4. Minimization.  The exact minimum of rho over the admissible isomers is returned,
     with the optimizer's point group -- graph automorphisms realized as isometries,
     after Mani's theorem -- and its face-size vector.

Usage
-----
    python3 catalan_search.py 40                 # sweep C(40); print the Table 1 row
    python3 catalan_search.py 40 --off out_dir   # ... and write the optimal OFF
    python3 catalan_search.py --all              # reproduce every row of Table 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from make_best_offs import (
    construct_polytope,
    spectral_embedding,
    canonicalize_midsphere,
    face_metrics,
    write_off,
    polish,
)

Face = Tuple[int, ...]
BUCKYGEN = os.environ.get("BUCKYGEN", "./buckygen/buckygen")

# The 30 achievable hexagon counts (Section 4).
ACHIEVABLE = [20, 25] + list(range(27, 53)) + [55, 60]


# ---------------------------------------------------------------------------
# 1.  Generation: buckygen planar_code -> face lists
# ---------------------------------------------------------------------------

_HEADER = b">>planar_code<<"


def generate_ipr(n: int, buckygen: str = BUCKYGEN, tmp: str = "/tmp/cn.pc") -> List[List[List[int]]]:
    """Return the rotation systems of all IPR fullerenes C_{2n+20}.

    buckygen is invoked with argument n+12 (the number of faces), the flag -d (cubic
    output) and -I (isolated pentagon rule).  Each returned graph is a list giving, for
    every vertex, its three neighbours in cyclic (embedding) order.
    """
    subprocess.run([buckygen, str(n + 12), "-d", "-I", tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    data = open(tmp, "rb").read()
    if data[:15] != _HEADER:
        raise ValueError("unexpected planar_code header")
    i, N, graphs = 15, len(data), []
    while i < N:
        nv = data[i]; i += 1
        block = np.frombuffer(data, dtype=np.uint8, count=nv * 4, offset=i).reshape(nv, 4)
        i += nv * 4
        graphs.append([(block[v, :3].astype(int) - 1).tolist() for v in range(nv)])
    return graphs


def trace_faces(rotation: Sequence[Sequence[int]]) -> List[Face]:
    """Recover the faces of a plane graph from its rotation system.

    The face to the right of the directed edge u->v continues with v->w, where w is the
    neighbour following u in the cyclic order at v.
    """
    seen, faces = set(), []
    for u in range(len(rotation)):
        for v in rotation[u]:
            if (u, v) in seen:
                continue
            face, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b)); face.append(a)
                r = rotation[b]; j = r.index(a); a, b = b, r[(j + 1) % len(r)]
            faces.append(tuple(face))
    return faces


def admissible_cn(rotation: Sequence[Sequence[int]]) -> Optional[Tuple[int, List[Face]]]:
    """Build C(n) from a fullerene and return it iff admissible (no hexagonal face)."""
    fullerene_faces = [f for f in trace_faces(rotation) if len(f) in (5, 6)]
    nv, faces, _ = construct_polytope(fullerene_faces)
    if max(len(f) for f in faces) > 5:      # an isolated hexagon survived
        return None
    return nv, faces


# ---------------------------------------------------------------------------
# 3.  Point group (Mani: graph automorphisms are isometries of the canonical form)
# ---------------------------------------------------------------------------

def _isometries(V: np.ndarray, faces: Sequence[Face]) -> List[np.ndarray]:
    G = nx.Graph()
    for f in faces:
        k = len(f)
        for i in range(k):
            G.add_edge(f[i], f[(i + 1) % k])
    Vc = V - V.mean(0); n = len(Vc); mats = []
    for iso in nx.algorithms.isomorphism.GraphMatcher(G, G).isomorphisms_iter():
        P = np.array([iso[i] for i in range(n)])
        U, _, Wt = np.linalg.svd(Vc[P].T @ Vc)
        M = U @ Wt
        if np.linalg.norm(M @ Vc.T - Vc[P].T) < 1e-2 * np.sqrt(n):
            if not any(np.allclose(M, X, atol=1e-2) for X in mats):
                mats.append(M)
    return mats


def _axis_order(M):
    ang = np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))
    if ang < 1e-2:
        return None, 1
    w, v = np.linalg.eig(M)
    ax = np.real(v[:, np.argmin(np.abs(w - 1))]); ax /= np.linalg.norm(ax)
    return ax, int(round(2 * np.pi / ang))


def _classify(mats: List[np.ndarray]) -> str:
    """Assign the Schoenflies symbol of a finite subgroup of O(3)."""
    I3 = np.eye(3)
    proper = [M for M in mats if np.linalg.det(M) > 0]
    refl = [M for M in mats if np.linalg.det(M) < 0 and abs(np.trace(M) - 1) < 1e-1]
    has_i = any(np.allclose(M, -I3, atol=1e-2) for M in mats)
    rots = [(ax, o) for ax, o in (_axis_order(M) for M in proper) if o > 1]
    maxn = max((o for _, o in rots), default=1)
    n3 = sum(1 for _, o in rots if o == 3)
    paxis = next((ax for ax, o in rots if o == maxn), None)
    npr = len(proper)
    if npr == 60:
        typ, nn = "I", 60
    elif npr == 24:
        typ, nn = "O", 24
    elif npr == 12 and maxn == 3 and n3 >= 8:
        typ, nn = "T", 12
    elif npr == 1:
        typ, nn = "C", 1
    elif npr == maxn:
        typ, nn = "C", maxn
    elif npr == 2 * maxn:
        typ, nn = "D", maxn
    else:
        return f"?{npr}/{len(mats)}"
    chiral = (len(mats) == npr)

    def normal(M):
        w, v = np.linalg.eig(M); nm = np.real(v[:, np.argmin(np.abs(w + 1))])
        return nm / np.linalg.norm(nm)

    sigma_h = paxis is not None and any(abs(abs(normal(M) @ paxis) - 1) < 1e-1 for M in refl)
    sigma_v = paxis is not None and any(abs(normal(M) @ paxis) < 1e-1 for M in refl)
    if typ in ("I", "O"):
        return typ if chiral else typ + "_h"
    if typ == "T":
        return "T" if chiral else ("T_h" if has_i else "T_d")
    if typ == "C":
        if nn == 1:
            return "C_i" if has_i else ("C_s" if refl else "C_1")
        if chiral:
            return f"C_{nn}"
        return f"C_{nn}h" if sigma_h else (f"C_{nn}v" if sigma_v else f"S_{2*nn}")
    if chiral:
        return f"D_{nn}"
    return f"D_{nn}h" if sigma_h else f"D_{nn}d"


def point_group(V: np.ndarray, faces: Sequence[Face]) -> str:
    return _classify(_isometries(V, faces))


# ---------------------------------------------------------------------------
# 4.  Exhaustive sweep
# ---------------------------------------------------------------------------

def sweep(n: int, iterations: int = 3000, buckygen: str = BUCKYGEN) -> Dict[str, object]:
    """Exhaustively minimize rho over the admissible IPR isomers of C_{2n+20}.

    Every admissible isomer is canonicalized (no pruning); the optimizer is then
    polished to machine precision and its point group computed.
    """
    best = None
    for rotation in generate_ipr(n, buckygen):
        cn = admissible_cn(rotation)
        if cn is None:
            continue
        nv, faces = cn
        V = canonicalize_midsphere(spectral_embedding(nv, faces), faces, iterations)
        rho = face_metrics(V, faces)["area_ratio"]
        if best is None or rho < best[0]:
            best = (rho, nv, faces)
    if best is None:
        raise RuntimeError(f"no admissible C({n})")

    _, nv, faces = best
    V = canonicalize_midsphere(spectral_embedding(nv, faces), faces, max(iterations, 6000))
    V = polish(V, faces)
    metrics = face_metrics(V, faces)
    return {
        "n": n,
        "source": 2 * n + 20,
        "rho": float(metrics["area_ratio"]),
        "iota": float(metrics["insphere_ratio"]),
        "point_group": point_group(V, faces),
        "face_vector": dict(sorted(Counter(len(f) for f in faces).items())),
        "vertices": V,
        "faces": [list(f) for f in faces],
    }


def _row(r: Dict[str, object]) -> str:
    return (f"n={r['n']:2d}  C{r['source']:<3}  {r['point_group']:>4}  "
            f"rho={r['rho']:.5f}  iota={r['iota']:.4f}  {r['face_vector']}")


def main(argv: Sequence[str] = ()) -> None:
    ap = argparse.ArgumentParser(description="Exhaustive search for optimal C(n).")
    ap.add_argument("n", nargs="*", type=int, help="hexagon counts to sweep")
    ap.add_argument("--all", action="store_true", help="reproduce every row of Table 1")
    ap.add_argument("--off", metavar="DIR", help="also write the optimal OFF to DIR")
    ap.add_argument("--buckygen", default=BUCKYGEN, help="path to the buckygen binary")
    args = ap.parse_args(list(argv) or None)

    targets = ACHIEVABLE if args.all else args.n
    if not targets:
        ap.error("give one or more n, or --all")
    if args.off:
        os.makedirs(args.off, exist_ok=True)
    for n in targets:
        r = sweep(n, buckygen=args.buckygen)
        print(_row(r), flush=True)
        if args.off:
            write_off(os.path.join(args.off, f"catalan-like_n{n}.off"), r["vertices"], r["faces"])


if __name__ == "__main__":
    main(sys.argv[1:])

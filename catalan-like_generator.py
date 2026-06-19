#!/usr/bin/env python3
"""
catalan-like_generator.py
===================
Generates valid n-faced Catalan-like polyhedra for n >= 27.

Faces consist of near-regular hexagons, shields (irregular pentagons), rhombi/trapezoids & near-regular triangles.
All non-pole vertices have degree 3; exactly 12 poles have degree 5.
The Euler identity 2*n_quads + n_shields = 12 is automatically satisfied.

Algorithm:
  1. Subdivision path k^2*(m+10)-10: exact and deterministic, tried first.
  2. Anchor + edge-split: build a nearby subdivisible n', split k hex-hex edges.
  3. Pure Thomson fallback with greedy edge flips and vertex-split repair.
  4. Stabilisation: face planarity -> midsphere canonicalisation -> area equalization.

Usage:
  python catalan-like_generator.py 400
  python catalan-like_generator.py 1000 --output my_polyhedron.off
  python catalan-like_generator.py 500 --no-stabilize
"""

import sys, os, time, argparse
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from collections import defaultdict, Counter


_PHI = (1 + np.sqrt(5)) / 2
_ICO = np.array([
    [0, 1, _PHI], [0, -1, _PHI], [0, 1, -_PHI], [0, -1, -_PHI],
    [1, _PHI, 0], [-1, _PHI, 0], [1, -_PHI, 0], [-1, -_PHI, 0],
    [_PHI, 0, 1], [-_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, -1],
], dtype=np.float64)
_ICO /= np.linalg.norm(_ICO[0])


def fibonacci_sphere(n, seed=0):
    rng = np.random.RandomState(seed)
    phi_g = (1 + np.sqrt(5)) / 2
    pts = []
    for i in range(n):
        theta = np.arccos(max(-1.0, min(1.0, 1 - 2*(i+0.5)/n)))
        phi_i = 2*np.pi*i / phi_g
        pts.append([np.sin(theta)*np.cos(phi_i),
                    np.sin(theta)*np.sin(phi_i),
                    np.cos(theta)])
    pts = np.array(pts) + rng.randn(n, 3) * 0.01
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def thomson(pts, n_iter=500):
    nv = len(pts)
    def eg(x):
        p = x.reshape(nv, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        d = p[:, None] - p[None]
        r2 = np.sum(d**2, axis=2); np.fill_diagonal(r2, 1.0)
        r = np.sqrt(r2); r3 = r2 * r
        E = np.sum(1.0/r) / 2
        G = (-d / r3[:,:,None]).sum(1)
        G -= np.sum(G*p, axis=1, keepdims=True) * p
        return E, G.flatten()
    res = minimize(eg, pts.flatten(), jac=True, method='L-BFGS-B',
                   options={'maxiter': n_iter, 'ftol': 1e-15, 'gtol': 1e-10})
    p = res.x.reshape(nv, 3)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def triangulate(pts):
    hull = ConvexHull(pts); cen = pts.mean(0); tris = []
    for tri in hull.simplices:
        v0, v1, v2 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        if np.dot(np.cross(v1-v0, v2-v0), (v0+v1+v2)/3 - cen) < 0:
            tris.append([tri[0], tri[2], tri[1]])
        else:
            tris.append(list(tri))
    return tris


def spring_relax(pts, tris, n_iter=200):
    n = len(pts)
    seen = set(); eu = []; ev = []
    for t in tris:
        for j in range(3):
            e = (min(t[j], t[(j+1)%3]), max(t[j], t[(j+1)%3]))
            if e not in seen:
                seen.add(e); eu.append(e[0]); ev.append(e[1])
    eu = np.array(eu); ev = np.array(ev)
    L0 = np.mean(np.linalg.norm(pts[eu] - pts[ev], axis=1))
    for _ in range(n_iter):
        diff = pts[ev] - pts[eu]; L = np.linalg.norm(diff, axis=1, keepdims=True)
        f = (L - L0) * diff / (L + 1e-12); Fi = np.zeros((n, 3))
        np.add.at(Fi, eu, f); np.add.at(Fi, ev, -f)
        Fi -= np.sum(Fi*pts, axis=1, keepdims=True) * pts
        pts = pts + 0.008*Fi
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _pen(d):
    if d in (5, 6): return 0
    if d in (4, 7): return 50
    return 500


def _build_mesh(tris):
    ef = defaultdict(list); deg = Counter(); ve = defaultdict(set)
    for t in tris:
        for v in t: deg[v] += 1
    for ti, t in enumerate(tris):
        for j in range(3):
            e = (min(t[j], t[(j+1)%3]), max(t[j], t[(j+1)%3]))
            ef[e].append(ti); ve[e[0]].add(e); ve[e[1]].add(e)
    return ef, deg, ve


def _score(deg, ef):
    sc = sum(_pen(d) for d in deg.values())
    for e, fis in ef.items():
        if fis and deg[e[0]] == 5 and deg[e[1]] == 5:
            sc += 100
    return sc


def flip_pass(n, tris, rng):
    ef, deg, ve = _build_mesh(tris)
    keys = list(ef.keys())
    for e_idx in rng.permutation(len(keys)):
        if e_idx >= len(keys): continue
        ek = keys[e_idx]
        fis = ef.get(ek, [])
        if len(fis) != 2: continue
        ti0, ti1 = fis; t0, t1 = tris[ti0], tris[ti1]
        cl = [v for v in t0 if v != ek[0] and v != ek[1]]
        dl = [v for v in t1 if v != ek[0] and v != ek[1]]
        if not cl or not dl or cl[0] == dl[0]: continue
        c, d = cl[0], dl[0]; a, b = ek

        def pa(a, b, c, d):
            na, nb, nc, nd = deg[a]-1, deg[b]-1, deg[c]+1, deg[d]+1
            sc = _pen(na)+_pen(nb)+_pen(nc)+_pen(nd)
            if nc == 5 and nd == 5: sc += 100
            if na == 5:
                for e2 in ve[a]:
                    u = e2[0] if e2[1] == a else e2[1]
                    if u != b and u != c and u != d and deg[u] == 5: sc += 100
                if nc == 5: sc += 100
                if nd == 5: sc += 100
            if nb == 5:
                for e2 in ve[b]:
                    u = e2[0] if e2[1] == b else e2[1]
                    if u != a and u != c and u != d and deg[u] == 5: sc += 100
                if nc == 5: sc += 100
                if nd == 5: sc += 100
            if nc == 5:
                for e2 in ve[c]:
                    u = e2[0] if e2[1] == c else e2[1]
                    if u != a and u != b and deg[u] == 5: sc += 100
            if nd == 5:
                for e2 in ve[d]:
                    u = e2[0] if e2[1] == d else e2[1]
                    if u != a and u != b and deg[u] == 5: sc += 100
            return sc

        def pb(a, b, c, d):
            sc = _pen(deg[a])+_pen(deg[b])+_pen(deg[c])+_pen(deg[d])
            if deg[a] == 5 and deg[b] == 5: sc += 100
            return sc

        if pa(a, b, c, d) <= pb(a, b, c, d):
            for ti, tri in [(ti0, t0), (ti1, t1)]:
                for j in range(3):
                    oe = (min(tri[j],tri[(j+1)%3]), max(tri[j],tri[(j+1)%3]))
                    if ti in ef[oe]: ef[oe].remove(ti)
                    ve[tri[j]].discard(oe); ve[tri[(j+1)%3]].discard(oe)
            nt0 = [a,c,d]; nt1 = [b,d,c]
            tris[ti0] = nt0; tris[ti1] = nt1
            for ti, tri in [(ti0,nt0),(ti1,nt1)]:
                for j in range(3):
                    ne = (min(tri[j],tri[(j+1)%3]), max(tri[j],tri[(j+1)%3]))
                    ef[ne].append(ti); ve[tri[j]].add(ne); ve[tri[(j+1)%3]].add(ne)
            deg[a] -= 1; deg[b] -= 1; deg[c] += 1; deg[d] += 1
            keys = list(ef.keys())
    return tris, _score(deg, ef), dict(deg)


def vertex_split_repair(pts, tris, deg_map, rng, verbose=False):
    """Inject one vertex on an edge adjacent to the worst bad vertex, then
    flip-pass to restore score=0. Returns (pts_new, tris_new, score_new)."""
    bad = [v for v, d in deg_map.items() if d not in (4, 5, 6)]
    if not bad:
        return pts, tris, _score(*_build_mesh(tris)[:2])

    bad_v = max(bad, key=lambda v: abs(deg_map[v] - 6))

    ef = defaultdict(list)
    for ti, t in enumerate(tris):
        for j in range(3):
            e = (min(t[j], t[(j+1)%3]), max(t[j], t[(j+1)%3]))
            ef[e].append(ti)

    candidates = [(e, fis) for e, fis in ef.items()
                  if bad_v in e and len(fis) == 2]
    if not candidates:
        return pts, tris, _score(*_build_mesh(tris)[:2])

    best_sc = _score(*_build_mesh(tris)[:2])
    best = (pts, tris, best_sc)

    for idx in list(rng.permutation(len(candidates)))[:min(len(candidates), 12)]:
        ce, cfis = candidates[idx]
        a, b = ce; ti0, ti1 = cfis
        t0, t1 = tris[ti0], tris[ti1]
        c_ = [v for v in t0 if v != a and v != b]
        d_ = [v for v in t1 if v != a and v != b]
        if not c_ or not d_: continue
        c_, d_ = c_[0], d_[0]

        vm = (pts[a] + pts[b]) / 2
        nm = np.linalg.norm(vm)
        if nm < 1e-10: continue
        vm /= nm; vi = len(pts)

        pts_try = np.vstack([pts, vm[None]])
        tris_try = [t for i, t in enumerate(tris) if i not in (ti0, ti1)]
        tris_try += [[a, vi, c_], [vi, b, c_], [a, d_, vi], [vi, d_, b]]

        rng2 = np.random.RandomState(int(rng.randint(100000)))
        for _pass in range(8):
            tris_try, sc_try, deg_try = flip_pass(len(pts_try), tris_try, rng2)
            if sc_try == 0: break
            pts_try = spring_relax(pts_try, tris_try, 150)
            try:
                tris_try = triangulate(pts_try)
            except Exception:
                sc_try = 9999; break

        if sc_try < best_sc:
            best_sc = sc_try
            best = (pts_try, tris_try, sc_try)
            if verbose:
                print(f'    split edge ({a},{b}): score -> {sc_try}')
            if sc_try == 0:
                break

    return best


def find_subdivision_params(n, m_min=27):
    """Find (k, m) with n = k^2*(m+10)-10, k>=2, m>=m_min."""
    results = []
    k = 2
    while k*k*(m_min+10) <= n + 10:
        if (n+10) % (k*k) == 0:
            m = (n+10)//(k*k) - 10
            if m >= m_min:
                results.append((k, m))
        k += 1
    return results


def _exact_geo_h0(h):
    ICO_FACES = [
        [0,1,8],[0,8,4],[0,4,5],[0,5,9],[0,9,1],
        [1,6,8],[8,6,10],[8,10,4],[4,10,2],[4,2,5],
        [5,2,11],[5,11,9],[9,11,7],[9,7,1],[1,7,6],
        [3,6,7],[3,7,11],[3,11,2],[3,2,10],[3,10,6],
    ]
    pts = []; vm = {}
    def av(pt):
        n2 = pt / np.linalg.norm(pt); k = tuple(np.round(n2, 14))
        if k not in vm: vm[k] = len(pts); pts.append(n2)
        return vm[k]
    tris = []
    for face in ICO_FACES:
        A, B, C = _ICO[face[0]], _ICO[face[1]], _ICO[face[2]]
        L = {}
        for r in range(h+1):
            for s in range(h+1-r): L[(r,s)] = av(r*A + s*B + (h-r-s)*C)
        for r in range(h):
            for s in range(h-r):
                tris.append([L[(r,s)], L[(r+1,s)], L[(r,s+1)]])
                if r+s+2 <= h: tris.append([L[(r+1,s)], L[(r+1,s+1)], L[(r,s+1)]])
    return np.array(pts), tris


def _dual_of_catalan(V, F):
    nV = len(V)
    ef = defaultdict(list)
    for ti, f in enumerate(F):
        k = len(f)
        for j in range(k):
            e = (min(f[j], f[(j+1)%k]), max(f[j], f[(j+1)%k]))
            ef[e].append(ti)
    vstar = defaultdict(list)
    for ti, f in enumerate(F):
        for v in f: vstar[v].append(ti)
    dv = np.array([V[f].mean(0) for f in F])
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    dual_F = []
    for v in range(nV):
        star = vstar.get(v, [])
        if len(star) < 3: continue
        nv = V[v]; ax = np.argmin(np.abs(nv))
        x = np.zeros(3); x[ax] = 1.0; x -= np.dot(x,nv)*nv; x /= np.linalg.norm(x)
        y = np.cross(nv, x)
        angs = []
        for fi in star:
            c = dv[fi] - nv; c -= np.dot(c,nv)*nv
            angs.append((np.arctan2(np.dot(c,y), np.dot(c,x)), fi))
        angs.sort(); ordered = [fi for _,fi in angs]
        fv = dv[ordered]; cen = fv.mean(0)
        if np.dot(np.cross(fv[1]-fv[0], fv[2]-fv[0]), cen) < 0:
            ordered = ordered[::-1]
        dual_F.append(ordered)
    return dv, dual_F


def _fan_triangulate(V, F):
    new_V = [v / np.linalg.norm(v) for v in V]; new_F = []
    for f in F:
        if len(f) == 3:
            new_F.append(list(f))
        else:
            cen = np.array(new_V)[f].mean(0)
            cen /= np.linalg.norm(cen)
            ci = len(new_V); new_V.append(cen)
            k = len(f)
            for j in range(k): new_F.append([f[j], f[(j+1)%k], ci])
    return np.array(new_V), new_F


def _subdivide(V, F, k):
    vm = {}; new_V = []
    def gv(p):
        n = p / np.linalg.norm(p); key = tuple(np.round(n, 10))
        if key not in vm: vm[key] = len(new_V); new_V.append(n)
        return vm[key]
    new_F = []
    for tri in F:
        A, B, C = V[tri[0]], V[tri[1]], V[tri[2]]
        L = {}
        for i in range(k+1):
            for j in range(k+1-i): L[(i,j,k-i-j)] = gv(i*A+j*B+(k-i-j)*C)
        for i in range(k):
            for j in range(k-i):
                l = k-1-i-j
                new_F.append([L[(i+1,j,l)], L[(i,j+1,l)], L[(i,j,l+1)]])
                if i+j+1 < k: new_F.append([L[(i+1,j,l)], L[(i+1,j+1,l-1)], L[(i,j+1,l)]])
    return np.array(new_V), new_F


def merge_pyramids(pts, tris):
    """Replace deg-4 and deg-5 pole vertices with quad/pentagon faces."""
    deg = defaultdict(int)
    for t in tris:
        for v in t: deg[v] += 1

    deg5 = [v for v, d in deg.items() if d == 5]
    deg4 = [v for v, d in deg.items() if d == 4]

    euler = 2*len(deg4) + len(deg5)
    if euler != 12 and len(deg4) == 0 and len(deg5) > 12:
        cand = pts[deg5]; used = set(); sel = []
        for iv in _ICO:
            ds = np.linalg.norm(cand - iv, axis=1)
            for idx in np.argsort(ds):
                if deg5[idx] not in used:
                    used.add(deg5[idx]); sel.append(deg5[idx]); break
        deg5 = sel
    elif euler != 12:
        print(f'  WARNING: Euler budget = {euler} (expected 12)')

    vstar = defaultdict(list)
    for ti, t in enumerate(tris):
        for v in t: vstar[v].append(ti)

    def ordered_ring(cap, star_tris):
        nv = pts[cap]; ax = np.argmin(np.abs(nv))
        x = np.zeros(3); x[ax] = 1.0; x -= np.dot(x,nv)*nv; x /= np.linalg.norm(x)
        y = np.cross(nv, x)
        angs = []
        for ti in star_tris:
            others = [v for v in tris[ti] if v != cap]
            cen = pts[others].mean(0) - nv; cen -= np.dot(cen,nv)*nv
            angs.append((np.arctan2(np.dot(cen,y), np.dot(cen,x)), ti))
        angs.sort(); ordered = [ti for _,ti in angs]
        ring = []
        f0 = tris[ordered[0]]; o0 = [v for v in f0 if v != cap]
        p0, p1 = pts[o0[0]], pts[o0[1]]
        mid = (p0+p1)/2 - nv; mid -= np.dot(mid,nv)*nv
        ed = p1-p0; ed -= np.dot(ed,nv)*nv
        if np.dot(np.cross(nv,ed), mid) < 0: o0 = o0[::-1]
        ring.append(o0[0])
        for ti in ordered[1:]:
            new = [v for v in tris[ti] if v != cap and v not in ring]
            if new: ring.append(new[0])
        return ring, ordered

    removed = set(); polys = []

    for cap in deg5:
        ring, ordered = ordered_ring(cap, vstar[cap])
        if len(ring) != 5:
            ring = []
            for ti in ordered:
                for v in tris[ti]:
                    if v != cap and v not in ring: ring.append(v)
            ring = ring[:5]
        polys.append(ring)
        for ti in vstar[cap]: removed.add(ti)

    for cap in deg4:
        ring, ordered = ordered_ring(cap, vstar[cap])
        if len(ring) != 4:
            ring = []
            for ti in ordered:
                for v in tris[ti]:
                    if v != cap and v not in ring: ring.append(v)
            ring = ring[:4]
        polys.append(ring)
        for ti in vstar[cap]: removed.add(ti)

    caps = set(deg5) | set(deg4); remap = {}; ni = 0
    for v in range(len(pts)):
        if v not in caps: remap[v] = ni; ni += 1
    new_pts = np.array([pts[v] for v in range(len(pts)) if v not in caps])
    new_tris = [[remap[v] for v in t] for i, t in enumerate(tris) if i not in removed]
    new_polys = [[remap[v] for v in p] for p in polys]
    return new_pts, new_tris + new_polys


def take_dual(pts, tris):
    n = len(pts)
    dv = np.array([pts[t].mean(0) for t in tris])
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    vstar = defaultdict(list)
    for ti, t in enumerate(tris):
        for v in t: vstar[v].append(ti)
    dual_f = []
    for v in range(n):
        star = vstar.get(v, [])
        if len(star) < 3: continue
        nv = pts[v]; ax = np.argmin(np.abs(nv))
        x = np.zeros(3); x[ax] = 1.0; x -= np.dot(x,nv)*nv; x /= np.linalg.norm(x)
        y = np.cross(nv, x)
        angs = []
        for fi in star:
            c = dv[fi] - nv; c -= np.dot(c,nv)*nv
            angs.append((np.arctan2(np.dot(c,y), np.dot(c,x)), fi))
        angs.sort(); ordered = [fi for _,fi in angs]
        fv = dv[ordered]; cen = fv.mean(0)
        if np.dot(np.cross(fv[1]-fv[0], fv[2]-fv[0]), cen) < 0:
            ordered = ordered[::-1]
        dual_f.append(ordered)
    return dv, dual_f


def stabilise(V, F, area_pull=0.20, verbose=True):
    V = _flatten(V, F, verbose)
    V, F = _midsphere(V, F, verbose)
    V, F = _area_opt(V, F, area_pull, verbose)
    return V, F


def _flatten(V, F, verbose=True):
    nV = len(V)
    by_sz = defaultdict(list)
    for f in F: by_sz[len(f)].append(f)
    groups = [np.array(fs, dtype=int) for fs in by_sz.values() if fs]

    def plan_obj(x, w):
        V_ = x.reshape(nV, 3); G = np.zeros_like(V_); val = 0.0
        for idx in groups:
            if len(idx) == 0: continue
            k = idx.shape[1]; pts = V_[idx]
            cen = pts.mean(1, keepdims=True); d = pts - cen
            dn = np.roll(d, -1, axis=1); raw = np.cross(d, dn).sum(1)
            area = np.linalg.norm(raw, axis=1) / 2
            nh = raw / (2*area[:,None]+1e-30)
            proj = np.einsum('nki,ni->nk', d, nh)
            val += w * float((proj**2).sum())
            np.add.at(G, idx.reshape(-1),
                      (2*w*proj[:,:,None]*nh[:,None,:]).reshape(-1, 3))
        return float(val), G.flatten()

    Vm = V.copy()
    mp = float('inf')
    for w in [1e3, 1e7, 1e11]:
        res = minimize(lambda x: plan_obj(x, w), Vm.flatten(), jac=True,
                       method='L-BFGS-B',
                       options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-10})
        Vm = res.x.reshape(nV, 3)
        mp = max((np.linalg.svd(Vm[f]-Vm[f].mean(0), full_matrices=False)[1][-1]
                  for f in F if len(f) >= 4), default=0)
        if verbose:
            print(f'  planarity w={w:.0e}: {mp:.2e}')
        if mp < 1e-11: break
    return Vm


def _midsphere(V, F, verbose=True):
    nV = len(V)
    edges = set()
    for f in F:
        k = len(f)
        for j in range(k): edges.add((min(f[j],f[(j+1)%k]), max(f[j],f[(j+1)%k])))
    edges = list(edges)
    ea = np.array([e[0] for e in edges]); eb = np.array([e[1] for e in edges])

    by_sz = defaultdict(list)
    for f in F:
        if len(f) >= 4: by_sz[len(f)].append(f)
    groups = {k: np.array(v, dtype=int) for k, v in by_sz.items()}

    def edge_d2(V_):
        u = V_[ea]; v_ = V_[eb]; d = v_-u
        dd = np.einsum('ij,ij->i', d, d).clip(1e-30)
        t = -np.einsum('ij,ij->i', u, d) / dd
        foot = u + t[:,None]*d
        return np.einsum('ij,ij->i', foot, foot)

    def recenter(V_):
        u = V_[ea]; v_ = V_[eb]; d = v_-u
        dd = np.einsum('ij,ij->i', d, d).clip(1e-30)
        t = -np.einsum('ij,ij->i', u, d) / dd
        return V_ - (u + t[:,None]*d).mean(0)

    r = np.sqrt(np.median(edge_d2(V)))
    if r > 1e-10: V = V / r
    V = recenter(V)

    for rnd, w_flat in enumerate([0, 1e2, 1e6, 1e10, 1e14, 1e16]):
        def obj(x, _w=w_flat):
            V_ = x.reshape(nV, 3); G = np.zeros_like(V_); val = 0.0
            u = V_[ea]; v_ = V_[eb]; d = v_-u
            dd = np.einsum('ij,ij->i', d, d).clip(1e-30)
            t = -np.einsum('ij,ij->i', u, d) / dd
            foot = u + t[:,None]*d; dist2 = np.einsum('ij,ij->i', foot, foot)
            res = dist2 - 1.0; val += float(np.sum(res**2))
            g_u = 2.0*res[:,None]*(1.0-t)[:,None]*foot*2
            g_v = 2.0*res[:,None]*t[:,None]*foot*2
            np.add.at(G, ea, g_u); np.add.at(G, eb, g_v)
            if _w > 0:
                for sz, idx in groups.items():
                    k = idx.shape[1]; pts = V_[idx]
                    cen = pts.mean(1, keepdims=True); d2 = pts-cen
                    dn = np.roll(d2,-1,axis=1); raw = np.cross(d2,dn).sum(1)
                    ar = np.linalg.norm(raw,axis=1)/2
                    nh = raw/(2*ar[:,None]+1e-30)
                    proj = np.einsum('nki,ni->nk', d2, nh)
                    val += _w*float((proj**2).sum())
                    np.add.at(G, idx.reshape(-1),
                              (2*_w*proj[:,:,None]*nh[:,None,:]).reshape(-1,3))
            return float(val), G.flatten()
        res = minimize(obj, V.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-20, 'gtol': 1e-12})
        V = res.x.reshape(nV, 3); V = recenter(V)
        d2 = edge_d2(V); r = np.sqrt(d2.mean())
        if r > 1e-10: V = V / r
        if verbose:
            pls = [np.linalg.svd(V[f]-V[f].mean(0),full_matrices=False)[1][-1]
                   for f in F if len(f)>=4]
            print(f'  midsphere {rnd+1}: edge_std={np.sqrt(edge_d2(V)).std():.2e}  '
                  f'plan={max(pls) if pls else 0:.2e}')

    F_out = []
    for f in F:
        pts = V[f]; cen = pts.mean(0); d_ = pts-cen
        dn = np.roll(d_,-1,axis=0); n = np.cross(d_,dn).sum(0)
        if np.dot(n,cen) < 0: f = f[::-1]
        F_out.append(list(f))
    return V, F_out


def _poly_area_grad(pts):
    k = len(pts); cen = pts.mean(0); d = pts-cen
    dn = np.roll(d,-1,axis=0); raw = np.cross(d,dn).sum(0)
    nl = np.linalg.norm(raw)+1e-30; area=nl/2; nh=raw/nl
    grad = np.array([np.cross(nh, pts[(j+1)%k]-pts[(j-1)%k])/2 for j in range(k)])
    return area, grad


def _area_opt(V, F, area_pull=0.20, verbose=True):
    nV = len(V); nF = len(F)
    edges = set()
    for f in F:
        k = len(f)
        for j in range(k): edges.add((min(f[j],f[(j+1)%k]),max(f[j],f[(j+1)%k])))
    edges = list(edges)
    ea = np.array([e[0] for e in edges]); eb = np.array([e[1] for e in edges])

    by_sz = defaultdict(list)
    for f in F: by_sz[len(f)].append(f)
    groups = {k: np.array(v,dtype=int) for k,v in by_sz.items() if k>=4}

    def combined(x, wm, wa, wf, ws):
        V_ = x.reshape(nV,3); G = np.zeros_like(V_); val = 0.0
        u=V_[ea]; v_=V_[eb]; d=v_-u
        dd=np.einsum('ij,ij->i',d,d).clip(1e-30)
        t=-np.einsum('ij,ij->i',u,d)/dd; foot=u+t[:,None]*d
        dist2=np.einsum('ij,ij->i',foot,foot); r2=dist2.mean()
        res_m=dist2-r2; val+=wm*float(np.sum(res_m**2))
        np.add.at(G,ea,2*wm*res_m[:,None]*2*(1-t)[:,None]*foot)
        np.add.at(G,eb,2*wm*res_m[:,None]*2*t[:,None]*foot)
        areas=np.zeros(nF); ags=[]
        for fi,f in enumerate(F):
            a,ag=_poly_area_grad(V_[f]); areas[fi]=a; ags.append(ag)
        am=areas.mean(); res_a=areas-am; val+=wa*float(np.sum(res_a**2))
        for fi,f in enumerate(F):
            if abs(res_a[fi])<1e-15: continue
            for j,vi in enumerate(f): G[vi]+=(2*wa*res_a[fi])*ags[fi][j]
        for sz,idx in groups.items():
            k=idx.shape[1]; pts=V_[idx]; cen=pts.mean(1,keepdims=True)
            d2=pts-cen; dn=np.roll(d2,-1,axis=1); raw=np.cross(d2,dn).sum(1)
            ar=np.linalg.norm(raw,axis=1)/2; nh=raw/(2*ar[:,None]+1e-30)
            proj=np.einsum('nki,ni->nk',d2,nh); val+=wf*float((proj**2).sum())
            np.add.at(G,idx.reshape(-1),(2*wf*proj[:,:,None]*nh[:,None,:]).reshape(-1,3))
        if ws>0:
            for f in F:
                k=len(f); pts=V_[f]
                sides=np.array([np.linalg.norm(pts[(j+1)%k]-pts[j]) for j in range(k)])
                L=sides.mean(); res_s=sides-L; val+=ws*float(np.sum(res_s**2))
                for j in range(k):
                    diff=V_[f[(j+1)%k]]-V_[f[j]]; ln=np.linalg.norm(diff)+1e-15
                    G[f[j]]+=(2*ws*res_s[j])*(-diff/ln); G[f[(j+1)%k]]+=(2*ws*res_s[j])*(diff/ln)
        return float(val), G.flatten()

    areas_init = np.array([_poly_area_grad(V[f])[0] for f in F])
    if areas_init.std()/areas_init.mean() < 1e-6:
        if verbose: print('  Areas already equal, skipping area optimisation')
        return V, F

    wm = 1e12; wa = area_pull*1e11; ws = 1e8
    Vm = V.copy()

    if verbose:
        areas=np.array([_poly_area_grad(Vm[f])[0] for f in F])
        u=Vm[ea]; v_=Vm[eb]; d=v_-u
        dd=np.einsum('ij,ij->i',d,d).clip(1e-30)
        t=-np.einsum('ij,ij->i',u,d)/dd; foot=Vm[ea]+t[:,None]*d
        mid=np.sqrt(np.einsum('ij,ij->i',foot,foot))
        print(f'  Before: mid_CV={mid.std()/mid.mean()*100:.3f}%  '
              f'area_CV={areas.std()/areas.mean()*100:.2f}%')

    for si, (wf, mi) in enumerate([(1e13,2000),(1e14,3000),(1e15,3000),(1e16,3000)]):
        res = minimize(lambda x: combined(x,wm,wa,wf,ws), Vm.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter':mi,'ftol':1e-21,'gtol':1e-14})
        Vm = res.x.reshape(nV,3)
        if verbose:
            areas=np.array([_poly_area_grad(Vm[f])[0] for f in F])
            u=Vm[ea]; v_=Vm[eb]; d=v_-u
            dd=np.einsum('ij,ij->i',d,d).clip(1e-30)
            t=-np.einsum('ij,ij->i',u,d)/dd; foot=Vm[ea]+t[:,None]*d
            mid=np.sqrt(np.einsum('ij,ij->i',foot,foot))
            print(f'  Stage {si+1}: mid_CV={mid.std()/mid.mean()*100:.3f}%  '
                  f'area_CV={areas.std()/areas.mean()*100:.2f}%')

    F_out = []
    for f in F:
        pts=Vm[f]; cen=pts.mean(0); d_=pts-cen
        dn=np.roll(d_,-1,axis=0); n=np.cross(d_,dn).sum(0)
        if np.dot(n,cen)<0: f=f[::-1]
        F_out.append(list(f))
    return Vm, F_out


_SEEDS = [42,7,1,13,53,79,97,101,23,61,37,83,17,31,41,43,47,59,67,71,
          73,89,103,107,109,113,127,131,137,139,149,151,157,163,167,173,
          179,181,191,193,197,199,211,223,227,229,233,239,241,251]


def _search_primal(n_primal, target_score=0, max_time=300, verbose=True):
    t0 = time.time()
    best_sc = float('inf'); best_pts = None; best_tris = None

    phase1_budget = min(max_time * 0.30, 45.0)
    seed_scores = []

    for seed in _SEEDS:
        if time.time() - t0 > phase1_budget:
            break
        ts_iters = min(700, max(400, n_primal * 2))
        pts = fibonacci_sphere(n_primal, seed=seed)
        pts = thomson(pts, ts_iters)
        tris = triangulate(pts)
        rng = np.random.RandomState(seed)
        for _ in range(4):
            tris, sc, deg = flip_pass(n_primal, tris, rng)
        seed_scores.append((sc, seed, pts.copy(), list(tris)))
        if verbose:
            print(f'  [scan] seed={seed:3d}: score={sc}')
        if sc <= target_score:
            if verbose: print(f'  -> score=0 found in Phase 1!')
            return pts, tris, sc

    seed_scores.sort(key=lambda x: x[0])
    if verbose:
        print(f'  Phase 1 done ({time.time()-t0:.0f}s). '
              f'Best: {seed_scores[0][0]} (seed {seed_scores[0][1]}). '
              f'Deep-diving top seeds...')

    n_deep = min(6, len(seed_scores))
    for sc_init, seed, pts_init, tris_init in seed_scores[:n_deep]:
        if time.time() - t0 > max_time:
            if verbose: print(f'  (time limit {max_time}s reached)')
            break

        pts = pts_init.copy(); tris = list(tris_init)
        rng = np.random.RandomState(seed)

        for pi in range(40):
            if time.time() - t0 > max_time: break
            tris, sc, deg = flip_pass(n_primal, tris, rng)
            if sc < best_sc:
                best_sc = sc; best_pts = pts.copy(); best_tris = list(tris)
                if verbose:
                    print(f'  [deep] seed={seed:3d} iter={pi}: score={sc}  '
                          f'deg={dict(Counter(deg.values()))}')
            if best_sc <= target_score:
                break
            pts = spring_relax(pts, tris, 200)
            tris = triangulate(pts)
            if pi > 20 and sc == best_sc: break

        if best_sc <= target_score:
            break

    return best_pts, best_tris, best_sc


def build_via_subdivision(m, k, area_pull=0.20, verbose=True):
    """Build n=k^2*(m+10)-10 faced Catalan via subdivision of an m-faced base."""
    n = k*k*(m+10) - 10
    if verbose: print(f'  Subdivision: m={m}, k={k} -> n={n}')

    if verbose: print(f'  [1/4] Building base m={m}...')
    V_m, F_m = _build_small_catalan(m, verbose=verbose)

    if verbose: print(f'  [2/4] Dual of m={m} Catalan...')
    V_d, F_d = _dual_of_catalan(V_m, F_m)

    if verbose: print(f'  [3/4] Fan-triangulate + {k}-subdivision...')
    V_d, F_d = _fan_triangulate(V_d, F_d)
    V_s, F_s = _subdivide(V_d, F_d, k)

    if verbose: print(f'  [4/4] Merge pyramids + dual...')
    V_s, F_s = merge_pyramids(V_s, F_s)
    V_n, F_n = take_dual(V_s, F_s)

    fc = Counter(len(f) for f in F_n)
    if verbose: print(f'  Result: {len(F_n)} faces: {dict(fc)}')
    return V_n, F_n


def _build_small_catalan(m, verbose=False):
    h = 2
    while 10*h*h - 10 < m: h += 1
    if 10*h*h - 10 == m:
        V_geo, tris_geo = _exact_geo_h0(h)
        pts_m, tris_m = merge_pyramids(V_geo, tris_geo)
        return take_dual(pts_m, tris_m)

    sub = find_subdivision_params(m)
    if sub:
        k2, m2 = sub[0]
        if verbose:
            print(f'  [base] m={m} -> recursive sub k={k2}, m={m2}')
        return build_via_subdivision(m2, k2, area_pull=0.0, verbose=verbose)

    n_primal_m = m + 12
    max_t = max(60, min(600, int(60 * (n_primal_m / 49) ** 2)))
    if verbose and n_primal_m > 200:
        print(f'  [base] Thomson at n={n_primal_m} (budget={max_t}s)...')
    pts, tris, sc = _search_primal(n_primal_m, target_score=0,
                                    max_time=max_t, verbose=verbose)
    if sc > 0 and verbose:
        print(f'  WARNING: base m={m} has score={sc}')
    pts_m, tris_m = merge_pyramids(pts, tris)
    return take_dual(pts_m, tris_m)


_COLORS = {
    'hexagon':   (180, 180, 190),
    'shield':    ( 60,  80, 180),
    'rhombus':   (180,  40,  40),
    'trapezoid': ( 40, 160,  60),
    'other':     (220, 120,  30),
}


def _classify_face(V, face):
    k = len(face)
    if k == 6: return 'hexagon'
    if k == 5: return 'shield'
    if k == 4:
        pts = V[face]
        edges = [np.linalg.norm(pts[(j+1)%4]-pts[j]) for j in range(4)]
        if max(edges)/max(min(edges),1e-10) < 1.08: return 'rhombus'
        return 'trapezoid'
    return 'other'


def write_off_colored(path, V, F):
    types = [_classify_face(V, f) for f in F]
    cnt = Counter(types)
    areas = []
    for f in F:
        pts = V[f]; cen = pts.mean(0); d = pts-cen
        dn = np.roll(d,-1,axis=0)
        areas.append(np.linalg.norm(np.cross(d,dn).sum(0))/2)
    total = sum(areas)
    n = len(F); scale = np.sqrt(n / total) if total > 0 else 1.0
    V_scaled = V * scale

    with open(path, 'w') as fh:
        fh.write(f'OFF\n{len(V_scaled)} {len(F)} 0\n')
        for v in V_scaled:
            fh.write(f'  {v[0]:.16f}  {v[1]:.16f}  {v[2]:.16f}\n')
        for f, t in zip(F, types):
            r, g, b = _COLORS.get(t, _COLORS['other'])
            fh.write(str(len(f)) + ' ' + ' '.join(map(str, f)) +
                     f'  {r} {g} {b}\n')
    return cnt


def quality_report(V, F):
    edges = {}
    for f in F:
        k = len(f)
        for j in range(k):
            e = (min(f[j],f[(j+1)%k]),max(f[j],f[(j+1)%k]))
            if e not in edges: edges[e]=np.linalg.norm(V[e[0]]-V[e[1]])
    elen = np.array(list(edges.values()))
    areas = []
    for f in F:
        pts=V[f]; cen=pts.mean(0); d=pts-cen
        dn=np.roll(d,-1,axis=0)
        areas.append(np.linalg.norm(np.cross(d,dn).sum(0))/2)
    areas = np.array(areas)
    pls = []
    for f in F:
        if len(f)>=4:
            _, s, _ = np.linalg.svd(V[f]-V[f].mean(0), full_matrices=False)
            pls.append(s[-1])
    vdeg = Counter(v for f in F for v in f)
    bad = sum(1 for d in vdeg.values() if d not in (3,5))
    print(f'  Faces:      {dict(Counter(len(f) for f in F))}')
    print(f'  Vertex deg: {dict(Counter(vdeg.values()))}  bad={bad}')
    print(f'  Edge CV:    {elen.std()/elen.mean()*100:.3f}%  '
          f'min={elen.min():.4f}  max={elen.max():.4f}')
    print(f'  Area CV:    {areas.std()/areas.mean()*100:.3f}%')
    print(f'  Planarity:  {max(pls) if pls else 0:.2e}')
    return bad


def run(n, out_path=None, area_pull=0.20, no_stabilise=False, verbose=True):
    if n < 27:
        print(f'ERROR: n must be at least 27. Got n={n}.')
        sys.exit(1)

    t0 = time.time()

    if out_path is None:
        desk = os.path.join(os.path.expanduser('~'), 'Desktop')
        out_path = os.path.join(desk if os.path.isdir(desk) else '.', f'catalan_{n}.off')

    print(f'\n{"="*62}')
    print(f'  Catalan-like polyhedron: n={n}')
    print(f'{"="*62}')

    sub = find_subdivision_params(n)
    if sub:
        k, m = sub[0]
        print(f'\n[SUBDIVISION] k={k}, m={m}  ->  n={n} (exact)')
        V, F = build_via_subdivision(m, k, area_pull=area_pull, verbose=verbose)
        n_actual = len(F)
    else:
        anchor_n = None; anchor_k = 0; anchor_sub = None; split_ok = False
        for delta in range(1, 9):
            sub_try = find_subdivision_params(n - delta)
            if sub_try:
                anchor_n = n - delta; anchor_k = delta; anchor_sub = sub_try[0]
                break

        anchor2_n = None; anchor2_sub = None
        for delta in range(1, 9):
            sub_try2 = find_subdivision_params(n + delta)
            if sub_try2 and delta <= 4:
                anchor2_n = n + delta; anchor2_sub = sub_try2[0]
                break

        max_time = 300

        if anchor_n is not None:
            k_sub, m_sub = anchor_sub
            anchor_label = f"n'={anchor_n} (k={k_sub},m={m_sub})"
            if anchor2_n is not None:
                k2, m2 = anchor2_sub
                anchor_label += f"; backup n''={anchor2_n} (k={k2},m={m2})"
            print(f'\n[ANCHOR] {anchor_label},  {anchor_k} split(s) -> n={n}')

            def _build_anchor_primal(n_anch, sub_params, verbose=False):
                k_a, m_a = sub_params
                V_a, F_a = build_via_subdivision(m_a, k_a, area_pull=area_pull,
                                                  verbose=verbose)
                V_da, F_da = _dual_of_catalan(V_a, F_a)
                V_pta, F_pta = _fan_triangulate(V_da, F_da)
                pts_a = np.array(V_pta)
                tris_a = [list(f) for f in F_pta]
                _, dm_a, _ = _build_mesh(tris_a)
                return pts_a, tris_a, dm_a

            pts_anchor, tris_anchor, dm_a = _build_anchor_primal(
                anchor_n, anchor_sub, verbose=verbose)
            ef_anchor, dm_a_check, _ = _build_mesh(tris_anchor)
            assert _score(dm_a_check, ef_anchor) == 0, "anchor primal must be score=0"
            if verbose:
                print(f'  Anchor primal: {len(pts_anchor)} verts, '
                      f'degrees={dict(Counter(dm_a.values()))}')

            deg_a = dict(dm_a)
            d5_set = set(v for v, d in deg_a.items() if d == 5)
            d5_pos = pts_anchor[list(d5_set)]
            ef_dict0 = defaultdict(list)
            for ti, t in enumerate(tris_anchor):
                for j in range(3):
                    e = (min(t[j], t[(j+1)%3]), max(t[j], t[(j+1)%3]))
                    ef_dict0[e].append(ti)
            hh0 = [(e, fis) for e, fis in ef_dict0.items()
                   if deg_a.get(e[0], 0)==6 and deg_a.get(e[1], 0)==6 and len(fis)==2]

            def _pdist(edge, pts=pts_anchor, d5p=d5_pos):
                mid = (pts[edge[0]] + pts[edge[1]]) / 2
                mid /= np.linalg.norm(mid) + 1e-12
                return float(np.min(np.linalg.norm(d5p - mid, axis=1)))

            hh0.sort(key=lambda x: -_pdist(x[0]))

            def _do_split(pts_in, tris_in, e, fis):
                a, b = e; ti0, ti1 = fis
                t0f = tris_in[ti0]; t1f = tris_in[ti1]
                cf = [v for v in t0f if v != a and v != b]
                df = [v for v in t1f if v != a and v != b]
                if not cf or not df: return None, None
                vm = (pts_in[a] + pts_in[b]) / 2
                nm = np.linalg.norm(vm)
                if nm < 1e-10: return None, None
                vm /= nm; vi = len(pts_in)
                pts_out = np.vstack([pts_in, vm[None]])
                tris_out = [t for i, t in enumerate(tris_in) if i not in (ti0, ti1)]
                tris_out += [[a,vi,cf[0]], [vi,b,cf[0]], [a,df[0],vi], [vi,df[0],b]]
                return pts_out, tris_out

            def _flip_clean(pts_in, tris_in, n_primal, seed, n_iter=40):
                best_sc = float('inf')
                best_pts = pts_in.copy(); best_tris = list(tris_in)
                if n_primal < 450:
                    offsets = [0, 3, 7, 13]
                elif n_primal < 650:
                    offsets = [0, 7]
                else:
                    offsets = [0]
                for s_offset in offsets:
                    rng = np.random.RandomState(seed + s_offset)
                    pts_w = pts_in.copy(); tris_w = list(tris_in)
                    for _ in range(n_iter):
                        tris_w, sc, dm = flip_pass(n_primal, tris_w, rng)
                        if sc < best_sc:
                            best_sc = sc
                            best_pts = pts_w.copy(); best_tris = list(tris_w)
                        if best_sc == 0: break
                        pts_w = spring_relax(pts_w, tris_w, 200)
                        try: tris_w = triangulate(pts_w)
                        except Exception: break
                    if best_sc == 0: break
                return best_pts, best_tris, best_sc

            def _try_splits_from_anchor(pts_a, tris_a, hh_a, k_needed, n_target,
                                         label=""):
                pts_cur = pts_a.copy(); tris_cur = list(tris_a)

                if k_needed == 1:
                    for ei, (e_s, fis_s) in enumerate(hh_a):
                        pts_t, tris_t = _do_split(pts_cur, tris_cur, e_s, fis_s)
                        if pts_t is None: continue
                        bp, bt, bs = _flip_clean(pts_t, tris_t, len(pts_t), ei*7)
                        if bs == 0 and (len(bp)-12) == n_target:
                            if verbose:
                                print(f'  {label}Split 1 edge {ei}: score=0 \u2713')
                            return bp, bt, True
                    return None, None, False

                elif k_needed == 2:
                    for ei1, (e1, fis1) in enumerate(hh_a):
                        pts_t1, tris_t1 = _do_split(pts_cur, tris_cur, e1, fis1)
                        if pts_t1 is None: continue
                        bp1, bt1, bs1 = _flip_clean(pts_t1, tris_t1, len(pts_t1),
                                                     ei1*7, n_iter=20)
                        if bs1 > 300: continue

                        ef_t1 = defaultdict(list)
                        for ti, t in enumerate(bt1):
                            for j in range(3):
                                e = (min(t[j],t[(j+1)%3]),max(t[j],t[(j+1)%3]))
                                ef_t1[e].append(ti)
                        dm_t1 = defaultdict(int)
                        for t in bt1:
                            for v in t: dm_t1[v]+=1
                        d5_t1 = set(v for v,d in dm_t1.items() if d==5)
                        d5p_t1 = bp1[list(d5_t1)] if d5_t1 else d5_pos
                        hh1 = [(e,fis) for e,fis in ef_t1.items()
                               if dm_t1.get(e[0],0)==6 and dm_t1.get(e[1],0)==6
                               and len(fis)==2]
                        hh1.sort(key=lambda x: -float(
                            np.min(np.linalg.norm(d5p_t1 -
                                   (bp1[x[0][0]]+bp1[x[0][1]])/2, axis=1))))

                        for ei2, (e2, fis2) in enumerate(hh1):
                            pts_t2, tris_t2 = _do_split(bp1, bt1, e2, fis2)
                            if pts_t2 is None: continue
                            bp2, bt2, bs2 = _flip_clean(
                                pts_t2, tris_t2, len(pts_t2), ei2*7)
                            if bs2 == 0 and (len(bp2)-12) == n_target:
                                if verbose:
                                    print(f'  {label}Splits ({ei1},{ei2}): score=0 \u2713')
                                return bp2, bt2, True
                    return None, None, False

                else:
                    for split_idx in range(k_needed):
                        ef_c = defaultdict(list)
                        for ti, t in enumerate(tris_cur):
                            for j in range(3):
                                e = (min(t[j],t[(j+1)%3]),max(t[j],t[(j+1)%3]))
                                ef_c[e].append(ti)
                        dm_c = defaultdict(int)
                        for t in tris_cur:
                            for v in t: dm_c[v]+=1
                        d5c = set(v for v,d in dm_c.items() if d==5)
                        d5p_c = pts_cur[list(d5c)] if d5c else d5_pos
                        hh_c = [(e,fis) for e,fis in ef_c.items()
                                if dm_c.get(e[0],0)==6 and dm_c.get(e[1],0)==6
                                and len(fis)==2]
                        hh_c.sort(key=lambda x: -float(
                            np.min(np.linalg.norm(d5p_c -
                                   (pts_cur[x[0][0]]+pts_cur[x[0][1]])/2,axis=1))))
                        is_last = (split_idx == k_needed - 1)
                        found = False
                        for ei, (e_s, fis_s) in enumerate(hh_c):
                            pts_t, tris_t = _do_split(pts_cur, tris_cur, e_s, fis_s)
                            if pts_t is None: continue
                            n_it = 40 if is_last else 20
                            bp, bt, bs = _flip_clean(pts_t, tris_t, len(pts_t),
                                                     ei*7, n_it)
                            if is_last:
                                if bs == 0 and (len(bp)-12) == n_target:
                                    if verbose:
                                        print(f'  {label}Split {split_idx+1} edge {ei}: \u2713')
                                    pts_cur = bp; tris_cur = bt; found = True; break
                                if bs == 50:
                                    ef_r, dm_r, _ = _build_mesh(bt)
                                    for rrs in [42, 7, 1, 13]:
                                        rng_r = np.random.RandomState(rrs)
                                        pr, tr, sr = vertex_split_repair(
                                            bp, bt, dict(dm_r), rng_r)
                                        if sr == 0 and (len(pr)-12) == n_target:
                                            if verbose:
                                                print(f'  {label}Split {split_idx+1} ' +
                                                      f'edge {ei}+repair \u2713')
                                            pts_cur = pr; tris_cur = tr
                                            found = True; break
                                    if found: break
                            else:
                                if bs <= 150:
                                    pts_cur = bp; tris_cur = bt; found = True
                                    if verbose:
                                        print(f'  {label}Split {split_idx+1} edge {ei}: ' +
                                              f'score={bs} (ok)')
                                    break
                        if not found: break
                    ef_f, dm_f, _ = _build_mesh(tris_cur)
                    if _score(dm_f, ef_f) == 0 and (len(pts_cur)-12) == n_target:
                        return pts_cur, tris_cur, True
                    return None, None, False

            pts_final, tris_final, split_ok = _try_splits_from_anchor(
                pts_anchor, tris_anchor, hh0, anchor_k, n, label="")

            if not split_ok and anchor2_n is not None:
                if verbose:
                    k2, m2 = anchor2_sub
                    print(f'  Primary failed. Trying backup anchor n\'\'={anchor2_n} '
                          f'(k={k2},m={m2}), {anchor_k} splits...')
                pts_a2, tris_a2, dm_a2 = _build_anchor_primal(
                    anchor2_n, anchor2_sub, verbose=False)
                ef_a2, dm_a2c, _ = _build_mesh(tris_a2)
                if _score(dm_a2c, ef_a2) == 0:
                    d5_a2 = set(v for v,d in dm_a2.items() if d==5)
                    d5p_a2 = pts_a2[list(d5_a2)]
                    ef_d2 = defaultdict(list)
                    for ti, t in enumerate(tris_a2):
                        for j in range(3):
                            e=(min(t[j],t[(j+1)%3]),max(t[j],t[(j+1)%3]))
                            ef_d2[e].append(ti)
                    hh_a2 = [(e,fis) for e,fis in ef_d2.items()
                              if dm_a2.get(e[0],0)==6 and dm_a2.get(e[1],0)==6
                              and len(fis)==2]
                    hh_a2.sort(key=lambda x: -float(
                        np.min(np.linalg.norm(d5p_a2 -
                               (pts_a2[x[0][0]]+pts_a2[x[0][1]])/2,axis=1))))
                    pts_final, tris_final, split_ok = _try_splits_from_anchor(
                        pts_a2, tris_a2, hh_a2, anchor_k, n, label="[bk] ")

            if not split_ok and verbose:
                print(f'  Anchor strategy exhausted, falling back to pure Thomson...')

        if not split_ok:
            print(f'\n[THOMSON] Pure Thomson fallback for n={n}...')
            n_primal = n + 12
            pts_final, tris_final, sc_final = _search_primal(
                n_primal, target_score=50, max_time=max_time, verbose=verbose)

            if sc_final > 0 and pts_final is not None:
                ef_vs, dm_vs, _ = _build_mesh(tris_final)
                bad_now = [v for v, d in dm_vs.items() if d not in (4,5,6)]
                if bad_now:
                    rng_vs = np.random.RandomState(31337)
                    for _att in range(len(bad_now)*3):
                        ef_c, dm_c, _ = _build_mesh(tris_final)
                        if _score(dm_c, ef_c) == 0: break
                        bn = [v for v,d in dm_c.items() if d not in (4,5,6)]
                        if not bn: break
                        pr, tr, sr = vertex_split_repair(pts_final, tris_final, dict(dm_c), rng_vs, verbose=verbose)
                        if sr < sc_final:
                            sc_final = sr; pts_final = pr; tris_final = tr
                        else: break
                        if sc_final == 0: break

            n_actual_fallback = len(pts_final)-12 if pts_final is not None else n
            if n_actual_fallback != n:
                print(f'  NOTE: face count {n_actual_fallback} != requested {n}')

        n_actual = len(pts_final) - 12
        print(f'\n[DUAL] Building {n_actual}-faced Catalan-like...')
        pts_m, tris_m = merge_pyramids(pts_final, tris_final)
        V, F = take_dual(pts_m, tris_m)
        fc = Counter(len(f) for f in F)
        if verbose: print(f'  {len(F)} faces: {dict(fc)}')

    if no_stabilise:
        print('\n[SKIP] Stabilisation skipped (--no-stabilise)')
    else:
        print(f'\n[STABILISE] Planarity -> midsphere -> area...')
        V, F = stabilise(V, F, area_pull=area_pull, verbose=verbose)

    print(f'\n[QUALITY]')
    bad = quality_report(V, F)

    if bad > 0:
        print(f'  WARNING: {bad} invalid vertex degrees remain.')

    cnt = write_off_colored(out_path, V, F)
    print(f'\n[OUTPUT] {out_path}')
    print(f'  Colors:   shields={cnt.get("shield",0)}  '
          f'hexagons={cnt.get("hexagon",0)}  '
          f'rhombi={cnt.get("rhombus",0)}  '
          f'traps={cnt.get("trapezoid",0)}')
    if n_actual != n:
        print(f'  Faces: {n_actual} (requested {n} -- vertex-split adjusted)')
    print(f'  Time:  {time.time()-t0:.1f}s')

    return V, F


def main():
    parser = argparse.ArgumentParser(
        description='Generate Catalan-like polyhedra for n >= 27')
    parser.add_argument('n', type=int, nargs='?',
                        help='Number of faces')
    parser.add_argument('--output', '-o', help='Output .off file path')
    parser.add_argument('--area-pull', type=float, default=0.20,
                        help='Area equalisation strength 0-1 (default 0.20)')
    parser.add_argument('--no-stabilise', action='store_true',
                        help='Skip stabilisation steps (faster, lower quality)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    if args.n is None:
        while True:
            try:
                n = int(input('Number of faces n (>= 27): ').strip())
                if n >= 27: break
                print('  n must be at least 27.')
            except ValueError:
                print('  Please enter a whole number.')
    else:
        n = args.n

    run(n, out_path=args.output, area_pull=args.area_pull,
        no_stabilise=args.no_stabilise, verbose=not args.quiet)


if __name__ == '__main__':
    main()
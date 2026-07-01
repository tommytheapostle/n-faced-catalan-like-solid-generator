# A Generalization of Catalan Solids -- code and data

This repository contains the software and data supporting the paper. It lets one
(i) reproduce Table 1 by exhaustive search, and (ii) regenerate the optimal
polyhedra C(n) as OFF files.

## Contents

| file | purpose |
|------|---------|
| `catalan_search.py` | Exhaustive search: for each n, enumerate all IPR fullerenes C_{2n+20}, construct C(n), canonicalize, and report the minimizer of the face-area ratio rho. This is the computation behind Table 1. |
| `make_best_offs.py` | Regenerates the 30 optimal C(n) as OFF files from the tabulated optimal fullerenes, without needing buckygen. Shared geometry routines (construction, canonicalization) live here. |
| `best_offs/` | The 30 optimal C(n) in OFF format (midsphere-canonical, Newton-polished to ~1e-14). |

## Dependencies

- Python 3.9+, `numpy`, `networkx` (`pip install numpy networkx`)
- **buckygen**, the fullerene generator of Brinkmann, Goedgebeur and McKay, required
  only for `catalan_search.py`. Build it from source, e.g.

  ```
  git clone https://github.com/evanberkowitz/buckygen
  cd buckygen && make        # produces the ./buckygen binary
  ```

  Point the search at it with `--buckygen /path/to/buckygen` or the `BUCKYGEN`
  environment variable (default `./buckygen/buckygen`).

## Reproducing Table 1

```
python3 catalan_search.py --all
```

For a single row (and to write its OFF):

```
python3 catalan_search.py 40 --off out
```

For each n the program enumerates **every** IPR isomer of C_{2n+20} (buckygen is a
complete generator; the isomer counts match the published IPR counts), forms C(n) by
contracting the twelve pentagons to poles, discards any isomer with an isolated
hexagon, and canonicalizes **all** admissible isomers -- no pruning. The reported
rho_min(n) are therefore exhaustive minima over the admissible family. Each output
row lists n, the source fullerene, the point group, rho, the insphere ratio iota, and
the optimizer's face-size vector.

The three isohedral cases n = 20, 30, 60 return rho = 1 exactly (icosahedron, rhombic
triacontahedron, pentagonal hexecontahedron); the search reproduces this as a check.

## Regenerating the OFF files

```
python3 make_best_offs.py            # all n
python3 make_best_offs.py 29 40      # selected n
```

Output is written to `best_offs/`.

## Numerical notes

- **Canonicalization.** The midsphere (Koebe-Andreev-Thurston) form is computed by an
  alternating projection: planarize each face, drive each edge toward unit distance
  from the origin, recenter on the edge-sphere tangency barycenter. It is iterated to
  an edge-tangency residual below 1e-12.
- **Polishing.** The optimal coordinates are refined below 1e-14 by a Gauss-Newton step
  on the exact tangency and planarity constraints, using a complex-step Jacobian
  (machine-precision derivatives). This does not change rho; it only cleans the
  coordinates, so the deposited OFF files import as exactly planar with a genuine
  common midsphere.
- **Point groups** are obtained from the graph automorphisms, realized as isometries of
  the canonical form (Mani's theorem) and classified into the Schoenflies symbol.
- **Validation.** The canonicalizer returns rho = 1 exactly on the three isohedral
  members, and each reported optimum is independent of the initial immersion to 1e-6
  or better.

## Attribution

Fullerene enumeration uses buckygen (Brinkmann, Goedgebeur, McKay); it is a
third-party program, cited in the paper and not redistributed here.

# Third-Party Notices

This repository vendors four upstream MAPF-LNS solver codebases under `src/` so that the
experiments are byte-for-byte reproducible. **Only the environment-gated layer blocks are our
addition** — every change to a host source is marked with a `*_PORT` / `AMOR_*` comment
(`SATA_PORT`, `BESTRET_PORT`, `CART_PORT`, `TRAJ_PORT`, `REPV5_PORT`, `ANCHORFIX_PORT`,
`AMOR_ON`, …). Diffing a host source against its upstream yields exactly the layer. With no
`L2_`/`BL_`/`AD_`/`TK_` environment variable set, each binary reproduces its upstream behavior.

| Directory | Upstream | License |
|---|---|---|
| `src/lns2` | MAPF-LNS2 (Li et al., AAAI 2022) | USC research/non-profit license — see `src/lns2/license.txt`; the `PIBT/` subtree is MIT (© 2019 Keisuke Okumura). |
| `src/balance` | BALANCE (Phan et al., AAAI 2024) | USC research/non-profit license — see `src/balance/license.txt`; `PIBT/` MIT. |
| `src/address` | ADDRESS (Phan et al., AAAI 2025) | USC research/non-profit license — see `src/address/license.txt`; `PIBT/` MIT. |
| `src/tackle` | TACKLE (Phan et al., AAAI 2026) | **No upstream license file was shipped at the time of vendoring; all rights remain with the original authors.** Vendored here for research reproducibility only — see `src/tackle/NOTICE.txt`. |

The USC license (lns2/balance/address) permits use, copying, modification, and distribution for
**educational, research, and non-profit** purposes provided the copyright notice is retained;
commercial use requires contacting the USC Stevens Center for Innovation.

**`src/tackle` carries no redistribution grant.** If you are the TACKLE authors and object to its
inclusion, please open an issue and we will remove it and replace it with a patch of the layer
additions only.

Everything else in this repository (the CARE layer additions, and all of `experiments/`,
`analysis/`, `setup/`, and top-level files) is released by the CARE authors under the MIT
License (see `LICENSE`).

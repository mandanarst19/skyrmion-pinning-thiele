"""Site / cluster labelling (paper definitions) and helpers.

Paper (Commun. Phys. 9:243, Methods, "Skyrmion tracking ... pinning cluster localization"):
 - pinning site: directly connected bins, all with E < -2.5 kBT, at least one with E < -3.5 kBT
   (levels before the unsampled areas are filled);
 - pinning cluster: connected region containing >= 1 site, all bins with known potential;
   expanded by 250 nm in all directions; clusters/sites touching the edge of the field of view are excluded.
Choices NOT specified by the paper (our choice): 8-neighbour connectivity and a Euclidean
250 nm disk; sensitivity: 4-neighbour connectivity and a square dilation.
"""
import numpy as np
from scipy import ndimage as ndi

S8 = np.ones((3, 3), dtype=bool)
S4 = ndi.generate_binary_structure(2, 1)


def label_sites(E, thr_site=-2.5, thr_core=-3.5, connectivity=8):
    st = S8 if connectivity == 8 else S4
    low = np.isfinite(E) & (E < thr_site)
    lab, n = ndi.label(low, structure=st)
    # keep components with at least one bin below thr_core
    mins = ndi.minimum(np.where(low, E, np.inf), lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = mins < thr_core
    lab = np.where(keep[lab], lab, 0)
    # relabel consecutively
    new = np.zeros_like(lab)
    for i, l in enumerate(np.unique(lab[lab > 0]), start=1):
        new[lab == l] = i
    return new, int(new.max())


def dilate(mask, radius_bins, shape="disk"):
    if shape == "disk":
        d = ndi.distance_transform_edt(~mask)
        return d <= radius_bins
    r = int(np.floor(radius_bins))
    return ndi.binary_dilation(mask, structure=np.ones((2 * r + 1, 2 * r + 1), dtype=bool))


def label_clusters(E, h, expand_nm=250.0, connectivity=8, dilation="disk", **site_kw):
    """Return (site_labels, n_sites, cluster_labels, n_clusters, cluster_edge_flag[1..n]).
    Cluster = connected component of finite bins containing >= 1 site, then dilated by expand_nm,
    then re-labelled (dilated components that touch merge)."""
    st = S8 if connectivity == 8 else S4
    sites, ns = label_sites(E, connectivity=connectivity, **site_kw)
    fin_lab, nf = ndi.label(np.isfinite(E), structure=st)
    ids = np.unique(fin_lab[sites > 0])
    base = np.isin(fin_lab, ids[ids > 0])
    exp = dilate(base, expand_nm * 1e-3 / h, dilation)
    clab, nc = ndi.label(exp, structure=st)
    edge = np.zeros(nc + 1, dtype=bool)
    for l in np.unique(np.concatenate([clab[0, :], clab[-1, :], clab[:, 0], clab[:, -1]])):
        if l > 0:
            edge[l] = True
    return sites, ns, clab, nc, edge, base


def bbox(mask, margin=0):
    rr, cc = np.where(mask)
    return rr.min() - margin, rr.max() + margin + 1, cc.min() - margin, cc.max() + margin + 1

"""Candidate GUDHI builders for benchmarking; no dataset or timings run here.

Use exactly the existing Delaunay triangles.  Keep the returned template
pristine: never compute persistence or extend_filtration on it.
"""

import gudhi as gd
import numpy as np


def prepare_triangle_template(triangles):
    """Cache a triangle complex with +inf vertices and -inf higher faces.

    triangles has the SciPy Delaunay shape (number_of_triangles, 3).
    The stored template deliberately is not a valid filtration.  A daily
    builder below restores validity before returning a tree.
    """
    triangles = np.asarray(triangles, dtype=np.intc)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Expected Delaunay triangles with shape (n, 3)")
    vertex_batch = np.unique(triangles)[None, :]
    template = gd.SimplexTree()
    template.insert_batch(triangles.T, np.full(triangles.shape[0], np.inf))
    template.reset_filtration(-np.inf, min_dim=1)
    return template, vertex_batch


def lower_star_copy_with_batch(template, vertex_batch, vertex_values):
    """Clone cached topology and assign a new exact float64 lower star.

    vertex_values is indexed by the original Delaunay vertex IDs.  Pass z
    for the existing extended-persistence approach, or -z when separately
    validating ordinary H1 of the upper-star filtration.
    Only vertices present in the original triangles are included.
    """
    values = np.asarray(vertex_values, dtype=np.float64)
    tree = template.copy()
    tree.insert_batch(vertex_batch, values[vertex_batch[0]])
    tree.make_filtration_non_decreasing()
    return tree


def lower_star_copy_with_assign(template, vertex_batch, vertex_values):
    """Reference candidate retaining per-vertex Python assignment."""
    values = np.asarray(vertex_values, dtype=np.float64)
    tree = template.copy()
    for vertex in vertex_batch[0]:
        tree.assign_filtration([int(vertex)], float(values[vertex]))
    tree.make_filtration_non_decreasing()
    return tree

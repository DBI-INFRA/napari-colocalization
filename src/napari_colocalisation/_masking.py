"""Region-of-interest helpers.

These functions convert a napari Shapes or Labels layer into an
integer label mask (0 = background, 1..N = regions) and iterate
over the non-zero regions. The helpers duck-type the layer
interface — they do not import napari — so they can be tested
with simple stand-in objects.
"""

import numpy as np


def shapes_to_label_mask(shapes_layer, image_shape):
    """Rasterise a napari Shapes layer to an integer label mask.

    Uses ``Shapes.to_labels(labels_shape=image_shape)``. The
    resulting mask has shape ``image_shape``, with 0 outside any
    shape and ``1..N`` inside each shape (where shape ``i`` of
    the layer maps to label ``i + 1``).

    Parameters
    ----------
    shapes_layer : napari.layers.Shapes
        Any object that exposes ``.to_labels(labels_shape=...)``;
        duck-typed so tests can pass a stand-in.
    image_shape : tuple of int
        The desired output shape, typically the spatial shape of
        the image being analysed.

    Returns
    -------
    mask : numpy.ndarray of int
        Integer label image of shape ``image_shape``.
    """
    labels = shapes_layer.to_labels(labels_shape=tuple(image_shape))
    return np.asarray(labels, dtype=int)


def labels_to_label_mask(labels_layer, image_shape):
    """Validate and return integer data from a napari Labels layer.

    Parameters
    ----------
    labels_layer : napari.layers.Labels
        Any object that exposes a ``.data`` attribute holding an
        integer ndarray.
    image_shape : tuple of int
        The expected shape; ``labels_layer.data.shape`` must
        match exactly.

    Returns
    -------
    mask : numpy.ndarray of int
        The label data, dtype-cast to ``int``.

    Raises
    ------
    ValueError
        If the layer's data shape does not match ``image_shape``.
    """
    data = np.asarray(labels_layer.data, dtype=int)
    if data.shape != tuple(image_shape):
        raise ValueError(
            f'labels shape {data.shape} does not match image '
            f'shape {tuple(image_shape)}'
        )
    return data


def iter_regions(label_mask):
    """Yield ``(label_id, bool_mask)`` for each non-zero region.

    Parameters
    ----------
    label_mask : numpy.ndarray of int, or None
        Integer label image. ``None`` is treated specially as
        "use the whole image".

    Yields
    ------
    label_id : int
        The integer label value (1, 2, ...). When ``label_mask``
        is ``None``, yields ``0`` instead.
    bool_mask : numpy.ndarray of bool, or None
        Boolean mask of the same shape as ``label_mask``,
        ``True`` where the label equals ``label_id``. ``None``
        when ``label_mask`` is ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from napari_colocalization._masking import iter_regions
    >>> mask = np.zeros((4, 4), dtype=int)
    >>> mask[:2, :2] = 1
    >>> mask[2:, 2:] = 2
    >>> [(label, int(m.sum())) for label, m in iter_regions(mask)]
    [(1, 4), (2, 4)]
    """
    if label_mask is None:
        yield 0, None
        return
    label_mask = np.asarray(label_mask)
    unique = np.unique(label_mask)
    for label in unique:
        if label == 0:
            continue
        yield int(label), label_mask == label

"""Compatibility import for the preprocessed TS-SatFire array dataset.

The dataset must be paired with a per-window event manifest; aggregate NPY files
without such a manifest are insufficient for paper-level event statistics.
"""

from satimg_dataset_processor.data_generator_torch import FireDataset, Normalize

__all__ = ["FireDataset", "Normalize"]

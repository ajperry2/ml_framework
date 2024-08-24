"""A Dataset which loads pairs of satellite images and knowledge graphs
"""

import warnings
with warnings.catch_warnings():
    # Annoying warning from osmnx for each map download...
    warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.utils.data import Dataset
from typing import List, Optional
from pathlib import Path
import osmnx as ox
import numpy as np
import random
from math import ceil
import os
import networkx as nx

from ml_framework.data.osm.map_image import get_mapbox_image


class OSMDataset(Dataset):
    def __init__(
        self,
        bounding_box: List[float],
        num_samples: int = 100,
        network_type: str = "all",
        cache_dir: Optional[Path] = Path("./data"),
        image_width: int = 256,
        image_height: int = 256,
        split_seed: int = 0,
        is_train: bool = True,
        train_split: float = 0.8
    ):
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        _available_network_types = {
            "drive", "drive_service", "walk", "bike", "all"
        }
        assert network_type in _available_network_types, \
            f"Invalid Network Type: {network_type}"
        self.network_type = network_type
        self.image_width, self.image_height = image_width, image_height
        assert len(bounding_box) == 4
        north, south, east, west = bounding_box
        assert num_samples ** 0.5 % 1 == 0
        sample_side_splits = int(num_samples ** 0.5)
        bottom = min(north, south)
        top = max(north, south)
        left = min(east, west)
        right = max(east, west)
        north_south_delta = (top - bottom) / sample_side_splits
        east_west_delta = (right - left) / sample_side_splits
        self.bounding_boxes: List[List[float]] = []
        # Find which regions will be train and which will be val
        random.seed(split_seed)
        random_samples = [random.random() for i in range(num_samples)]
        num_train_samples = ceil(num_samples * train_split)
        if is_train:
            # If `is_train` Keep first num_samples * train_percent
            self.indices = np.argsort(random_samples)[:num_train_samples]
        else:
            # keep last indices for testing
            self.indices = np.argsort(random_samples)[num_train_samples:]

        region_index = 0

        for side_i in range(sample_side_splits):
            for side_j in range(sample_side_splits):
                region_left = left + east_west_delta * side_i
                region_right = left + east_west_delta * (side_i+1)
                region_bottom = bottom + north_south_delta * side_j
                region_top = bottom + north_south_delta * (side_j+1)

                if region_index in self.indices:
                    self.bounding_boxes.append(
                        [region_left, region_bottom, region_right, region_top]
                    )
                    if (
                        self.cache_dir is not None
                        and not os.path.exists(
                            self.cache_dir/f"{region_index}.npy")):
                        # Save satellite imagery to disk
                        map_image = get_mapbox_image(
                            region_left, region_bottom,
                            region_right, region_top,
                            image_width=image_width, image_height=image_height
                        )
                        map_img_array = np.array(map_image)
                        np.save(
                            self.cache_dir/f"{region_index}.npy",
                            map_img_array)
                        try:
                            graph = ox.graph_from_bbox(
                                region_top, region_bottom,
                                region_right, region_left,
                                network_type=self.network_type)
                            # Save network info to disk
                            ox.save_graphml(
                                graph,
                                filepath=self.cache_dir/f"{region_index}.graphml")
                        except ValueError:
                            graph = nx.MultiDiGraph()
                            nx.write_graphml(
                                graph,
                                self.cache_dir/f"{region_index}.graphml"
                            )
                region_index += 1

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        if self.cache_dir is not None:
            # Load from files
            item_name = self.indices[i]
            sat_image = np.load(self.cache_dir/f"{item_name}.npy")
            graph = ox.load_graphml(self.cache_dir/f"{item_name}.graphml")
        else:
            # Download
            bounding_box = self.bounding_boxes[i]
            region_left, region_bottom, region_right, region_top = bounding_box
            map_image = get_mapbox_image(
                            region_left, region_bottom,
                            region_right, region_top,
                            image_width=self.image_width,
                            image_height=self.image_height
                        )
            sat_image = np.array(map_image)
            graph = ox.graph_from_bbox(
                            region_top, region_bottom,
                            region_right, region_left,
                            network_type=self.network_type)
        return sat_image, graph

    def plot(self, i):
        sat_image, graph = self[i]
        fig, ax = ox.plot.plot_graph(
            graph,
            node_size=1,
            node_alpha=0.1,
            edge_linewidth=1,
            show=False)
        ax.imshow(sat_image, extent=[*ax.get_xlim(), *ax.get_ylim()])
        return ax
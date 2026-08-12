import unittest

import torch

from tokenfovea.pyramid import LayerKVPyramid
from tokenfovea.topology import DeviceTreeTopology, VisualTokenForest


class PyramidTest(unittest.TestCase):
    def test_levelwise_pyramid_matches_leaf_means(self):
        forest = VisualTokenForest.from_grids([(3, 5)])
        topology = DeviceTreeTopology.build(forest, torch.device("cpu"))
        leaves = forest.num_leaves
        keys = torch.arange(2 * leaves * 3, dtype=torch.float32).reshape(1, 2, leaves, 3)
        values = keys + 1000
        positions = torch.arange(3 * leaves, dtype=torch.float32).reshape(3, 1, leaves)
        pyramid = LayerKVPyramid.from_kv(topology, keys, values, positions)

        for node in forest.nodes:
            indices = torch.tensor(node.leaf_indices)
            self.assertTrue(
                torch.allclose(
                    pyramid.raw_keys[..., node.node_id, :],
                    keys.index_select(-2, indices).mean(-2),
                )
            )
            self.assertTrue(
                torch.allclose(
                    pyramid.values[..., node.node_id, :],
                    values.index_select(-2, indices).mean(-2),
                )
            )
            self.assertTrue(
                torch.allclose(
                    pyramid.native_positions[..., node.node_id],
                    positions.index_select(-1, indices).mean(-1),
                )
            )

    def test_position_modes_gather_expected_keys(self):
        forest = VisualTokenForest.from_grids([(2, 2)])
        topology = DeviceTreeTopology.build(forest, torch.device("cpu"))
        keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        values = keys + 100
        positions = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)
        rotated = keys + 200
        pyramid = LayerKVPyramid.from_kv(topology, keys, values, positions, rotated)
        active_ids = torch.tensor([forest.roots[0]])

        def encoder(reference, node_positions):
            return node_positions, node_positions

        def rotate(raw_keys, cos, sin):
            return raw_keys

        native, _ = pyramid.gather(active_ids, "native_center", rotate, encoder, keys)
        no_rope, _ = pyramid.gather(active_ids, "no_rope", rotate, encoder, keys)
        post_rope, _ = pyramid.gather(active_ids, "post_rope_pool", rotate, encoder, keys)
        anchors = torch.zeros(3, 1, 1)
        anchored, _ = pyramid.gather(active_ids, "text_anchor", rotate, encoder, keys, anchors)

        expected_raw = keys.mean(dim=-2, keepdim=True)
        self.assertTrue(torch.allclose(native, expected_raw))
        self.assertTrue(torch.allclose(no_rope, expected_raw))
        self.assertTrue(torch.allclose(anchored, expected_raw))
        self.assertTrue(torch.allclose(post_rope, rotated.mean(dim=-2, keepdim=True)))

    def test_hidden_pooling_projects_after_pooling(self):
        forest = VisualTokenForest.from_grids([(2, 2)])
        topology = DeviceTreeTopology.build(forest, torch.device("cpu"))
        hidden = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        positions = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4)

        def projector(nodes):
            return nodes.square().unsqueeze(1), (nodes + 1).unsqueeze(1)

        pyramid = LayerKVPyramid.from_hidden(topology, hidden, positions, projector)
        root = forest.roots[0]
        expected_hidden = hidden.mean(dim=1)
        self.assertTrue(torch.allclose(pyramid.raw_keys[..., root, :], expected_hidden.square().unsqueeze(1)))
        self.assertTrue(torch.allclose(pyramid.values[..., root, :], (expected_hidden + 1).unsqueeze(1)))

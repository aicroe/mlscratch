"""RandomMiniBatchGenerator test suit."""
from unittest import TestCase
import numpy as np

from mlbatch.chunk_generator import RandomMiniBatchGenerator


class RandomMiniBatchGeneratorTest(TestCase):

    def test_generate_assert_number_of_chunks(self):
        generator = RandomMiniBatchGenerator()

        size, chunks = generator.generate(
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            2,
        )

        self.assertEqual(size, 3)
        self.assertEqual(size, len(list(chunks)))

    def test_generate_assert_zero_number_of_chunks(self):
        generator = RandomMiniBatchGenerator()

        size, chunks = generator.generate(
            np.array([]),
            np.array([]),
            5,
        )

        self.assertEqual(size, 0)
        self.assertEqual(size, len(list(chunks)))

    def test_generate_assert_chunks_number_when_requested_size_is_bigger(self):
        generator = RandomMiniBatchGenerator()

        size, chunks = generator.generate(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            5,
        )

        self.assertEqual(size, 1)
        self.assertEqual(size, len(list(chunks)))

    def test_generate_assert_chunk_size_is_whole_data(self):
        generator = RandomMiniBatchGenerator()

        size, chunks = generator.generate(
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            3,
        )

        self.assertEqual(size, 1)
        self.assertEqual(size, len(list(chunks)))

    def test_generate_assert_first_chunk_size(self):
        generator = RandomMiniBatchGenerator(0)
        dataset = np.array([[1, 2], [2, 3], [3, 3]])
        labels = np.array([0, 1, 3])

        _, iterator = generator.generate(dataset, labels, 2)

        (dataset_minibatch, labels_minibatch), *_ = list(iterator)
        dataset_minibatch_size, _ = dataset_minibatch.shape
        label_minibatch_size, = labels_minibatch.shape
        self.assertEqual(dataset_minibatch_size, 2)
        self.assertEqual(label_minibatch_size, 2)

    def test_generate_assert_first_generated_chunk(self):
        generator = RandomMiniBatchGenerator(0)
        dataset = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[2, 4, 6], [8, 0, 2]],
            [[0, 1, 2], [3, 2, 1]],
        ])
        labels = np.array([[10], [20], [15]])

        _, iterator = generator.generate(dataset, labels, 2)

        (dataset_minibatch, labels_minibatch), *_ = list(iterator)
        self.assertTrue(np.array_equal(
            np.array([
                [[0, 1, 2], [3, 2, 1]],
                [[2, 4, 6], [8, 0, 2]],
            ]),
            dataset_minibatch,
        ))
        self.assertTrue(np.array_equal(
            np.array([[15], [20]]),
            labels_minibatch,
        ))

    def test_generate_assert_not_even_sets_of_chunks(self):
        generator = RandomMiniBatchGenerator(0)
        dataset = np.array([[0, 1], [1, 0], [2, 2], [4, 4]])
        labels = np.array([9, 8, 7, 6])

        _, iterator = generator.generate(dataset, labels, 3)

        minibatches = list(iterator)
        self.assertEqual(len(minibatches), 2)
        self.assertTrue(np.array_equal(
            np.array([[2, 2], [4, 4], [1, 0]]),
            minibatches[0][0],
        ))
        self.assertTrue(np.array_equal(
            np.array([7, 6, 8]),
            minibatches[0][1],
        ))
        self.assertTrue(np.array_equal(
            np.array([[0, 1]]),
            minibatches[1][0],
        ))
        self.assertTrue(np.array_equal(
            np.array([9]),
            minibatches[1][1],
        ))

    def test_generate_assert_high_dim_chunks(self):
        generator = RandomMiniBatchGenerator(0)
        dataset = np.array([
            [[1, 2, 3, 4], [3, 6, 8, 8], [0, 1, 2, 9]],
            [[4, 3, 2, 1], [4, 5, 7, 9], [7, 2, 1, 2]],
            [[0, 1, 0, 1], [1, 2, 8, 0], [6, 3, 9, 1]],
            [[9, 9, 1, 1], [8, 4, 9, 1], [9, 4, 1, 0]],
        ])
        labels = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])

        _, iterator = generator.generate(dataset, labels, 2)

        minibatches = list(iterator)
        self.assertEqual(len(minibatches), 2)
        self.assertTrue(np.array_equal(
            np.array([
                [[0, 1, 0, 1], [1, 2, 8, 0], [6, 3, 9, 1]],
                [[9, 9, 1, 1], [8, 4, 9, 1], [9, 4, 1, 0]],
            ]),
            minibatches[0][0],
        ))
        self.assertTrue(np.array_equal(
            np.array([[1, 1], [0, 0]]),
            minibatches[0][1],
        ))
        self.assertTrue(np.array_equal(
            np.array([
                [[4, 3, 2, 1], [4, 5, 7, 9], [7, 2, 1, 2]],
                [[1, 2, 3, 4], [3, 6, 8, 8], [0, 1, 2, 9]],
            ]),
            minibatches[1][0],
        ))
        self.assertTrue(np.array_equal(
            np.array([[1, 0], [0, 1]]),
            minibatches[1][1],
        ))

    def test_generate_assert_only_chunk_is_whole_data(self):
        generator = RandomMiniBatchGenerator(0)
        dataset = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
        labels = np.array([0, 9, 8])

        _, iterator = generator.generate(dataset, labels, 3)

        minibatches = list(iterator)
        self.assertEqual(len(minibatches), 1)
        self.assertTrue(np.array_equal(
            np.array([[2, 1, 3], [3, 2, 1], [1, 2, 3]]),
            minibatches[0][0]
        ))
        self.assertTrue(np.array_equal(
            np.array([8, 9, 0]),
            minibatches[0][1]
        ))

    def test_generate_when_chunk_size_is_one(self):
        generator = RandomMiniBatchGenerator(0)
        dataset = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 9, 8, 7, 6])

        size, iterator = generator.generate(dataset, labels, 1)

        self.assertEqual(size, 5)
        self.assertTrue(np.array_equal(
            [([3], [8]), ([1], [0]), ([2], [9]), ([4], [7]), ([5], [6])],
            list(iterator),
        ))

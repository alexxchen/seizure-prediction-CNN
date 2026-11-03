"""Evaluation summaries use the existing equal-mouse final-support pool metrics."""

import copy
import unittest

from meta_learning.training_log import evaluation_event, readable_lines


class EvaluationLogTests(unittest.TestCase):
    def test_support_mean_weights_mice_equally_and_uses_final_pool_scope(self):
        reports = {
            'query': {'aggregate': {'accuracy': .7, 'loss': .5}, 'subjects': {
                'X1': {'classifier_fit': {'last_support_accuracy': .8, 'last_support_loss': .4,
                    'support_pool_samples': 10, 'history': [{'support_accuracy': .1, 'support_loss': 2.0}],
                    'support_crop_validation': {'accuracy': .3, 'loss': 1.5}}},
                'X2': {'classifier_fit': {'last_support_accuracy': 1., 'last_support_loss': .2,
                    'support_pool_samples': 90, 'history': [{'support_accuracy': .2, 'support_loss': 3.0}],
                    'support_crop_validation': {'accuracy': .4, 'loss': 1.4}}}}},
            'test': {'aggregate': {'accuracy': .6, 'loss': .8}}}
        before = copy.deepcopy(reports)
        event = evaluation_event(10, reports)
        self.assertAlmostEqual(event['support']['accuracy'], .9)
        self.assertAlmostEqual(event['support']['loss'], .3)
        self.assertEqual(event['query'], reports['query']['aggregate'])
        self.assertEqual(event['test'], reports['test']['aggregate'])
        self.assertEqual(reports, before)
        self.assertEqual(readable_lines('evaluation_step', event, 800),
            ['[010/800] EVAL MEAN | support 90.00%/0.3000 | query 70.00%/0.5000 | test 60.00%/0.8000'])


if __name__ == '__main__':
    unittest.main()

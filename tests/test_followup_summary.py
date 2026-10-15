"""Offline artifact fixtures; no attack, training, or network use."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from gbhaa_experiments import summarize_followup as summary
from gbhaa_experiments.followup_protocol import array_hash, graph_hash, prediction_metrics, seed_stream
from gbhaa_experiments.run_followup import PROTOCOL, fingerprint, make_jobs, settings, protocol_for_suite
from gbhaa_experiments.followup_config import pr_solver_settings, factorial_operator_settings


class FollowupSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'results'
        self.output = Path(self.temp.name) / 'reports'

    @staticmethod
    def provenance(protocol=PROTOCOL):
        digest = hashlib.sha256(b'historical fixture, not current source').hexdigest()
        return dict(protocol=protocol, sources={'historical.py': digest},
                    input_files={'historical-data.npz': digest},
                    python_executable='/historical/python', python_environment={'python': '3.fixture'},
                    api_base_url='http://offline.invalid', generation_model='fixture-model',
                    machine='fixture', cuda_visible_devices=None, thread_count=1,
                    generation_temperature=0.0)

    @staticmethod
    def original_graph():
        dense = np.zeros((12, 12), dtype=float)
        for u, v in list(zip(*np.triu_indices(12, 1)))[:40]:
            dense[u, v] = dense[v, u] = 1
        return sp.csr_matrix(dense)

    @staticmethod
    def refresh_fingerprint(row):
        job = next(j for j in make_jobs(row['suite'], [row['seed']], [row['budget']])
                   if j['variant'] == row['variant'])
        row['fingerprint'] = fingerprint(dict(job=job, settings=row['settings'], provenance=row['provenance']))

    def directory(self, row):
        return self.root / row['suite'] / row['dataset'] / f"ptb_{row['budget']:g}" / f"seed_{row['seed']}" / row['variant']

    def save_record(self, row):
        directory = self.directory(row)
        (directory / 'result.json').write_text(json.dumps(row))
        for name in ('attack', 'evaluation'):
            (directory / 'artifacts' / f'{name}.json').write_text(json.dumps(row[name]))

    def set_predictions(self, row, *, ids=None, labels=None, clean=None, attacked=None):
        ids = np.arange(6) if ids is None else np.asarray(ids)
        labels = np.zeros(6, dtype=np.int64) if labels is None else np.asarray(labels)
        clean = np.zeros(6, dtype=np.int64) if clean is None else np.asarray(clean)
        attacked = np.array([0, 0, 0, 1, 1, 1]) if attacked is None else np.asarray(attacked)
        artifacts = self.directory(row) / 'artifacts'
        for name, predictions in (('clean', clean), ('combined', attacked)):
            np.savez(artifacts / f'{name}_predictions.npz', node_ids=ids, labels=labels, predictions=predictions)
        metrics = prediction_metrics(clean, attacked, labels)
        metrics['combined_accuracy'] = metrics['attacked_accuracy']
        row['evaluation']['metrics'] = metrics
        row['evaluation']['clean_predictions_hash'] = array_hash(clean)
        row['attack']['target_nodes_hash'] = array_hash(ids)
        row['attack']['evaluation_labels_hash'] = array_hash(labels)

    def set_actions(self, row, actions):
        modified = self.original_graph().tolil(copy=True)
        for u, v, value in actions:
            modified[u, v] = modified[v, u] = value
        row['attack'].update(actions=actions,
            actions_hash=hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
            modified_adj_hash=graph_hash(modified))
        sp.save_npz(self.directory(row) / 'artifacts/modified_adj.npz', modified.tocsr())

    def fixture(self, suite='selectors', seeds=(15,), budget=.05):
        rows = []
        for job in make_jobs(suite, seeds, [budget]):
            protocol = protocol_for_suite(suite)
            row = {**job, 'status': 'complete', 'protocol': protocol, 'returncode': 0,
                   'settings': settings(job), 'provenance': self.provenance(protocol),
                   'wall_time_seconds': 2., 'peak_rss_mib': 100.,
                   'attack': {}, 'evaluation': {'protocol': protocol,
                       'victim_seed': seed_stream(job['seed'], 'victim')}}
            artifacts = self.directory(row) / 'artifacts'
            artifacts.mkdir(parents=True, exist_ok=True)
            original = self.original_graph()
            sp.save_npz(artifacts / 'original_adj.npz', original)
            edges = int((original.nnz // 2) * budget)
            row['attack'] = {**{key: job[key] for key in (
                'seed', 'dataset', 'selector', 'local_graph', 'global_transition', 'global_source')},
                'protocol': protocol, 'budget_complete': True, 'target_edges': edges,
                'completed_edges': edges, 'original_adj_hash': graph_hash(original),
                'original_features_hash': array_hash(np.eye(12)),
                'training_nodes_hash': array_hash(np.arange(6, 12)),
                'generation_model': 'fixture-model', 'generation_seed': seed_stream(job['seed'], 'generation'),
                'generation_temperature': 0., 'changed_feature_nodes': 1,
                'changed_feature_entries': 2, 'attribute_node_visits': 1, 'structural_seconds': 1.}
            if job['selector'] == 'pr':
                row['attack']['pr_solver'] = pr_solver_settings(row['settings']['text_ppr_alpha'])
            if suite == 'factorial':
                row['attack']['operator_settings'] = factorial_operator_settings()
                row['attack']['requested_local_graph'] = 'pre'
            actions = [[int(u), int(v), 0] for u, v in list(zip(*sp.triu(original, k=1).nonzero()))[:edges]]
            self.set_actions(row, actions)
            attacked = np.ones(6, dtype=np.int64) if job['selector'] == 'ae_pr' else np.array([0, 0, 0, 1, 1, 1])
            self.set_predictions(row, attacked=attacked)
            self.refresh_fingerprint(row)
            self.save_record(row)
            rows.append(row)
        return rows

    def run_summary(self, *, suites=('selectors',), seeds=(15,), budgets=(.05,)):
        with contextlib.redirect_stdout(io.StringIO()):
            code = summary.main(['--root', str(self.root), '--output-dir', str(self.output),
                '--suites', *suites, '--seeds', *map(str, seeds), '--budgets', *map(str, budgets)])
        return code, json.loads((self.output / 'summary.json').read_text())

    def assert_rejected(self, expected_error):
        code, report = self.run_summary()
        self.assertEqual(code, 2)
        self.assertFalse(report['ready_for_full_seed_reporting'])
        self.assertFalse(report['complete_blocks'])
        self.assertFalse(report['summary'])
        self.assertIn(expected_error.lower(), json.dumps(report['rejected_blocks']).lower())
        return report

    def test_valid_artifacts_all_suites_and_historical_source_are_accepted_readonly(self):
        suites = ('factorial', 'selectors', 'ordering', 'uniform_repair')
        for suite in suites:
            self.fixture(suite)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        code, report = self.run_summary(suites=suites)
        self.assertEqual(code, 0)
        self.assertTrue(report['ready_for_full_seed_reporting'])
        self.assertEqual(len(report['complete_blocks']), 4)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_duplicate_axes_and_invalid_budgets_are_cli_errors(self):
        for args in (dict(seeds=(15, 15)), dict(budgets=(.05, .05)),
                     dict(suites=('selectors', 'selectors')), dict(budgets=(0,)),
                     dict(budgets=(float('nan'),)), dict(budgets=(1.1,)),
                     dict(budgets=(.05, .0500000001))):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_summary(**args)
                self.assertEqual(error.exception.code, 2)

    def test_selector_legacy_protocol_is_rejected_even_with_matching_pre_graph(self):
        row = self.fixture()[0]
        row['protocol'] = row['provenance']['protocol'] = PROTOCOL
        row['attack']['protocol'] = row['evaluation']['protocol'] = PROTOCOL
        self.refresh_fingerprint(row)
        self.save_record(row)
        self.assert_rejected('protocol')

    def test_pr_solver_metadata_must_match_the_declared_uniform_source_solver(self):
        base = next(row for row in self.fixture() if row['selector'] == 'pr')
        for key, value in (('personalization', 'endpoints'), ('damping', .85),
                           ('max_iter', 60), ('weight', 'weight'), (None, None)):
            with self.subTest(key=key):
                row = copy.deepcopy(base)
                if key is None:
                    del row['attack']['pr_solver']
                else:
                    row['attack']['pr_solver'][key] = value
                self.save_record(row)
                self.assert_rejected('pr_solver')

    def test_summary_resolves_each_suite_default_root_independently(self):
        roots = {}
        for suite in ('selectors', 'factorial'):
            self.root = Path(self.temp.name) / suite
            roots[suite] = self.root
            self.fixture(suite)
        with patch.object(summary, 'default_output', side_effect=roots.__getitem__) as resolve, \
                contextlib.redirect_stdout(io.StringIO()):
            code = summary.main(['--output-dir', str(self.output), '--suites',
                'selectors', 'factorial', '--seeds', '15', '--budgets', '.05'])
        self.assertEqual(code, 0)
        self.assertEqual({call.args[0] for call in resolve.call_args_list}, set(roots))
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['protocols'], {suite: protocol_for_suite(suite) for suite in roots})
        self.assertEqual(len(report['complete_blocks']), 2)

    def test_missing_arm_has_exact_job_mapping_and_no_partial_block(self):
        row = self.fixture()[0]
        path = self.directory(row) / 'result.json'
        path.unlink()
        code, report = self.run_summary()
        self.assertEqual(code, 2)
        self.assertEqual(report['missing'], [str(path)])
        self.assertEqual(report['missing_jobs'][0]['variant'], row['variant'])
        self.assertEqual(report['missing_jobs'][0]['seed'], 15)
        self.assertFalse(report['complete_blocks'])

    def test_seed_pairing_uses_unique_complete_blocks_in_requested_order(self):
        for row in self.fixture(seeds=(15, 16)):
            correct = (1 if row['seed'] == 15 else 4) + (row['selector'] != 'ae_pr')
            self.set_predictions(row, attacked=np.array([0] * correct + [1] * (6 - correct)))
            self.save_record(row)
        code, report = self.run_summary(seeds=(16, 15))
        self.assertEqual(code, 0)
        for contrast in report['contrasts']:
            self.assertEqual(contrast['n'], 2)
            self.assertEqual(contrast['seeds'], [16, 15])
            self.assertAlmostEqual(contrast['mean'], 100 / 6)
            self.assertAlmostEqual(contrast['sd'], 0)

    def test_required_provenance_and_fingerprint_are_checked(self):
        base = self.fixture()[0]
        mutations = [
            (lambda r: r.update(fingerprint='bad'), 'fingerprint'),
            (lambda r: r.pop('fingerprint'), 'fingerprint'),
            (lambda r: r['provenance'].update(sources={}), 'sources'),
            (lambda r: r['provenance'].update(input_files={}), 'input_files'),
            (lambda r: r['provenance'].pop('cuda_visible_devices'), 'cuda_visible_devices'),
            (lambda r: r['provenance'].update(python_environment=None), 'python_environment'),
        ]
        for mutate, error in mutations:
            with self.subTest(error=error):
                row = copy.deepcopy(base)
                mutate(row)
                if error != 'fingerprint':
                    self.refresh_fingerprint(row)
                self.save_record(row)
                self.assert_rejected(error)

    def test_nested_identity_protocol_and_seed_streams_are_required(self):
        base = self.fixture()[0]
        changes = [
            ('attack', 'protocol', 'legacy'), ('evaluation', 'protocol', 'legacy'),
            ('attack', 'seed', 19), ('attack', 'dataset', 'cora'),
            ('attack', 'selector', 'random'), ('attack', 'local_graph', 'post'),
            ('attack', 'global_transition', 'topology'), ('attack', 'global_source', 'uniform'),
            ('attack', 'generation_seed', seed_stream(19, 'generation')),
            ('evaluation', 'victim_seed', seed_stream(19, 'victim')),
            ('attack', 'generation_temperature', 1.), ('attack', 'generation_model', 'other'),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                row = copy.deepcopy(base)
                row[section][key] = value
                self.save_record(row)
                self.assert_rejected(key)

    def test_nonzero_returncode_and_changed_settings_are_rejected(self):
        base = self.fixture()[0]
        row = copy.deepcopy(base)
        row['returncode'] = 1
        self.save_record(row)
        self.assert_rejected('returncode')
        row = copy.deepcopy(base)
        row['settings']['text_max_added_words'] += 1
        self.refresh_fingerprint(row)
        self.save_record(row)
        self.assert_rejected('settings')

    def test_prediction_population_must_match_across_arms(self):
        row = self.fixture()[0]
        self.set_predictions(row, ids=np.arange(6, 12))
        self.save_record(row)
        self.assert_rejected('target_nodes_hash')

    def test_prediction_labels_must_match_across_arms(self):
        row = self.fixture()[0]
        self.set_predictions(row, labels=np.array([1, 0, 0, 0, 0, 0]))
        self.save_record(row)
        self.assert_rejected('evaluation_labels_hash')

    def test_prediction_shapes_unique_ids_and_integer_classes_are_checked(self):
        base = self.fixture()[0]
        for ids, error in (([0, 0, 2, 3, 4, 5], 'unique'), ([0], 'shape'),
                           ([0, 1, 2, 3, 4, 12], 'node_ids'),
                           ([[0, 1, 2, 3, 4, 5]], 'shape')):
            with self.subTest(ids=ids):
                row = copy.deepcopy(base)
                self.set_predictions(row, ids=ids)
                self.save_record(row)
                self.assert_rejected(error)
        row = copy.deepcopy(base)
        self.set_predictions(row, attacked=np.full(6, .5))
        self.save_record(row)
        self.assert_rejected('integer')

    def test_summary_rejects_forged_asr_even_when_accuracy_matches(self):
        row = self.fixture()[0]
        row['evaluation']['metrics']['conditional_asr'] = .999
        self.save_record(row)
        self.assert_rejected('conditional_asr')

    def test_missing_and_forged_prediction_metrics_are_rejected(self):
        base = self.fixture()[0]
        for key, value in (('clean_accuracy', -9), ('accuracy_drop_pp', 999),
                           ('clean_correct_count', True), ('combined_accuracy', .8)):
            with self.subTest(key=key):
                row = copy.deepcopy(base)
                row['evaluation']['metrics'][key] = value
                self.save_record(row)
                self.assert_rejected(key)
        row = copy.deepcopy(base)
        row['evaluation']['metrics'].pop('conditional_asr')
        self.save_record(row)
        self.assert_rejected('conditional_asr')

    def test_undefined_asr_and_optional_counters_need_no_extra_artifacts(self):
        for row in self.fixture():
            self.set_predictions(row, clean=np.ones(6, dtype=np.int64))
            row['attack']['optional_counter'] = 3
            self.save_record(row)
        code, report = self.run_summary()
        self.assertEqual(code, 0)
        with (self.output / 'per_seed.csv').open() as handle:
            self.assertTrue(all(r['conditional_asr'] == '' for r in csv.DictReader(handle)))
        self.assertTrue(report['ready_for_full_seed_reporting'])

    def test_export_uses_exact_recomputed_metrics_within_tolerance(self):
        row = self.fixture()[0]
        row['evaluation']['metrics']['clean_accuracy'] -= 5e-11
        self.save_record(row)
        self.assertEqual(self.run_summary()[0], 0)
        with (self.output / 'per_seed.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(float(r['clean_accuracy']) == 1. for r in rows))

    def test_saved_json_payloads_must_match_result_payloads(self):
        row = self.fixture()[0]
        for name in ('attack', 'evaluation'):
            with self.subTest(name=name):
                self.save_record(row)
                (self.directory(row) / 'artifacts' / f'{name}.json').write_text('{}')
                self.assert_rejected(name + '.json')

    def test_actions_digest_count_and_real_graph_budget_are_checked(self):
        base = self.fixture()[0]
        for mutation, error in (
            (lambda a: a.update(actions_hash='a' * 64), 'actions_hash'),
            (lambda a: a.update(actions=a['actions'][:1]), 'actions'),
            (lambda a: a.update(target_edges=1, completed_edges=1), 'budget'),
        ):
            with self.subTest(error=error):
                row = copy.deepcopy(base)
                mutation(row['attack'])
                self.save_record(row)
                self.assert_rejected(error)

    def test_reflips_self_loops_nonflips_and_invalid_endpoints_are_rejected(self):
        base = self.fixture()[0]
        for actions, error in (
            ([[0, 1, 0], [1, 0, 1]], 'reflip'),
            ([[0, 0, 1], [0, 1, 0]], 'self'),
            ([[0, 1, 1], [0, 2, 0]], 'flip'),
            ([[0, 12, 1], [0, 2, 0]], 'endpoint'),
            ([[0, 1, False], [0, 2, 0]], 'integer'),
        ):
            with self.subTest(actions=actions):
                row = copy.deepcopy(base)
                row['attack']['actions'] = actions
                row['attack']['actions_hash'] = hashlib.sha256(json.dumps(actions).encode()).hexdigest()
                self.save_record(row)
                self.assert_rejected(error)

    def test_modified_graph_must_equal_replayed_actions_even_with_valid_hash(self):
        row = self.fixture()[0]
        modified = self.original_graph().tolil()
        modified[0, 3] = modified[3, 0] = 0
        modified[0, 4] = modified[4, 0] = 0
        sp.save_npz(self.directory(row) / 'artifacts/modified_adj.npz', modified.tocsr())
        row['attack']['modified_adj_hash'] = graph_hash(modified)
        self.save_record(row)
        self.assert_rejected('replay')

    def test_graphs_must_be_binary_symmetric_and_hash_verified(self):
        row = self.fixture()[0]
        path = self.directory(row) / 'artifacts/original_adj.npz'
        for value, symmetric, error in ((.5, True, 'binary'), (0, False, 'symmetric')):
            with self.subTest(error=error):
                graph = self.original_graph().tolil()
                graph[0, 1] = value
                if symmetric:
                    graph[1, 0] = value
                sp.save_npz(path, graph.tocsr())
                self.assert_rejected(error)
        sp.save_npz(path, self.original_graph())
        row['attack']['original_adj_hash'] = 'b' * 64
        self.save_record(row)
        self.assert_rejected('original_adj_hash')

    def test_missing_or_corrupt_artifacts_are_reported_not_raised(self):
        row = self.fixture()[0]
        for name in ('original_adj.npz', 'modified_adj.npz', 'attack.json', 'evaluation.json',
                     'clean_predictions.npz', 'combined_predictions.npz'):
            path = self.directory(row) / 'artifacts' / name
            original = path.read_bytes()
            for content in (None, b'corrupt-not-a-valid-artifact'):
                with self.subTest(name=name, missing=content is None):
                    if content is None:
                        path.unlink()
                    else:
                        path.write_bytes(content)
                    self.assert_rejected(name)
                    path.write_bytes(original)

    def test_malformed_records_and_missing_export_fields_do_not_abort(self):
        base = self.fixture()[0]
        path = self.directory(base) / 'result.json'
        for text in ('[]', 'null', '{broken', json.dumps({**base, 'evaluation': None}),
                     json.dumps({**base, 'attack': None})):
            with self.subTest(text=text[:30]):
                path.write_text(text)
                self.assert_rejected('result.json')
        for mutate, error in (
            (lambda r: r.pop('wall_time_seconds'), 'wall_time_seconds'),
            (lambda r: r['evaluation']['metrics'].update(combined_accuracy='.5'), 'combined_accuracy'),
            (lambda r: r['attack'].update(changed_feature_entries=-1), 'changed_feature_entries'),
            (lambda r: r.update(wall_time_seconds=float('nan')), 'non-finite'),
        ):
            with self.subTest(error=error):
                row = copy.deepcopy(base)
                mutate(row)
                self.save_record(row)
                self.assert_rejected(error)

    def test_factorial_allows_different_global_actions_but_not_within_global(self):
        rows = self.fixture('factorial')
        for row in rows:
            if row['global_transition'] == 'attribute':
                self.set_actions(row, [[0, 3, 0], [0, 4, 0]])
                self.save_record(row)
        self.assertEqual(self.run_summary(suites=('factorial',))[0], 0)
        self.set_actions(rows[0], [[0, 5, 0], [0, 6, 0]])
        self.save_record(rows[0])
        code, report = self.run_summary(suites=('factorial',))
        self.assertEqual(code, 2)
        self.assertIn('actions', json.dumps(report['rejected_blocks']))

    def test_factorial_global_local_and_interaction_use_paired_seed_differences(self):
        correct_by_seed = {
            15: {'topology_ppr': 6, 'topology_ae_pr': 4, 'attribute_ppr': 5, 'attribute_ae_pr': 2},
            16: {'topology_ppr': 5, 'topology_ae_pr': 3, 'attribute_ppr': 3, 'attribute_ae_pr': 2},
        }
        for budget in (.05, .1):
            for row in self.fixture('factorial', seeds=(15, 16), budget=budget):
                correct = correct_by_seed[row['seed']][row['variant']]
                self.set_predictions(row, attacked=np.array([0] * correct + [1] * (6 - correct)))
                self.save_record(row)
        code, report = self.run_summary(suites=('factorial',), seeds=(16, 15), budgets=(.05, .1))
        self.assertEqual(code, 0)
        self.assertEqual(report['holm_family']['size'], 10)
        self.assertEqual(len(report['contrasts']), 10)
        for row in report['contrasts']:
            self.assertEqual(row['seeds'], [16, 15])
            self.assertEqual(row['n'], 2)
            if row['contrast'].startswith('global_gain'):
                self.assertAlmostEqual(row['mean'], 25.)
                self.assertAlmostEqual(row['sd'], 100 / 6 / np.sqrt(2))
            elif row['contrast'] == 'interaction_in_accuracy_drop_pp':
                self.assertAlmostEqual(row['mean'], 0.)
                self.assertAlmostEqual(row['sd'], 100 / 3 / np.sqrt(2))

    def test_factorial_rejects_old_protocol_post_graph_and_changed_operators(self):
        base = self.fixture('factorial')[0]
        for mutation in ('protocol', 'local_graph', 'requested_local_graph', 'operator_settings', 'readout'):
            with self.subTest(mutation=mutation):
                row = copy.deepcopy(base)
                if mutation == 'protocol':
                    row['protocol'] = row['provenance']['protocol'] = PROTOCOL
                    row['attack']['protocol'] = row['evaluation']['protocol'] = PROTOCOL
                    self.refresh_fingerprint(row)
                elif mutation == 'operator_settings':
                    del row['attack']['operator_settings']
                elif mutation == 'readout':
                    row['attack']['operator_settings']['global']['readout'] = 'last_state'
                else:
                    row['attack'][mutation] = 'post'
                self.save_record(row)
                code, report = self.run_summary(suites=('factorial',))
                self.assertEqual(code, 2)
                self.assertFalse(report['complete_blocks'])
                self.assertFalse(report['summary'])
                self.assertIn('operator_settings' if mutation == 'readout' else mutation,
                              json.dumps(report['rejected_blocks']))

    def test_mixed_provenance_suppresses_aggregate_and_readiness(self):
        for row in self.fixture(seeds=(15, 16)):
            if row['seed'] == 16:
                row['provenance']['machine'] = 'other-fixture'
                self.refresh_fingerprint(row)
                self.save_record(row)
        code, report = self.run_summary(seeds=(15, 16))
        self.assertEqual(code, 2)
        self.assertFalse(report['summary'])
        self.assertTrue(report['warnings'])

    def test_planned_holm_family_keeps_missing_budget_contrasts(self):
        self.fixture(seeds=(15, 16, 17, 18, 19))
        code, report = self.run_summary(seeds=(15, 16, 17, 18, 19), budgets=(.05, .1))
        self.assertEqual(code, 2)
        self.assertEqual(len(report['missing_jobs']), 20)
        self.assertEqual(report['holm_family']['size'], 6)
        self.assertEqual(len(report['holm_family']['missing_contrasts']), 3)
        for contrast in report['contrasts']:
            self.assertAlmostEqual(contrast['p_signflip'], .0625)
            self.assertAlmostEqual(contrast['p_holm'], .375)

    def test_paired_statistics_and_holm(self):
        result = summary.paired_statistics([2, 2, 2, 2, 2])
        self.assertEqual(result['mean'], 2)
        self.assertEqual(result['sd'], 0)
        self.assertAlmostEqual(result['p_signflip'], 2 / 32)
        self.assertIsNone(summary.paired_statistics([2])['ci95'])
        self.assertEqual(summary.holm([.01, .04, .2]), [.03, .08, .2])


if __name__ == '__main__':
    unittest.main()

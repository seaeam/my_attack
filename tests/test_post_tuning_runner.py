import contextlib
import argparse
import ast
import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gbhaa_experiments import run_post_tuning as runner
from gbhaa_experiments.post_tuning_protocol import PROTOCOL, get_profile


class TuningRunnerTests(unittest.TestCase):
    def execution_fixture(self, directory):
        root=Path(directory)/'new'; parent_root=Path(directory)/'parents'
        args=SimpleNamespace(parent_root=parent_root,python='python',api_base_url='http://localhost:11434/v1',retry_failed=False)
        jobs=runner.make_jobs(['citeseer'],[15],.1,1,profile_ids=['baseline'])[:1]
        patches=[patch.object(runner,'source_manifest',return_value={'code.py':'a'*64}),
            patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')),
            patch.object(runner,'input_manifest',return_value={'data.npz':'b'*64}),
            patch.object(runner,'python_environment',return_value={'version':'fixture'}),
            patch.object(runner,'generation_model',return_value='fixture'),
            patch.object(runner,'check_unchanged_inputs'),
            patch('gbhaa_experiments.action_state_validation.validate_result',return_value={
                'fingerprint':'c'*64,'trajectory':{'actions':[[0,1,1]]}}),
            patch('gbhaa_experiments.post_tuning_validation.load_trial',return_value={
                'attack':{'checked':True},'evaluation':{'split':'validation'}}),
            patch('gbhaa_experiments.post_tuning_validation.validate_result')]
        for item in patches:
            item.start(); self.addCleanup(item.stop)
        return root,args,jobs

    def test_missing_runtime_dependency_stops_before_plan_or_child(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root,args,jobs=self.execution_fixture(directory)
            response=SimpleNamespace(returncode=1,stdout='',stderr="ModuleNotFoundError: No module named 'httpx'")
            success=SimpleNamespace(returncode=0,wall_time_seconds=.1,peak_rss_mib=None)
            with patch('subprocess.run',return_value=response),patch.object(runner,'run_command',return_value=success) as child:
                with self.assertRaisesRegex(RuntimeError,'httpx'):
                    runner._execute(args,jobs,root)
            child.assert_not_called()
            self.assertFalse((root/'plan.json').exists())

    def test_plan_mismatch_identifies_interpreter_difference_and_preserves_plan(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root,args,jobs=self.execution_fixture(directory)
            response=SimpleNamespace(returncode=0,wall_time_seconds=.1,peak_rss_mib=None)
            with patch.object(runner,'run_command',return_value=response):
                runner._execute(args,jobs,root)
            path=root/'plan.json';plan=json.loads(path.read_text())
            plan['provenance']['python_executable']='/wrong/base/python'
            path.write_text(json.dumps(plan));before=path.read_bytes()
            with patch.object(runner,'run_command') as child:
                with self.assertRaisesRegex(ValueError,'provenance.python_executable') as caught:
                    runner._execute(args,jobs,root)
            self.assertIn('/wrong/base/python',str(caught.exception))
            child.assert_not_called();self.assertEqual(path.read_bytes(),before)

    def test_launcher_prefers_active_conda_python_over_path_and_honors_override(self):
        launcher=runner.REPO_ROOT/'gbhaa_experiments/launchers/run_post_tuning.sh'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);conda=root/'conda';fallback=root/'base'
            for path,marker in ((conda/'bin/python','conda-python'),(fallback/'python','base-python')):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('#!/bin/bash\nprintf "%s\\n" "'+marker+'"\n')
                path.chmod(0o755)
            env=os.environ.copy();env.pop('PYTHON_BIN',None)
            env.update(CONDA_PREFIX=str(conda),PATH=str(fallback)+':/usr/bin:/bin')
            result=subprocess.run(['/bin/bash',str(launcher)],env=env,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),'conda-python')
            env['PYTHON_BIN']=str(fallback/'python')
            result=subprocess.run(['/bin/bash',str(launcher)],env=env,capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.strip(),'base-python')

    def test_execute_records_then_validates_and_resumes_exact_plan(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root,args,jobs=self.execution_fixture(directory)
            response=SimpleNamespace(returncode=0,wall_time_seconds=.1,peak_rss_mib=None)
            with patch.object(runner,'run_command',return_value=response) as child:
                self.assertEqual(runner._execute(args,jobs,root),0)
                saved=json.loads((runner.job_directory(root,jobs[0])/'result.json').read_text())
                self.assertEqual(saved['status'],'complete')
                self.assertEqual(saved['evaluation']['split'],'validation')
                self.assertEqual(runner._execute(args,jobs,root),0)
                self.assertEqual(child.call_count,1)
                changed=runner.make_jobs(['citeseer'],[15],.1,1,profile_ids=['floor025'])
                with self.assertRaisesRegex(ValueError,'Saved plan differs'):
                    runner._execute(args,changed,root)
                self.assertEqual(child.call_count,1)

    def test_failure_is_retained_and_retry_archives_original_attempt(self):
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            root,args,jobs=self.execution_fixture(directory)
            response=SimpleNamespace(returncode=1,wall_time_seconds=.1,peak_rss_mib=None)
            with patch.object(runner,'run_command',return_value=response) as child:
                with self.assertRaisesRegex(RuntimeError,'Child exited'):
                    runner._execute(args,jobs,root)
                path=runner.job_directory(root,jobs[0])/'result.json'
                self.assertEqual(json.loads(path.read_text())['status'],'failed')
                with self.assertRaisesRegex(ValueError,'retry-failed'):
                    runner._execute(args,jobs,root)
                self.assertEqual(child.call_count,1)
                args.retry_failed=True;response.returncode=0
                runner._execute(args,jobs,root)
                archives=list(path.parent.parent.glob(path.parent.name+'.failed-*'))
                self.assertEqual(len(archives),1)
                self.assertEqual(json.loads((archives[0]/'result.json').read_text())['status'],'failed')
                self.assertEqual(json.loads(path.read_text())['status'],'complete')

    def test_default_twelve_trial_plan_never_runs_or_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'results'; output = io.StringIO()
            with patch.object(runner, 'run_command', side_effect=AssertionError('no real run')), \
                    patch.object(runner, 'python_environment', side_effect=AssertionError('no environment probe')), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(runner.main(['--output-root',str(root)]),0)
            self.assertFalse(root.exists())
            self.assertEqual(output.getvalue().count('--tune-profile '),12)
            self.assertEqual(output.getvalue().count('--tune-arm endpoint_post'),10)
            self.assertEqual(output.getvalue().count('--tune-max-actions 60'),12)

    def test_profile_controls_reach_actual_cli_settings(self):
        jobs=runner.make_jobs(['citeseer'],[15],.1,60,profile_ids=['alpha010'])
        trial=next(j for j in jobs if j['profile_id']=='alpha010')
        command=runner.build_command(trial,'python',Path('/tmp/tune'),Path('/tmp/parents'),'http://localhost:11434/v1',Path('/tmp/cache'))
        self.assertEqual(command[command.index('--text_ppr_alpha')+1],'0.1')
        self.assertEqual(command[command.index('--text_ppr_iters')+1],'120')
        self.assertEqual(command[command.index('--global_seed_strategy')+1],'degree')
        self.assertIn('/tmp/parents/citeseer/ptb_0.1/seed_15/structure',command)

    def test_every_catalog_profile_is_accepted_by_actual_meta_argument_parser(self):
        # Execute only parser declarations: no dataset loading, training, or network.
        tree=ast.parse((runner.REPO_ROOT/'meta.py').read_text())
        declarations=[node for node in tree.body if (
            isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parser' for t in node.targets)
        ) or (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
              and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name)
              and node.value.func.value.id=='parser' and node.value.func.attr=='add_argument')]
        from dataset_support import DEEPROBUST_DATASETS, normalize_dataset_name
        scope={'argparse':argparse,'os':os,'DEEPROBUST_DATASETS':DEEPROBUST_DATASETS,
               'normalize_dataset_name':normalize_dataset_name}
        exec(compile(ast.Module(body=declarations,type_ignores=[]),'meta-parser','exec'),scope)
        for job in [j for profile in runner.profile_catalog() for j in
                    runner.make_jobs(['citeseer'],[15],.1,60,profile_ids=[profile])]:
            cmd=runner.build_command(job,'python',Path('/tmp/tune'),Path('/tmp/parent'),'http://localhost:11434/v1',Path('/tmp/cache'))
            # Six tuning flags precede the legacy parser arguments.
            legacy=cmd[cmd.index('--tune-max-actions')+2:]
            args=scope['parser'].parse_args(legacy)
            for key,value in runner.runtime_settings(job['config']).items():
                self.assertEqual(getattr(args,key),value)

    def test_recommendation_requires_complete_validation_selection_and_distinct_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recommendation.json'
            record=dict(protocol=PROTOCOL,status='complete',selection_split='validation',recommendations=[
                dict(dataset='citeseer',source_budget=.1,max_actions=60,seeds=[15],
                     profile_id='combined',config=get_profile('combined'))])
            path.write_text(json.dumps(record))
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(runner.main(['--recommendation',str(path),'--seeds','16']),0)
            self.assertEqual(output.getvalue().count('--tune-profile '),4)
            self.assertIn('--tune-profile combined --tune-arm endpoint_pre',output.getvalue())
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                runner.main(['--recommendation',str(path),'--seeds','15'])
            record['status']='incomplete';path.write_text(json.dumps(record))
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                runner.main(['--recommendation',str(path),'--seeds','16'])

    def test_legacy_result_roots_are_rejected(self):
        for path in ('followup_results_action_state','followup_results','results'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                runner.main(['--output-root',str(runner.REPO_ROOT/'gbhaa_experiments'/path/'tuning')])

    def test_refinement_confirmation_counts_for_baseline_and_new_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'recommendation.json'
            for profile, expected in [('baseline',12), ('a50_f03_diffused',16)]:
                path.write_text(json.dumps(dict(protocol=PROTOCOL,status='complete',selection_split='validation',
                    recommendations=[dict(dataset='citeseer',source_budget=.1,max_actions=60,seeds=[15],
                        profile_id=profile,config=get_profile(profile))])))
                output=io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(runner.main(['--recommendation',str(path),'--seeds','16','17','18','19',
                        '--max-actions','120','--output-root',str(Path(directory)/'results')]),0)
                self.assertEqual(output.getvalue().count('--tune-profile '),expected)
                self.assertEqual(output.getvalue().count('--tune-max-actions 120'),expected)
                self.assertFalse((Path(directory)/'results').exists())


if __name__=='__main__':unittest.main()

"""Seed-paired factorial contrasts for the four matched text-editor cells."""
import numpy as np
from scipy.stats import t
from gbhaa_experiments.post_tuning_protocol import EDIT_PROFILES, TRANSFER_PROFILES, OBJECTIVE_PROFILES


def factorial_effects(by_trial,seeds,cell,matrix_complete):
    return _effects(by_trial,seeds,cell,matrix_complete,EDIT_PROFILES,('gain','words'))


def transfer_effects(by_trial,seeds,cell,matrix_complete):
    return _effects(by_trial,seeds,cell,matrix_complete,TRANSFER_PROFILES,('proxy','residual'))


def objective_effects(by_trial,seeds,cell,matrix_complete):
    return _effects(by_trial,seeds,cell,matrix_complete,OBJECTIVE_PROFILES,('margin','local'))


def joint_advantage(by_trial,seeds,cell,matrix_complete,profiles):
    """Post must strictly beat both matched arms on ACC to count as a joint win."""
    rows=[]
    arms=('endpoint_post','endpoint_pre','parallel')
    for profile in profiles:
        paired=[]
        for seed in seeds:
            if not all(seed in by_trial.get((profile,arm),{}) for arm in arms):continue
            post,pre,parallel=[100*by_trial[profile,arm][seed]['evaluation']['metrics']['validation_accuracy'] for arm in arms]
            paired.append(dict(seed=seed,pre_minus_post_pp=pre-post,parallel_minus_post_pp=parallel-post,
                minimum_advantage_pp=min(pre-post,parallel-post),strict_joint_win=bool(post<pre and post<parallel)))
        complete=matrix_complete and len(paired)==len(seeds)
        values=np.array([p['minimum_advantage_pp'] for p in paired])
        mean=float(values.mean()) if complete and len(values) else None
        sd=float(values.std(ddof=1)) if complete and len(values)>1 else None
        half=float(t.ppf(.975,len(values)-1))*sd/np.sqrt(len(values)) if sd is not None else None
        rows.append(dict(dataset=cell[0],source_budget=cell[1],max_actions=cell[2],profile_id=profile,
            n=len(paired),valid_n=len(paired) if complete else 0,expected_n=len(seeds),
            status='complete' if complete else 'partial',
            strict_joint_wins=sum(p['strict_joint_win'] for p in paired) if complete else None,
            minimum_advantage_mean_pp=mean,minimum_advantage_sd_pp=sd,
            minimum_advantage_ci95_pp=[mean-half,mean+half] if half is not None else None,
            worst_seed_minimum_advantage_pp=float(values.min()) if complete and len(values) else None,
            paired_differences=paired,
            ci_method='unadjusted paired Student-t on within-seed minimum; descriptive development interval'))
    return rows


def _effects(by_trial,seeds,cell,matrix_complete,profiles,factors):
    rows=[]
    for arm in ('endpoint_post','endpoint_pre','parallel'):
        values={name:[] for name in (*factors,'interaction')}
        for seed in seeds:
            if not all(seed in by_trial.get((p,arm),{}) for p in profiles):continue
            c,g,w,b=[100*by_trial[p,arm][seed]['evaluation']['metrics']['validation_accuracy'] for p in profiles]
            for name,value in ((factors[0],((c-g)+(w-b))/2),(factors[1],((c-w)+(g-b))/2),('interaction',g+w-c-b)):
                values[name].append(dict(seed=seed,difference_pp=value))
        for name,differences in values.items():
            complete=matrix_complete and len(differences)==len(seeds)
            v=np.array([d['difference_pp'] for d in differences]);mean=float(v.mean()) if complete and len(v) else None
            sd=float(v.std(ddof=1)) if complete and len(v)>1 else None
            half=float(t.ppf(.975,len(v)-1))*sd/np.sqrt(len(v)) if sd is not None else None
            rows.append(dict(dataset=cell[0],source_budget=cell[1],max_actions=cell[2],arm=arm,effect=name,
                n=len(v),valid_n=len(v) if complete else 0,expected_n=len(seeds),status='complete' if complete else 'partial',
                mean_pp=mean,sd_pp=sd,ci95_pp=[mean-half,mean+half] if half is not None else None,
                paired_differences=differences,positive_means='lower attacked ACC / stronger attack'))
    return rows

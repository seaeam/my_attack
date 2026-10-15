"""Opt-in text editing primitives; candidates are scored only after projection."""
from __future__ import annotations
import numpy as np
from gbhaa_experiments.followup_protocol import seed_stream

VERSION = 'projected-text-editor-v1'

def word_plan(current, gradient, common, cap, guided, budget, width):
    x=np.asarray(current)[:width];g=np.asarray(gradient)[:width]
    if not np.isfinite(x).all() or not np.isfinite(g).all():raise ValueError('Nonfinite word-plan inputs')
    ids=np.arange(width)
    if guided:
        add=sorted(ids[(x==0)&(g>0)],key=lambda i:(-float(g[i]),int(i)))[:cap]
        drop=sorted(ids[(x>0)&(g<0)],key=lambda i:(float(g[i]*x[i]),int(i)))[:cap]
    else:
        add=list(common[:cap]);drop=[]
    # Match legacy active-word threshold, but never reinsert planned deletions.
    binary=np.all((x==0)|(x==1))
    active=np.flatnonzero(x==1 if binary else x>.1)
    keep=[int(i) for i in active if i not in drop][:budget]
    return dict(add=list(map(int,add)),drop=list(map(int,drop)),keep=keep,guided=bool(guided))

def project(current, raw, max_added, similarity_min, preferred):
    x=np.asarray(current,dtype=np.float32);y=np.asarray(raw,dtype=np.float32).copy()
    if x.shape!=y.shape or not np.isfinite(y).all() or np.any(y<0):raise ValueError('Invalid candidate vector')
    added=np.flatnonzero((x==0)&(y>0)).tolist()
    preferred=set(preferred)
    order=sorted(added,key=lambda i:(i not in preferred,-float(y[i]),i))
    y[order[max_added:]]=0
    nx=np.linalg.norm(x);ny=np.linalg.norm(y);fraction=1.
    cosine=float(x@y/(nx*ny)) if nx>1e-12 and ny>1e-12 else None
    if nx>1e-12 and ny<=1e-12 and similarity_min>0:
        # Cosine is undefined for zero; use the feasible endpoint of this projection.
        y=x.copy();fraction=0.
    elif cosine is not None and cosine<similarity_min:
        lo,hi=0.,1.
        for _ in range(20):
            mid=(lo+hi)/2;z=mid*y+(1-mid)*x;nz=np.linalg.norm(z)
            if nz<1e-12:break
            if float(x@z/(nx*nz))>=similarity_min:lo=mid
            else:hi=mid
        fraction=lo;y=lo*y+(1-lo)*x
    return y,dict(projection_fraction=fraction,raw_cosine=cosine,
                  added=int(np.sum((x==0)&(y>0))),changed_entries=int(np.count_nonzero(y!=x)))

def adapt_candidate(text, plan, current, vectorizer, vocab, max_added, similarity_min, residual=False):
    # Canonical token adaptation lets the actual vectorizer enforce exclusions.
    # The saved raw LLM response remains available separately from this BoW text.
    forbidden={vocab[i] for i in plan['drop']}
    if plan['guided']:
        forbidden.update(vocab[i] for i in range(min(len(vocab),len(current)))
                         if current[i]==0 and i not in plan['add'])
    tokens=[t for t in vectorizer.build_analyzer()(text) if t not in forbidden]
    tokens += [vocab[i] for i in dict.fromkeys(plan['keep']+plan['add'])]
    adapted=' '.join(tokens)
    bow=np.asarray(vectorizer.transform([adapted]).toarray()[0],dtype=np.float32)
    required=plan['keep']+plan['add']
    if any(bow[i]<=0 for i in required) or any(bow[i]!=0 for i in plan['drop']):
        raise ValueError('Vectorizer cannot enforce the requested word plan')
    aligned=np.zeros_like(current,dtype=np.float32);width=min(len(bow),len(aligned));aligned[:width]=bow[:width]
    if residual:
        if not plan['guided']:raise ValueError('Residual edits require an explicit signed word plan')
        # This is a feature patch supported by vectorized words, not a complete
        # text replacement. Unmapped columns and unplanned fractional values stay.
        aligned=np.asarray(current,dtype=np.float32).copy()
        aligned[plan['add']]=bow[plan['add']]
        aligned[plan['drop']]=0
    result,info=project(current,aligned,max_added,similarity_min,plan['add'])
    if residual:
        allowed=np.zeros(len(current),dtype=bool);allowed[plan['add']+plan['drop']]=True
        result[~allowed]=current[~allowed]  # Eliminate blend roundoff off support.
        info.update(edit_policy='planned_residual',changed_entries=int(np.count_nonzero(result!=current)),
                    unmapped_changed_entries=int(np.count_nonzero(result[width:]!=current[width:])),
                    actual_l1=float(np.abs(result-current).sum()))
    info.update(adapted_text=adapted,actual_dropped=sum(result[i]<current[i] for i in plan['drop']),
                actual_added=sum(result[i]>current[i] for i in plan['add']))
    # Convert numpy scalar counters to JSON-safe ints.
    info['actual_dropped']=int(info['actual_dropped']);info['actual_added']=int(info['actual_added'])
    info['unplanned_decreased_entries']=sum(int(i not in plan['drop']) for i in np.flatnonzero(result<current))
    info['unplanned_increased_entries']=sum(int(i not in plan['add']) for i in np.flatnonzero(result>current))
    return result,info

def choose_candidate(losses, gain, seed, round_index, node):
    losses=np.asarray(losses,dtype=np.float64)
    if losses.ndim!=1 or not len(losses) or not np.isfinite(losses).all():raise ValueError('Invalid candidate losses')
    if gain:return int(np.argmax(losses))
    return int(np.random.default_rng(seed_stream(seed,f'editor-choice:{node}',round_index)).integers(len(losses)))

def common_order(features,nodes,width):
    x=np.asarray(features);delta=x[:,:width].mean(0)-x[nodes,:width].mean(0)
    scores=np.maximum(delta,0)
    if scores.max(initial=0)<=1e-12:scores=np.abs(delta)
    return np.lexsort((np.arange(width),-scores)).tolist()

def messages_for(plans,vocab,slot):
    # Prompt content depends on the word policy, never on the candidate-choice factor or arm name.
    constraints=[dict(node=int(node),include=[vocab[i] for i in p['add']],avoid=[vocab[i] for i in p['drop']])
                 for node,p in sorted(plans.items())]
    import json
    return [dict(role='system',content='Write one short research abstract. Return only the abstract; no alternatives or delimiters.'),
            dict(role='user',content=f'Candidate {slot+1}. Write a shared template for these node-specific edits. '
                 'Include the requested concepts where possible. Each node is adapted separately afterward. '
                 +json.dumps(constraints,ensure_ascii=False,sort_keys=True))]

"""Run all assignment experiments with a single immutable saved split.

Usage: python -m src.experiment --config configs/experiment.json
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sklearn
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from .models import Bagging, SAMME, aligned_proba, leaf_stats, disagreement, margin, samme_update

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def rows_csv(name, rows):
    path = ROOT / 'results' / name
    path.parent.mkdir(exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def acc(p, y):
    return float(np.mean(p.argmax(1) == y))


def stream(seed, *keys):
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1)[0])


def prepare(config):
    seed = int(''.join(c for c in config['roll_number'] if c.isdigit()))
    data = fetch_olivetti_faces(data_home=ROOT / 'data/cache', shuffle=False)
    X, y = data.data, data.target
    assert X.shape == (400, 4096) and np.all(np.bincount(y) == 10)
    split_path = ROOT / 'data/split.json'
    digest = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    if split_path.exists():
        split = json.loads(split_path.read_text())
        assert split['seed'] == seed and split['dataset_sha256'] == digest
    else:
        rng = np.random.default_rng(seed)
        split = dict(seed=seed, roll_number=config['roll_number'], dataset_sha256=digest,
                     train=[], validation=[], test=[])
        for identity in range(40):
            ix = rng.permutation(np.flatnonzero(y == identity))
            for name, subset in [('train', ix[:6]), ('validation', ix[6:8]), ('test', ix[8:])]:
                split[name].extend(subset.tolist())
        save_json(split_path, split)
    combined = sum((split[n] for n in ['train', 'validation', 'test']), [])
    assert sorted(combined) == list(range(400))
    for name, count in [('train', 6), ('validation', 2), ('test', 2)]:
        assert np.all(np.bincount(y[split[name]]) == count)
    train, val, test = [np.asarray(split[n]) for n in ['train','validation','test']]
    masked = X[test].reshape(-1,64,64).copy()
    masked[:,24:40,24:40] = 0
    assert np.all(masked[:,24:40,24:40] == 0)
    outside = np.ones((64,64), dtype=bool)
    outside[24:40,24:40] = False
    np.testing.assert_array_equal(masked[:,outside], X[test].reshape(-1,64,64)[:,outside])
    return seed, X, y, train, val, test, masked.reshape(-1,4096)


def toy():
    X = np.arange(9, dtype=float).reshape(-1,1)
    y = np.array([0,0,0,1,1,1,2,2,2])
    tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X,y)
    pred = tree.predict(X)
    w = np.full(9,1/9)
    error, alpha, new, status = samme_update(w,pred != y,3)
    save_json(ROOT/'results/toy_samme.json', dict(X=X.tolist(), targets=y.tolist(),
        initial_weights=w.tolist(), predictions=pred.tolist(), weighted_error=error,
        alpha=alpha, updated_weights=new.tolist(), status=status))


def refit_job(args):
    r, config, selected, resample, Xtr, ytr, Xt, yt = args
    seed = int(''.join(c for c in config['roll_number'] if c.isdigit()))
    XX, yy = Xtr[resample], ytr[resample]
    outputs, rows = {}, []
    depths = selected['bias_variance_depths']
    unrestricted = None
    for depth in depths:
        tag = str(depth)
        if depth == 64 and unrestricted is not None and unrestricted[0].get_depth() < 64:
            tree = unrestricted[0]
        else:
            tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=selected['leaf'],
                 random_state=stream(seed,30,r)).fit(XX,yy)
        pt = aligned_proba(tree,Xt,40)
        outputs['tree_'+tag] = pt
        stats = leaf_stats(tree,XX)
        rows.append(dict(repetition=r, model='tree', depth=tag, accuracy=acc(pt,yt),
                         disagreement=None, **stats))
        B = max(100,selected['bag_B']) if depth == selected['depth'] else 100
        if depth == 64 and unrestricted is not None and all(t.get_depth() < 64 for t in unrestricted[1].trees[:B]):
            # With identical bootstrap/tree seeds, a never-reached depth cap
            # gives exactly the same trees. Reuse this proven identical fit.
            bag = unrestricted[1]
        else:
            bag = Bagging(B,depth,selected['leaf'],stream(seed,31,r),
                          compress_duplicates=(selected['leaf']==1)).fit(XX,yy)
        outputs['bag100_'+tag] = bag.predict_proba(Xt,100)
        rows.append(dict(repetition=r, model='bag100', depth=tag,
            accuracy=acc(outputs['bag100_'+tag],yt), disagreement=disagreement(bag.predictions(Xt,100)),
            **{k:float(np.mean([leaf_stats(t,XX[s])[k] for t,s in zip(bag.trees[:100],bag.samples[:100])]))
               for k in stats}))
        if depth == selected['depth']:
            outputs['tree'] = pt
            outputs['bagging'] = bag.predict_proba(Xt,selected['bag_B'])
        if depth is None:
            unrestricted = (tree,bag)
    boost = SAMME(selected['boost_B'],selected['boost_depth'],stream(seed,32,r)).fit(XX,yy)
    outputs['boosting'] = boost.predict_proba(Xt)
    path = ROOT/f'results/refits/refit_{r:02d}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**outputs)
    save_json(path.with_suffix('.json'),rows)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/experiment.json')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume-refits',action='store_true')
    parser.add_argument('--refits-only',action='store_true',help='Reuse frozen selections and completed grid logs')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    toy()
    seed,X,y,tr,va,te,masked = prepare(config)
    Xtr,ytr = X[tr],y[tr]
    if args.refits_only:
        selected=json.loads((ROOT/'results/selected.json').read_text())
        run_refits(seed,config,selected,Xtr,ytr,X,y,te,args.workers,args.resume_refits)
        return
    datasets = {'train':Xtr,'validation':X[va],'test':X[te],'masked':masked}
    targets = {'train':ytr,'validation':y[va],'test':y[te],'masked':y[te]}
    print('Data/split verified; toy SAMME round logged.',flush=True)
    baselines, trees = [], []
    for depth in config['tree_depths']:
        for leaf in config['tree_leaves']:
            tree = DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,
                random_state=stream(seed,1)).fit(Xtr,ytr)
            row = dict(depth=depth,leaf=leaf,**leaf_stats(tree,Xtr))
            row.update({n:acc(aligned_proba(tree,z,40),targets[n]) for n,z in datasets.items() if n!='masked'})
            baselines.append(row)
            trees.append(tree)
    # Iteration order breaks validation ties: shallower depth, then smaller leaf.
    best = int(np.argmax([r['validation'] for r in baselines]))
    tree = trees[best]
    depth,leaf = baselines[best]['depth'],baselines[best]['leaf']
    rows_csv('tree_grid.csv',baselines)
    print('Frozen baseline:',baselines[best],flush=True)
    bag = Bagging(max(config['bag_sizes']),depth,leaf,stream(seed,2)).fit(Xtr,ytr)
    bagrows=[]
    for B in config['bag_sizes']:
        oob,coverage=bag.oob(Xtr,ytr,B)
        probs={n:bag.predict_proba(z,B) for n,z in datasets.items()}
        bagrows.append(dict(B=B,**{n:acc(p,targets[n]) for n,p in probs.items() if n!='masked'},
             distinct_mean=float(np.mean([len(np.unique(s)) for s in bag.samples[:B]])),
             distinct_theory=float(len(tr)*(1-(1-1/len(tr))**len(tr))),oob_accuracy=oob,
             oob_coverage=coverage,disagreement=disagreement(bag.predictions(X[va],B)),
             vote_margin=margin(probs['validation']),train_seconds=bag.times[B-1]))
    bag_B=bagrows[int(np.argmax([r['validation'] for r in bagrows]))]['B']
    rows_csv('bagging_grid.csv',bagrows)
    print('Frozen bagging B:',bag_B,flush=True)
    boosts,boostrows={},[]
    for d in config['boost_depths']:
        model=SAMME(max(config['boost_rounds']),d,stream(seed,3,d)).fit(Xtr,ytr)
        boosts[d]=model
        for B in config['boost_rounds']:
            boostrows.append(dict(depth=d,B=B,accepted_rounds=min(B,len(model.trees)),
                **{n:acc(model.predict_proba(z,B),targets[n]) for n,z in datasets.items() if n!='masked'},
                train_seconds=model.times[min(B,len(model.trees))-1],stop_reason=model.stop_reason))
        print('Boosting depth',d,'completed',flush=True)
    selected_boost=boostrows[int(np.argmax([r['validation'] for r in boostrows]))]
    boost_depth,boost_B=selected_boost['depth'],selected_boost['B']
    boost=boosts[boost_depth]
    rows_csv('boosting_grid.csv',boostrows)
    # Fixed depth choices based on selected depth, never on test results.
    shallow=2 if depth != 2 else 1
    deep=None if depth is not None else 64
    bv_depths=[shallow,depth,deep] if depth is not None else [2,16,None,64]
    selected=dict(depth=depth,leaf=leaf,bag_B=bag_B,boost_depth=boost_depth,boost_B=boost_B,
                  bias_variance_depths=bv_depths,tie_rule='first maximum in configured grid order')
    save_json(ROOT/'results/selected.json',selected)
    print('All selections frozen:',selected,flush=True)
    rows_csv('boosting_history.csv',boost.history[:boost_B])
    np.savez_compressed(ROOT/'results/boosting_history.npz',weights=boost.sample_weights[:boost_B+1],
        predictions=boost.train_predictions[:boost_B],train_indices=tr,targets=ytr)
    final,arrays=[],{}
    for name in ['tree','bagging','boosting']:
        for subset,z in datasets.items():
            p=aligned_proba(tree,z,40) if name=='tree' else (
                bag.predict_proba(z,bag_B) if name=='bagging' else boost.predict_proba(z,boost_B))
            arrays[name+'_'+subset]=p
            final.append(dict(model=name,subset=subset,accuracy=acc(p,targets[subset]),
                              confidence=float(p.max(1).mean())))
    arrays.update(test_indices=te,targets=y[te],
        bagging_clean_predictions=bag.predictions(X[te],bag_B),
        bagging_masked_predictions=bag.predictions(masked,bag_B),
        boosting_clean_predictions=boost.predictions(X[te],boost_B),
        boosting_masked_predictions=boost.predictions(masked,boost_B),
        boosting_alphas=boost.alphas[:boost_B])
    np.savez_compressed(ROOT/'results/final_predictions.npz',**arrays)
    rows_csv('final_metrics.csv',final)
    compare=[]
    for B in [25,50,100]:
        br=next(r for r in bagrows if r['B']==B)
        sr=next(r for r in boostrows if r['B']==B and r['depth']==boost_depth)
        for name,row in [('bagging',br),('boosting',sr)]:
            compare.append(dict(model=name,B=B,**{k:row[k] for k in ['train','validation','test','train_seconds']}))
    rows_csv('matched_learners.csv',compare)
    run_refits(seed,config,selected,Xtr,ytr,X,y,te,args.workers,args.resume_refits)


def run_refits(seed,config,selected,Xtr,ytr,X,y,te,workers,resume_refits):
    resample_path=ROOT/'data/refit_indices.npy'
    if resample_path.exists():
        resamples=np.load(resample_path)
    else:
        rng=np.random.default_rng(stream(seed,4))
        resamples=np.array([np.concatenate([rng.choice(np.flatnonzero(ytr==c),6,replace=True)
                                           for c in range(40)]) for _ in range(30)])
        np.save(resample_path,resamples)
    assert resamples.shape==(30,240)
    assert all(np.all(np.bincount(ytr[s])==6) for s in resamples)
    jobs=[(r,config,selected,resamples[r],Xtr,ytr,X[te],y[te]) for r in range(30)
          if not(resume_refits and (ROOT/f'results/refits/refit_{r:02d}.npz').exists()
                 and (ROOT/f'results/refits/refit_{r:02d}.json').exists())]
    print('Starting',len(jobs),'refits with',workers,'workers.',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(refit_job,jobs):
            print('Completed refit',r+1,'/ 30',flush=True)
    summaries,depth_rows=[],[]
    loaded=[np.load(ROOT/f'results/refits/refit_{r:02d}.npz') for r in range(30)]
    for r in range(30):
        depth_rows.extend(json.loads((ROOT/f'results/refits/refit_{r:02d}.json').read_text()))
    onehot=np.eye(40)[y[te]]
    for name in loaded[0].files:
        p=np.stack([a[name] for a in loaded]); mean=p.mean(0)
        bias=float(np.mean(np.sum((mean-onehot)**2,axis=1)))
        variance=float(np.mean(np.sum((p-mean)**2,axis=2)))
        mse=float(np.mean(np.sum((p-onehot)**2,axis=2)))
        assert np.isclose(bias+variance,mse,atol=1e-12)
        scores=np.mean(p.argmax(2)==y[te],axis=1)
        summaries.append(dict(model=name,bias_squared=bias,variance=variance,brier=mse,
                              accuracy_mean=float(scores.mean()),accuracy_sd=float(scores.std(ddof=1))))
    rows_csv('bias_variance.csv',summaries)
    rows_csv('depth_refits.csv',depth_rows)
    save_json(ROOT/'results/environment.json',dict(python=platform.python_version(),numpy=np.__version__,
        sklearn=sklearn.__version__,platform=platform.platform(),processor=platform.processor(),
        config=config,workers=workers,refit_bootstrap='integer counts of duplicate rows; literal grid bootstrap timings'))
    print('COMPLETE: all metrics and 30 refits saved.',flush=True)


if __name__=='__main__':
    main()

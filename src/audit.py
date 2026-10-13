"""Independent checks of persisted split, predictions, selection and decomposition."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from sklearn.datasets import fetch_olivetti_faces

ROOT=Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT/'results'/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    checks=[]
    split=json.loads((ROOT/'data/split.json').read_text())
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    selected=json.loads((ROOT/'results/selected.json').read_text())
    data=fetch_olivetti_faces(data_home=ROOT/'data/cache',shuffle=False)
    seed=int(''.join(c for c in cfg['roll_number'] if c.isdigit()))
    assert split['seed']==seed==26002
    assert hashlib.sha256(data.data.tobytes()+data.target.tobytes()).hexdigest()==split['dataset_sha256']
    rng=np.random.default_rng(seed)
    regenerated={key:[] for key in ['train','validation','test']}
    for c in range(40):
        ix=rng.permutation(np.flatnonzero(data.target==c))
        for key,v in [('train',ix[:6]),('validation',ix[6:8]),('test',ix[8:])]:
            regenerated[key].extend(v.tolist())
    assert all(regenerated[k]==split[k] for k in regenerated)
    assert sorted(sum(regenerated.values(),[]))==list(range(400))
    checks.append('Saved split matches default_rng(26002) exactly; disjoint 6/2/2 per identity')
    for model,file in [('tree','tree_grid.csv'),('bagging','bagging_grid.csv'),('boosting','boosting_grid.csv')]:
        rr=rows(file); winner=max(rr,key=lambda r:float(r['validation']))
        if model=='tree':
            assert (None if winner['depth']=='' else int(winner['depth']))==selected['depth']
            assert int(winner['leaf'])==selected['leaf']
        elif model=='bagging': assert int(winner['B'])==selected['bag_B']
        else:
            assert int(winner['depth'])==selected['boost_depth'] and int(winner['B'])==selected['boost_B']
    checks.append('Frozen settings reproduce first-maximum validation-only grid selections')
    a=np.load(ROOT/'results/final_predictions.npz')
    metrics=rows('final_metrics.csv')
    for m in metrics:
        p=a[m['model']+'_'+m['subset']]
        target=data.target[split['test' if m['subset']=='masked' else m['subset']]]
        assert p.shape==(len(target),40)
        assert np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
        assert np.isclose(np.mean(p.argmax(1)==target),float(m['accuracy']))
        assert np.isclose(p.max(1).mean(),float(m['confidence']))
    # Recompute weighted and unweighted test votes from individual learners.
    for subset,predkey in [('test','clean'),('masked','masked')]:
        for model in ['bagging','boosting']:
            pred=a[f'{model}_{predkey}_predictions']; totals=np.zeros((80,40))
            weights=np.ones(len(pred)) if model=='bagging' else a['boosting_alphas']
            for w,pr in zip(weights,pred): totals[np.arange(80),pr]+=w
            np.testing.assert_allclose(totals/weights.sum(),a[model+'_'+subset],atol=1e-12)
    checks.append('Every accuracy/confidence recomputed; clean/masked votes independently reconstructed')
    h=np.load(ROOT/'results/boosting_history.npz'); hh=rows('boosting_history.csv')
    np.testing.assert_allclose(h['weights'].sum(1),1,atol=1e-12)
    for t,row in enumerate(hh):
        w=h['weights'][t]; wrong=h['predictions'][t]!=h['targets']
        err=float(w@wrong); alpha=np.log((1-err)/err)+np.log(39)
        assert np.isclose(err,float(row['error'])) and np.isclose(alpha,float(row['alpha']))
        expected=w*np.exp(alpha*wrong); expected/=expected.sum()
        np.testing.assert_allclose(expected,h['weights'][t+1],atol=1e-12)
    checks.append('All selected SAMME rounds independently checked with direct multiplicative update')
    refit=np.load(ROOT/'data/refit_indices.npy')
    ytr=data.target[split['train']]
    assert refit.shape==(30,240) and refit.min()>=0 and refit.max()<240
    assert all(np.all(np.bincount(ytr[s])==6) for s in refit)
    refit_seed=int(np.random.SeedSequence([seed,4]).generate_state(1)[0])
    rng=np.random.default_rng(refit_seed)
    expected=np.array([np.concatenate([rng.choice(np.flatnonzero(ytr==c),6,replace=True)
                                       for c in range(40)]) for _ in range(30)])
    np.testing.assert_array_equal(refit,expected)
    arrays=[np.load(ROOT/f'results/refits/refit_{r:02d}.npz') for r in range(30)]
    targets=data.target[split['test']]; eye=np.eye(40)[targets]
    for row in rows('bias_variance.csv'):
        pp=np.stack([x[row['model']] for x in arrays]); mean=pp.mean(0)
        assert pp.shape==(30,80,40)
        np.testing.assert_allclose(pp.sum(2),1,atol=1e-12)
        bias=np.square(mean-eye).sum(1).mean()
        variance=np.square(pp-mean).sum(2).mean()
        brier=np.square(pp-eye).sum(2).mean()
        scores=(pp.argmax(2)==targets).mean(1)
        for value,key in [(bias,'bias_squared'),(variance,'variance'),(brier,'brier'),
                          (scores.mean(),'accuracy_mean'),(scores.std(ddof=1),'accuracy_sd')]:
            assert np.isclose(value,float(row[key]),atol=1e-12)
        assert np.isclose(bias+variance,brier,atol=1e-12)
    if selected['depth'] is None:
        for a_refit in arrays:
            np.testing.assert_array_equal(a_refit['tree_None'],a_refit['tree_64'])
            np.testing.assert_array_equal(a_refit['bag100_None'],a_refit['bag100_64'])
            np.testing.assert_array_equal(a_refit['tree_None'],a_refit['tree'])
            if selected['bag_B']==100:
                np.testing.assert_array_equal(a_refit['bag100_None'],a_refit['bagging'])
    checks.append('All 30 stratified refits validated; bias/variance/Brier/accuracy recomputed')
    pdf=ROOT/'output/pdf/ML_Assignment_2_Report.pdf'
    assert pdf.exists(), 'Generate the final PDF before running the audit'
    if pdf.exists():
        reader=PdfReader(pdf)
        assert len(reader.pages)==8
        assert all(float(p.mediabox.width)==612 and float(p.mediabox.height)==792 for p in reader.pages)
        page1=reader.pages[0].extract_text()
        assert cfg['name'] in page1 and cfg['roll_number'] in page1 and cfg['repository'] in page1
        git_head=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=ROOT,capture_output=True,text=True)
        head=git_head.stdout.strip() if git_head.returncode==0 else ''
        assert head, 'Commit the source before auditing the final PDF'
        assert head in page1
        checks.append('PDF: 8 Letter pages; page 1 includes confirmed identity, repository and current source commit')
    report=dict(status='PASS',checks=checks)
    (ROOT/'results/audit.json').write_text(json.dumps(report,indent=2))
    for check in checks: print('PASS:',check)


if __name__=='__main__': main()

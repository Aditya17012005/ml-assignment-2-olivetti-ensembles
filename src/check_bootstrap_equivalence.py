"""Validate exact duplicate-count optimization on actual saved face resamples."""
import json
import time
import numpy as np
from .experiment import ROOT,prepare,stream,save_json
from .models import Bagging,aligned_proba


def main():
    config=json.loads((ROOT/'configs/experiment.json').read_text())
    seed,X,y,tr,va,te,masked=prepare(config)
    resamples=np.load(ROOT/'data/refit_indices.npy')
    records=[]
    for r in range(3):
        ix=tr[resamples[r]]
        for d in [2,16,None]:
            start=time.perf_counter()
            a=Bagging(3,d,1,stream(seed,31,r)).fit(X[ix],y[ix])
            literal=time.perf_counter()-start
            start=time.perf_counter()
            b=Bagging(3,d,1,stream(seed,31,r),compress_duplicates=True).fit(X[ix],y[ix])
            compressed=time.perf_counter()-start
            np.testing.assert_array_equal(a.samples,b.samples)
            for ta,tb in zip(a.trees,b.trees):
                np.testing.assert_array_equal(ta.tree_.feature,tb.tree_.feature)
                np.testing.assert_array_equal(ta.tree_.threshold,tb.tree_.threshold)
                np.testing.assert_array_equal(aligned_proba(ta,X,40),aligned_proba(tb,X,40))
            records.append(dict(repetition=r,depth=d,trees=3,status='identical splits and probabilities on all 400 faces',
                                literal_seconds=literal,count_seconds=compressed))
            print('PASS: refit',r,'depth',d,'three trees',flush=True)
    save_json(ROOT/'results/bootstrap_equivalence.json',records)


if __name__=='__main__': main()

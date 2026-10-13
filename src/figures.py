"""Regenerate every figure and case-study log from saved experimental outputs."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':220,'font.family':'DejaVu Sans'})
COLORS=['#335d92','#23856a','#c46b32']


def rows(name):
    with (ROOT/'results'/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(rows,key):
    return np.array([float(r[key]) if r[key] else np.nan for r in rows])


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white',
                metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def main():
    selected=json.loads((ROOT/'results/selected.json').read_text())
    data=fetch_olivetti_faces(data_home=ROOT/'data/cache',shuffle=False)
    a=np.load(ROOT/'results/final_predictions.npz')
    b=rows('bagging_grid.csv'); B=num(b,'B')
    fig,axes=plt.subplots(2,2,figsize=(6.8,4.8),layout='constrained')
    for i,n in enumerate(['train','validation','test']):
        axes[0,0].plot(B,num(b,n)*100,'o-',label=n,color=COLORS[i])
    axes[0,0].set_ylabel('Accuracy (%)'); axes[0,0].legend(fontsize=8)
    axes[0,1].plot(B,num(b,'oob_accuracy')*100,'o-',label='OOB accuracy')
    axes[0,1].plot(B,num(b,'oob_coverage')*100,'s--',label='OOB coverage')
    axes[0,1].set_ylabel('Percent'); axes[0,1].legend(fontsize=8)
    axes[1,0].plot(B,num(b,'disagreement'),'o-',color=COLORS[1]); axes[1,0].set_ylabel('Validation disagreement')
    axes[1,1].plot(B,num(b,'vote_margin'),'o-',color=COLORS[2]); axes[1,1].set_ylabel('Validation vote margin')
    for ax in axes.flat:
        ax.set_xscale('log'); ax.set_xticks(B,labels=B.astype(int)); ax.set_xlabel('Number of trees B')
        ax.axvline(selected['bag_B'],color='gray',ls=':',lw=1); ax.grid(alpha=.18)
    save(fig,'bagging_curves')
    history=rows('boosting_history.csv'); rounds=num(history,'round')
    fig,axes=plt.subplots(2,2,figsize=(6.8,4.3),layout='constrained')
    for ax,key,label in zip(axes.flat,['error','alpha','max_weight_after','top10_weight_after'],
        ['Weighted training error','Learner weight alpha','Maximum sample weight (after)','Top-10 total weight (after)']):
        ax.plot(rounds,num(history,key),color=COLORS[2],lw=1.4)
        ax.set_xlabel('Round'); ax.set_ylabel(label); ax.grid(alpha=.2)
    save(fig,'boosting_traces')
    truth=a['targets']; tree=a['tree_test'].argmax(1); bag=a['bagging_test'].argmax(1)
    corrected=np.flatnonzero((tree!=truth)&(bag==truth))[:2]
    failed=np.flatnonzero((tree!=truth)&(bag!=truth))[:2]
    cases=[]
    fig,axes=plt.subplots(2,2,figsize=(4.6,4.0),layout='constrained')
    for ax,i in zip(axes.flat,np.concatenate([corrected,failed])):
        ix=int(a['test_indices'][i]); counts=np.bincount(a['bagging_clean_predictions'][:,i],minlength=40)
        votes={str(k):int(v) for k,v in enumerate(counts) if v}
        cases.append(dict(test_position=int(i),dataset_index=ix,target=int(truth[i]),
            tree_prediction=int(tree[i]),bagging_prediction=int(bag[i]),votes=votes,
            kind='corrected' if bag[i]==truth[i] else 'both_wrong'))
        ax.imshow(data.images[ix],cmap='gray',vmin=0,vmax=1); ax.axis('off')
        ax.set_title(f'Image {ix}; true {truth[i]}\nTree {tree[i]}, bag {bag[i]}',fontsize=11)
    save(fig,'bagging_cases')
    h=np.load(ROOT/'results/boosting_history.npz')
    wr,wi=np.unravel_index(np.argmax(h['weights'][1:]),h['weights'][1:].shape); wr+=1
    weights=h['weights'][:,wi]; preds=h['predictions'][:,wi]; target=int(h['targets'][wi])
    heavy=dict(training_position=int(wi),dataset_index=int(h['train_indices'][wi]),target=target,
        peak_round=int(wr),peak_weight=float(weights[wr]),predictions=preds.tolist(),weights=weights.tolist())
    fig=plt.figure(figsize=(3.3,1.6),layout='constrained'); gs=fig.add_gridspec(1,3)
    ax=fig.add_subplot(gs[0,0]); ax.imshow(data.images[heavy['dataset_index']],cmap='gray',vmin=0,vmax=1)
    ax.axis('off'); ax.set_title(f"Image {heavy['dataset_index']}\nclass {target}",fontsize=8)
    ax=fig.add_subplot(gs[0,1:]); ax.plot(np.arange(len(weights)),weights,color=COLORS[2],label='Sample weight')
    wrong=np.flatnonzero(preds!=target)+1
    ax.scatter(wrong,weights[wrong],color='#9f273c',s=12,label='Misclassified that round',zorder=3)
    ax.axvline(wr,ls=':',color='gray'); ax.set_xlabel('Round',fontsize=8); ax.set_ylabel('Weight',fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=.2); save(fig,'heavy_sample')
    bp=a['boosting_test'].argmax(1); bm=a['bagging_masked'].argmax(1); sm=a['boosting_masked'].argmax(1)
    changed_b=(bag==truth)!=(bm==truth); changed_s=(bp==truth)!=(sm==truth)
    candidate=np.flatnonzero(changed_b != changed_s)
    if not len(candidate):
        candidate=np.flatnonzero((bag!=bm)!=(bp!=sm))
    i=int(candidate[0]) if len(candidate) else int(np.argmax(np.abs(a['bagging_test']-a['bagging_masked']).sum(1)))
    ix=int(a['test_indices'][i]); original=data.images[ix]; masked=original.copy(); masked[24:40,24:40]=0
    robust=dict(test_position=i,dataset_index=ix,target=int(truth[i]),
        bagging_clean=int(bag[i]),bagging_masked=int(bm[i]),boosting_clean=int(bp[i]),boosting_masked=int(sm[i]))
    for label,key in [('clean','bagging_clean_predictions'),('masked','bagging_masked_predictions')]:
        counts=np.bincount(a[key][:,i],minlength=40); total=counts.sum()
        robust['bagging_'+label+'_counts']={str(k):int(v) for k,v in enumerate(counts) if v}
        robust['bagging_'+label+'_disagreement']=float(1-np.sum(counts*(counts-1))/(total*(total-1))) if total>1 else None
    robust['bagging_changed_tree_fraction']=float(np.mean(a['bagging_clean_predictions'][:,i]!=a['bagging_masked_predictions'][:,i]))
    changed=a['boosting_clean_predictions'][:,i]!=a['boosting_masked_predictions'][:,i]
    robust['boosting_changed_alpha_fraction']=float(a['boosting_alphas'][changed].sum()/a['boosting_alphas'].sum())
    fig,axes=plt.subplots(1,2,figsize=(4.6,2.5),layout='constrained')
    for ax,im,label in zip(axes,[original,masked],['Clean','Masked']):
        ax.imshow(im,cmap='gray',vmin=0,vmax=1); ax.axis('off'); ax.set_title(label+f'; image {ix}, true {truth[i]}',fontsize=10.5)
    save(fig,'robustness_face')
    fig,axes=plt.subplots(2,1,figsize=(3.3,3.0),layout='constrained')
    for ax,name in zip(axes,['bagging','boosting']):
        p,q=a[name+'_test'][i],a[name+'_masked'][i]
        labels=np.unique(np.r_[np.argsort(p)[-4:],np.argsort(q)[-4:],truth[i]])
        xx=np.arange(len(labels)); ax.bar(xx-.18,p[labels],.36,label='Clean',color=COLORS[1])
        ax.bar(xx+.18,q[labels],.36,label='Masked',color=COLORS[2]); ax.set_xticks(xx,labels=labels)
        ax.set_xlabel('Class'); ax.set_ylabel('Normalized vote'); ax.set_title(name.capitalize())
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,fontsize=8)
    save(fig,'robustness_votes')
    (ROOT/'results/case_studies.json').write_text(json.dumps(dict(bagging=cases,heavy_sample=heavy,robustness=robust),indent=2))
    if not (ROOT/'results/bias_variance.csv').exists():
        print('Grid, sample and robustness figures complete; bias/variance refits still pending.')
        return
    bv=rows('bias_variance.csv'); dr=rows('depth_refits.csv')
    chosen=[next(r for r in bv if r['model']==m) for m in ['tree','bagging','boosting']]
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.8),layout='constrained'); xx=np.arange(3)
    axes[0].bar(xx,num(chosen,'bias_squared'),label='Bias squared',color=COLORS[0])
    axes[0].bar(xx,num(chosen,'variance'),bottom=num(chosen,'bias_squared'),label='Variance',color=COLORS[2])
    axes[0].set_xticks(xx,labels=['Tree','Bagging','Boosting']); axes[0].set_ylabel('Squared vector error'); axes[0].legend(fontsize=8)
    axes[1].errorbar(xx,num(chosen,'accuracy_mean')*100,yerr=num(chosen,'accuracy_sd')*100,fmt='o',capsize=5,color=COLORS[1])
    axes[1].set_xticks(xx,labels=['Tree','Bagging','Boosting']); axes[1].set_ylabel('Refit accuracy: mean +/- SD (%)'); axes[1].grid(alpha=.2)
    save(fig,'bias_variance')
    depths=selected['bias_variance_depths']; labels=['None' if d is None else str(d) for d in depths]
    fig,axes=plt.subplots(1,2,figsize=(6.8,2.8),layout='constrained')
    for j,name in enumerate(['tree','bag100']):
        rr=[next(r for r in bv if r['model']==name+'_'+str(d)) for d in depths]
        axes[0].plot(labels,num(rr,'variance'),'o-',label=name,color=COLORS[j])
        axes[1].plot(labels,num(rr,'accuracy_mean')*100,'o-',label=name,color=COLORS[j])
    axes[0].set_ylabel('Variance'); axes[1].set_ylabel('Mean refit test accuracy (%)')
    for ax in axes:
        ax.set_xlabel('Maximum depth (None = unrestricted)'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig,'depth_comparison')
    print('All figures and case-study logs generated.')


if __name__=='__main__':
    main()

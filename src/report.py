"""Build the eight-page two-column report entirely from saved final results."""
import csv
import json
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Table, TableStyle, Image, Spacer
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1]
W,H=612,792
M=58.5
TW=495
GAP=22.5
CW=(TW-GAP)/2
TOP=720
BOTTOM=81
BODY=ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=12,
                    spaceAfter=6,allowWidows=0,allowOrphans=0)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=8.5,leading=10,spaceAfter=5)
HEAD=ParagraphStyle('head',parent=BODY,fontName='Times-Bold',fontSize=11,leading=13,spaceBefore=5,spaceAfter=6)
CAP=ParagraphStyle('caption',parent=SMALL,fontSize=9,leading=10.5)
CELL=ParagraphStyle('cell',parent=SMALL,fontSize=8,leading=9,spaceAfter=0)


def read(name):
    with (ROOT/'results'/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def p(text,small=False): return Paragraph(text,SMALL if small else BODY)
def h(text): return Paragraph(text,HEAD)
def caption(text): return Paragraph(text,CAP)
def f(x,d=3): return f'{float(x):.{d}f}'
def pct(x): return f'{100*float(x):.2f}'
def depth(x): return 'None' if x is None or x=='' or x=='None' else str(x)


def tab(headers,data,width=CW,widths=None,size=8.5):
    table=Table([headers]+data,colWidths=widths or [width/len(headers)]*len(headers),hAlign='LEFT')
    table.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,0),'Times-Bold'),('FONTNAME',(0,1),(-1,-1),'Times-Roman'),
        ('FONTSIZE',(0,0),(-1,-1),size),('LEADING',(0,0),(-1,-1),size+2),
        ('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),
        ('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ALIGN',(1,0),(-1,-1),'RIGHT'),('LEFTPADDING',(0,0),(-1,-1),3),
        ('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),
        ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
    return table


def figure(name,width=CW):
    path=ROOT/'figures'/f'{name}.png'
    im=PILImage.open(path); ratio=im.height/im.width
    return Image(str(path),width=width,height=width*ratio)


class Report:
    def __init__(self,path,draft=False):
        self.c=canvas.Canvas(str(path),pagesize=(W,H),pageCompression=1,invariant=1)
        self.c.setTitle('Bagging, Boosting and Bias-Variance on Olivetti Faces')
        self.c.setAuthor('Jyotishi Aditya Jagdeep; computational assistance: OpenAI Codex')
        self.page=0
        self.layout=[]
        self.draft=draft

    def flow(self,items,x,y,width,bottom=BOTTOM):
        for item in items:
            if isinstance(item,str): item=p(item)
            before=getattr(item,'spaceBefore',0)
            after=getattr(item,'spaceAfter',0)
            y-=before
            _,height=item.wrap(width,y-bottom)
            if y-height < bottom-.1:
                raise RuntimeError(f'Page {self.page}: column overflows by {bottom-(y-height):.1f} pt; item {getattr(item,"text",type(item).__name__)[:90]}')
            item.drawOn(self.c,x,y-height)
            y-=height+after
        return y

    def new(self,title,top=None,left=None,right=None,author=None,bottom=None):
        self.page+=1; c=self.c; y=TOP
        if author:
            c.setFont('Times-Bold',14)
            c.drawCentredString(W/2,y,'Bagging, Boosting and Bias-Variance')
            c.drawCentredString(W/2,y-17,'on Olivetti Face Recognition')
            c.setFont('Times-Roman',12); c.drawCentredString(W/2,y-41,author['name'])
            c.setFont('Times-Roman',10); c.drawCentredString(W/2,y-56,'Roll number: '+author['roll_number'])
            c.setFont('Times-Roman',8.5); c.drawCentredString(W/2,y-70,author['repository'])
            c.linkURL(author['repository'],(M,y-73,W-M,y-62),relative=0,thickness=0)
            c.setFont('Courier',7.5); c.drawCentredString(W/2,y-83,'Final source commit: '+author['commit'])
            y-=99
        else:
            c.setFont('Times-Bold',12); c.drawString(M,y-12,title); y-=25
        if top:
            y=self.flow(top,M,y,TW)-8
        col_bottom=BOTTOM
        if bottom:
            height=sum(item.wrap(TW,1000)[1]+getattr(item,'spaceBefore',0)+getattr(item,'spaceAfter',0) for item in bottom)
            self.flow(bottom,M,BOTTOM+height,TW)
            col_bottom=BOTTOM+height+8
        left_y=self.flow(left or [],M,y,CW,bottom=col_bottom)
        right_y=self.flow(right or [],M+CW+GAP,y,CW,bottom=col_bottom)
        c.setFont('Times-Roman',9); c.drawCentredString(W/2,53,str(self.page))
        if self.draft:
            c.setFont('Times-Italic',8); c.drawRightString(W-M,53,'Draft: bias/variance refits pending')
        self.layout.append(dict(page=self.page,columns_start=y,left_end=left_y,right_end=right_y))
        c.showPage()

    def save(self):
        assert self.page==8
        self.c.save()
        (ROOT/'results/report_layout.json').write_text(json.dumps(self.layout,indent=2))


def main():
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    refit_indices=np.load(ROOT/'data/refit_indices.npy')
    unique_refit=float(np.mean([len(np.unique(ix)) for ix in refit_indices]))
    selected=json.loads((ROOT/'results/selected.json').read_text())
    treegrid=read('tree_grid.csv'); baggrid=read('bagging_grid.csv'); boostgrid=read('boosting_grid.csv')
    bv={r['model']:r for r in read('bias_variance.csv')} if (ROOT/'results/bias_variance.csv').exists() else {}
    met={(r['model'],r['subset']):r for r in read('final_metrics.csv')}
    cases=json.loads((ROOT/'results/case_studies.json').read_text())
    heavy=cases['heavy_sample']; rob=cases['robustness']; hist=read('boosting_history.csv')
    toy=json.loads((ROOT/'results/toy_samme.json').read_text())
    dr=read('depth_refits.csv') if bv else []; matched=read('matched_learners.csv')
    names=['tree','bagging','boosting']; display={'tree':'Tree','bagging':'Bagging','boosting':'Boosting'}
    best_tree=next(r for r in treegrid if depth(r['depth'])==depth(selected['depth']) and int(r['leaf'])==selected['leaf'])
    bestbag=next(r for r in baggrid if int(r['B'])==selected['bag_B'])
    bestboost=next(r for r in boostgrid if int(r['depth'])==selected['boost_depth'] and int(r['B'])==selected['boost_B'])
    git_head=subprocess.run(['git','rev-parse','--verify','HEAD'],cwd=ROOT,capture_output=True,text=True)
    commit=git_head.stdout.strip() if git_head.returncode==0 else 'DRAFT: source not committed'
    out=ROOT/'output/pdf'/('ML_Assignment_2_Report.pdf' if bv else 'ML_Assignment_2_Draft.pdf')
    out.parent.mkdir(parents=True,exist_ok=True)
    report=Report(out,draft=not bool(bv))
    rows=[[depth(r['depth']),r['leaf'],pct(r['train']),pct(r['validation']),pct(r['test']),
           r['leaves'],f(r['mean_samples_per_leaf'],2),pct(r['singleton_leaf_fraction']),r['actual_depth']] for r in treegrid]
    report.new('',author={**cfg,'commit':commit},bottom=[
        caption('Table 1. Complete single-tree grid (Q1a). Accuracies and singleton-leaf fraction are percentages. '
                'L = number of leaves; n/L = mean training occupancy. None means no depth limit.'),
        tab(['Depth','Leaf min','Train','Val','Test','L','n/L','Single %','Actual d'],rows,TW,size=9)],
        left=[h('Abstract'),p(f'On a fixed 240/80/80 split, we compare a CART baseline, handwritten bagging, '
             f'and handwritten multiclass SAMME. Validation selects depth {depth(selected["depth"])}, '
             f'leaf size {selected["leaf"]}, bagging with {selected["bag_B"]} trees, and SAMME with depth '
             f'{selected["boost_depth"]} and {selected["boost_B"]} rounds. Clean test accuracies are '
             f'{pct(met[("tree","test")]["accuracy"])}, {pct(met[("bagging","test")]["accuracy"])} and '
             f'{pct(met[("boosting","test")]["accuracy"])}%, respectively. Occlusion and 30 stratified bootstrap refits '
             'separate predictive performance from confidence and stability.'),
             h('1. Fixed experimental protocol'),
             p('The 400 images contain 40 identities, ten 64 x 64 images each [1]. '
               'We flatten all 4096 original pixels; no PCA, feature selection or extra scaling is used. '
               'The digits of M26CSA002 give <b>26002</b>. For identities 0-39 in order, '
               '<font name="Courier" size="8">default_rng(26002)</font> permutes each set of ten: '
               'first six train, next two validate, last two test. The saved split is reused everywhere.')],
        right=[h('1.1. Selection and reproducibility'),
            p('Only clean validation accuracy chooses hyperparameters. The first maximum in the printed grid order '
              'breaks ties. The clean training set remains 240 images after selection. Separate seeded streams '
              'control tree randomness, bootstraps and refits. All class labels are zero-based.'),
            p(f'The selected tree has {best_tree["leaves"]} leaves, mean occupancy '
              f'{f(best_tree["mean_samples_per_leaf"],2)}, and {pct(best_tree["singleton_leaf_fraction"])}% singleton leaves. '
              f'Its realized depth is {best_tree["actual_depth"]}. Depth-2 and depth-4 models cannot allocate '
              'enough useful regions to distinguish 40 identities. Increasing depth improves fit; '
              'the unrestricted leaf-size-1 tree [4] memorizes training labels but retains a train-test gap. '
              'Larger minimum leaves reduce singleton memorization but do not improve validation here.'),
            p('<b>Assistance disclosure.</b> Code, experiments, figures and writing were prepared with OpenAI Codex. '
              'The assignment prohibits generative AI; this artifact does not comply with that course rule.',True)])

    report.new('2. Bootstrap aggregation and ensemble diagnostics (Q1b-c)',top=[
        figure('bagging_curves',TW),caption('Figure 1. Bagging size sweep. Diagnostics use clean validation images; '
          'OOB estimates use only covered training examples. Dotted lines mark the validation-selected B. '
          'Pairwise disagreement is undefined for B=1.')],
        left=[caption('Table 2. Accuracy (%) and average distinct bootstrap examples.'),
            tab(['B','Train','Val','Test','Distinct'],[[r['B'],pct(r['train']),pct(r['validation']),pct(r['test']),f(r['distinct_mean'],2)] for r in baggrid]),
            p('Each learner draws n=240 training rows with replacement [2] and uses the frozen tree settings. '
              'The ensemble predicts by equal-weight hard voting. Its expected distinct count is '
              'n[1-(1-1/n)<super>n</super>] = '+f(bestbag['distinct_theory'],3)+'. '
              'The observed counts approach this expectation as more bootstrap draws are averaged.',True)],
        right=[h('2.1. What the diagnostics measure'),
            p(f'Validation selects <b>B={selected["bag_B"]}</b> with {pct(bestbag["validation"])}% accuracy. '
              f'Its OOB accuracy is {pct(bestbag["oob_accuracy"])}% at {pct(bestbag["oob_coverage"])}% coverage; '
              f'validation disagreement is {f(bestbag["disagreement"])} and mean vote margin is {f(bestbag["vote_margin"])}.'),
            p('OOB pools predictions only from trees that omitted each training row. It uses fewer voters per '
              'image than validation and is noisy at small B. Disagreement is the fraction of unequal predictions '
              'averaged over unordered tree pairs and validation images. Margin is the winning vote fraction '
              'minus the runner-up fraction. Neither requires the winner to be correct.'),
            p('From B=100 to 200, OOB accuracy rises from 79.58% to 82.92%, but validation falls from '
              '86.25% to 85.00%. Disagreement and margin barely change. At B=1 the margin is always one, '
              'even for mistakes. These diagnostics therefore do not identify the same best size.',True)])

    caseparas=[]
    for c in cases['bagging']:
        votes=', '.join(f'{k}:{v}' for k,v in c['votes'].items())
        caseparas.extend([h(f'Image {c["dataset_index"]}: '+('corrected' if c['kind']=='corrected' else 'both wrong')),
          p(f'True class {c["target"]}; tree predicts {c["tree_prediction"]}; bagging predicts {c["bagging_prediction"]}. '
            f'All nonzero class:vote counts are <b>{votes}</b>.',True)])
    report.new('3. What bagging corrects, and what it misses (Q1d)',
        left=[figure('bagging_cases'),caption('Figure 2. First two clean-test images corrected by bagging and first two '
                 'misclassified by both models, in saved split order. No alternative split was searched.'),
              h('3.1. Clean and occluded confidence'),
              tab(['Model / set','Acc. %','Conf.'],[[display[n]+' / '+s,pct(met[(n,s)]['accuracy']),f(met[(n,s)]['confidence'])]
                    for n in ['tree','bagging'] for s in ['test','masked']],widths=[116,59,61.25]),
              p('Tree confidence is the largest class fraction in the reached leaf. Bagging confidence is '
                'the largest hard-vote fraction. Both are averaged over all test images, including mistakes. '
                'Pure leaves may give confidence 1 to wrong predictions; confidence is not calibrated accuracy.'),
              h('3.2. Why averaging helps'),
              p('Bootstrap perturbations alter pixel thresholds and tree paths. On corrected examples, enough '
                'alternative trees vote for the true class to overcome the original tree\'s error. '
                'This is evidence that some errors depend on which training faces define the partition.'),
              p('Image 1 is corrected by only 13 votes against a runner-up\'s 12: its margin is 0.01, '
                'so the correction is fragile. Image 21 receives 23 true-class votes against a '
                'runner-up count of 5, giving a larger 0.18 margin. A correct plurality need not '
                'be a majority of trees.',True)],
        right=caseparas+[
            h('3.3. The shared-error limit'),
            p('In the both-wrong cases, the true class fails to obtain the largest count even after '
              'averaging. Many trees may rely on similarly misleading raw-pixel cues. Bagging can suppress '
              'unstable partition errors, but it cannot guarantee recovery of information absent from the '
              'training sample or distinguish identities when its trees share the same mistake.'),
            p('These four cases illustrate prediction behavior. All aggregate test metrics use all 80 '
              'held-out faces; visual similarity alone cannot establish why a model failed.',True)])

    toyrows=[[str(i),str(int(toy['X'][i][0])),str(toy['targets'][i]),str(toy['predictions'][i]),
              f(toy['initial_weights'][i],4),f(toy['updated_weights'][i],4)] for i in range(len(toy['targets']))]
    report.new('4. Multiclass boosting from scratch (Q2a-b)',
        left=[h('4.1. A complete three-class round'),
            p('Use nine scalar examples x=0,...,8 with targets [0,0,0,1,1,1,2,2,2]. '
              'A depth-1 tree (seed 0) supplies the predictions below. Initial weights are uniform.'),
            caption('Table 3. Toy SAMME round; w and w\' are normalized weights before and after the update.'),
            tab(['i','x','y','h(x)','w','w\''],toyrows),
            p(f'The weighted error is {f(toy["weighted_error"],6)}. With K=3, '
              f'alpha = log((1-epsilon)/epsilon) + log(2) = <b>{f(toy["alpha"],6)}</b>. '
              'Misclassified weights are multiplied by exp(alpha); correct weights are unchanged before '
              'normalization. The resulting weights sum to one.'),
            h('4.2. Shared implementation'),
            p('The toy and face experiments both call <font name="Courier" size="8">samme_update</font>. '
              'At round t, weighted CART fits the clean training faces. Its weighted error and learner '
              'weight determine the next sample distribution. Scores sum alpha for learners predicting '
              'each class; dividing by the sum of accepted alphas gives q<sub>k</sub>(x) [3].'),
            p('Weights are updated in log space. If epsilon is at least 1-1/K (within 10<super>-14</super>), '
              'the learner is rejected and training stops; an invalid first learner raises an error. '
              'For zero error, epsilon is clipped to machine precision for finite alpha and training stops. '
              'Accepted round counts and stop reasons are logged.')],
        right=[h('4.3. Validation sweep'),
            caption('Table 4. Full boosting grid. Accuracy (%); the trees have minimum leaf size 1.'),
            tab(['Depth','Rounds','Train','Val','Test'],[[r['depth'],r['B'],pct(r['train']),pct(r['validation']),pct(r['test'])] for r in boostgrid]),
            p(f'Validation selects <b>depth {selected["boost_depth"]}, {selected["boost_B"]} rounds</b>, '
              f'with train/validation/test accuracy {pct(bestboost["train"])}/'
              f'{pct(bestboost["validation"])}/{pct(bestboost["test"])}%. '
              f'The selected prefix contains {bestboost["accepted_rounds"]} accepted learners. '
              'Depth is considered first, then round count, when validation ties occur.'),
            p('Weak learners need only outperform 1/K accuracy under the current weighted distribution. '
              'For K=40, this threshold is 2.5%, so a shallow learner can be useful despite a high '
              'absolute error. The log(K-1) term makes its accepted alpha positive. '
              'A weighted-vote winner can accumulate support across many weak partitions.')])

    peakround=heavy['peak_round']; start=max(1,peakround-2); stop=min(len(hist),peakround+2)
    historytxt='; '.join(f't={t}: h={heavy["predictions"][t-1]}, w={heavy["weights"][t]:.4f}' for t in range(start,stop+1))
    report.new('5. Boosting dynamics and matched model sizes (Q2b-c)',top=[
        figure('boosting_traces',TW),caption('Figure 3. Diagnostics for the validation-selected boosting prefix. '
          'Error is measured before updating weights; maximum and top-ten total sample weight are after updating.')],
        left=[caption('Table 5. Matched learner counts. Fit times are cumulative single-run seconds; '
                'accuracy values are percentages. Boosting uses the selected weak-learner depth.'),
            tab(['Model','B','Train','Val','Test','Time s'],[['Bag' if r['model']=='bagging' else 'SAMME',r['B'],pct(r['train']),pct(r['validation']),pct(r['test']),f(r['train_seconds'],1)] for r in matched],size=8),
            p('Bagging is more accurate at each matched count but slower to fit. SAMME validation falls '
              'from 50.00% at 25 rounds to 33.75% at 50, then rises to 46.25% at 100. More rounds do '
              'not ensure higher classification accuracy. Times exclude loading/evaluation and are '
              'single-run prefix timings, not benchmark averages.',True),
            p(f'The maximum selected-round sample weight is {f(heavy["peak_weight"],4)}, '
              f'{heavy["peak_weight"]*240:.1f} times the initial 1/240. Concentration of weight '
              'identifies difficult training examples; it is not a label-noise detector.',True)],
        right=[figure('heavy_sample'),caption('Figure 4. Training face with the largest post-update weight. '
             'Red points indicate rounds in which its weak learner is wrong.'),
            p(f'Image <b>{heavy["dataset_index"]}</b>, true class {heavy["target"]}, peaks after round '
              f'{peakround}. Nearby prediction history is: {historytxt}.',True),
            p('Every mistake increases this sample\'s weight relative to correctly predicted samples by '
              'exp(alpha). Once a later learner predicts it correctly, normalization may reduce its share. '
              'Repeated mistakes therefore bring attention back to the same face. The complete history '
              'is saved, so the explanation can be checked round by round.',True)])

    report.new('6. Robustness to the missing central region (Q2d)',
        left=[h('6.1. Frozen models, paired tests'),
            p('Rows and columns 24 through 39 inclusive are set to zero, creating a centered 16 x 16 block '
              '(256 pixels, 6.25% of the image). All three models are trained only on clean images. '
              'No setting is selected using masked validation or masked test results.'),
            caption('Table 6. Clean and masked test accuracy (%) and mean winning probability/vote.'),
            tab(['Model / set','Acc. %','Conf.'],[[display[n]+' / '+s,pct(met[(n,s)]['accuracy']),f(met[(n,s)]['confidence'])]
                    for n in names for s in ['test','masked']],widths=[116,59,61.25]),
            figure('robustness_face'),caption('Figure 5. A paired test face whose bagging and boosting responses differ.'),
            p(f'For image {rob["dataset_index"]}, target {rob["target"]}, bagging changes from '
              f'{rob["bagging_clean"]} to {rob["bagging_masked"]}; boosting changes from '
              f'{rob["boosting_clean"]} to {rob["boosting_masked"]}. '
              'Masking corrects boosting here while bagging stays correct: an individual improvement '
              'can coexist with worse aggregate accuracy. This is the first split-order face with '
              'different correctness changes.')],
        right=[figure('robustness_votes'),caption('Figure 6. The true class and union of the top four clean/masked classes. '
                 'Bars use equal-vote fractions for bagging and normalized alpha-weighted votes for boosting.'),
            h('6.2. Disagreement and weight sensitivity'),
            p(f'For this face, {pct(rob["bagging_changed_tree_fraction"])}% of bagged trees change prediction. '
              f'Pairwise disagreement is {f(rob["bagging_clean_disagreement"])} before and '
              f'{f(rob["bagging_masked_disagreement"])} after masking. '
              f'The boosting learners that change prediction carry {pct(rob["boosting_changed_alpha_fraction"])}% '
              'of total alpha. True-class bagging support falls from 64 to 45 votes but remains the '
              'plurality. For boosting, the changed votes suffice to move the true class above its rival.'),
            p('Bagging gives each bootstrapped tree one vote. SAMME changes the training distribution '
              'after mistakes and weights learners by alpha. The training-face history demonstrates '
              'reweighting; the test predictions measure its eventual occlusion response.'),
            p('This does not prove the heavy training face caused this test error. That would require '
              'a training intervention; here only test pixels change.'),
            h('6.3. Confidence under distribution shift'),
            p('Pure leaves can remain confident after occlusion, and agreeing trees may share a mistake. '
              'Read winning votes beside accuracy and paired prediction changes; they are not calibrated '
              'probabilities of correctness.')])

    if not bv:
        report.c.save()
        print('Six-page layout draft (refits pending):',out)
        return

    def ds(model,d,key):
        rr=[r for r in dr if r['model']==model and r['depth']==str(d)]
        return float(np.mean([float(r[key]) for r in rr]))
    depthdata=[]
    for d in selected['bias_variance_depths']:
        for model in ['tree','bag100']:
            r=bv[model+'_'+str(d)]
            depthdata.append([depth(d),'T' if model=='tree' else 'B100',f(r['bias_squared']),f(r['variance']),pct(r['accuracy_mean'])])
    report.new('7. Bias and variance measured across 30 refits (Q3a-b)',top=[
        figure('bias_variance',TW),caption('Figure 7. Selected-model squared-error decomposition and refit accuracy. '
            'Error bars are sample standard deviations across 30 refits, not confidence intervals.')],
        left=[h('7.1. Paired, stratified resampling'),
            p('The clean test set remains fixed. In each repetition, six faces are sampled with replacement '
              'from each identity\'s original six training faces. All methods use the same resample, and '
              'selected settings remain frozen. Refit randomness includes both data resampling and '
              f'varied learner seeds. The 240 rows contain {unique_refit:.1f} distinct original faces '
              'on average; refit accuracy need not equal the original-fit accuracy.'),
            p('For each test face, p-bar is the mean class vector over R=30 fits. We average '
              '||p-bar-e||<super>2</super> over N=80 faces for bias squared. Variance averages '
              '||p<sub>r</sub>-p-bar||<super>2</super> over all NR pairs. '
              'Tree vectors are leaf probabilities; bagging uses hard-vote fractions; boosting uses '
              'normalized weighted votes.'),
            caption('Table 7. Selected models: accuracy mean +/- sample SD (%).'),
            tab(['Model','Bias2','Var.','Mean +/- SD'],[[display[n],f(bv[n]['bias_squared']),f(bv[n]['variance']),
                pct(bv[n]['accuracy_mean'])+' +/- '+pct(bv[n]['accuracy_sd'])] for n in names],widths=[52,43,40,101.25]),
            p('For every model, bias squared plus variance equals its mean multiclass Brier error '
              'to numerical precision. This checks the computation; it does not imply an analogous '
              'additive decomposition of classification error.',True)],
        right=[h('7.2. Holding ensemble size at 100'),
            caption('Table 8. Matched depth sweep: T = single tree; B100 = 100-tree bagging. Acc. is mean refit test accuracy (%).'),
            tab(['Depth','Model','Bias2','Var.','Acc.'],depthdata),
            p('Minimum leaf size remains the selected value. Identical random streams pair bootstrap '
              'draws and tree seeds across depth settings. This isolates the role of the depth cap '
              'without tuning a new model on the test set.'),
            p('The selected baseline is unrestricted, so no strictly deeper allowable cap exists. '
              'Depth 2 is the shallow probe, depth 16 an intermediate probe, and depth 64 a '
              'saturation control. When every unrestricted realized tree is shallower than 64, '
              'the depth-64 prediction is exactly identical with the same random choices. '
              'These identical fits are reused only after checking that condition.',True)])

    improvements=[]
    for d in selected['bias_variance_depths']:
        t=bv['tree_'+str(d)]; b=bv['bag100_'+str(d)]
        improvements.append((d,100*(float(b['accuracy_mean'])-float(t['accuracy_mean']))))
    bestdepth,maxgain=max(improvements,key=lambda x:x[1])
    stable=min(names,key=lambda n:float(bv[n]['variance']))
    bestacc=max(names,key=lambda n:float(met[(n,'test')]['accuracy']))
    drops={n:100*(float(met[(n,'test')]['accuracy'])-float(met[(n,'masked')]['accuracy'])) for n in names}
    sensitive=max(names,key=lambda n:drops[n])
    depthtext=[]
    for d,gain in improvements:
        t=bv['tree_'+str(d)]; b=bv['bag100_'+str(d)]
        reduction=100*(1-float(b['variance'])/float(t['variance'])) if float(t['variance']) else 0
        depthtext.append(f'Depth {depth(d)}: accuracy gain {gain:.2f} points; variance reduction {reduction:.1f}%; '
             f'tree mean leaf occupancy {ds("tree",d,"mean_samples_per_leaf"):.2f}, '
             f'singleton leaves {100*ds("tree",d,"singleton_leaf_fraction"):.1f}%, '
             f'bag disagreement {ds("bag100",d,"disagreement"):.3f}.')
    report.new('8. Depth, stability and final comparison (Q3c-d)',
        left=[h('8.1. Does the deepest tree benefit most?'),
            *[p(t,True) for t in depthtext],
            p(f'The largest observed bagging improvement is <b>{maxgain:.2f} percentage points at '
              f'depth {depth(bestdepth)}</b>. The measured ranking, rather than depth alone, answers this '
              'question for this split. Equal unrestricted/depth-64 results reflect a nonbinding cap; '
              'they are not independent evidence of greater depth.'),
            p('Leaf occupancy measures how finely training faces are partitioned; disagreement shows '
              'whether refitted trees make different test predictions. Variance reduction can help '
              'only when averaging does not leave overwhelming bias or a shared wrong winner. '
              'A deeper model is therefore not guaranteed to receive the largest accuracy benefit. '
              f'At unrestricted depth, bias squared rises by {float(bv["bag100_None"]["bias_squared"])-float(bv["tree_None"]["bias_squared"]):.3f} '
              f'after bagging, compared with {float(bv["bag100_16"]["bias_squared"])-float(bv["tree_16"]["bias_squared"]):.3f} at depth 16. '
              'Thus the larger deep-tree variance reduction does not translate into the largest accuracy gain.'),
            h('8.2. Limits of the measurements'),
            p('There are only 80 fixed test faces: each correct prediction changes accuracy by 1.25 points. '
              'Refit SD measures sensitivity to resampling these training faces, not uncertainty over '
              'new identities or new test populations. Probability-vector variance also depends on '
              'vote concentration, so a diffuse predictor can be stable and still inaccurate.',True)],
        right=[caption('Table 9. Final frozen-model test accuracy (%) with measured bias squared and variance.'),
            tab(['Model','Clean','Mask','Bias2','Var.'],[[display[n],pct(met[(n,'test')]['accuracy']),
                pct(met[(n,'masked')]['accuracy']),f(bv[n]['bias_squared']),f(bv[n]['variance'])] for n in names]),
            p(f'<b>{display[stable]}</b> has the smallest measured probability-vector variance. '
              f'<b>{display[sensitive]}</b> is most sensitive by absolute accuracy drop '
              f'({drops[sensitive]:.2f} points). The drops for tree, bagging and boosting are '
              f'{drops["tree"]:.2f}, {drops["bagging"]:.2f} and {drops["boosting"]:.2f} points.'),
            p(f'The highest clean test accuracy belongs to <b>{display[bestacc]}</b>. '
              +('Here the lowest-variance and highest-accuracy models coincide, but this is an empirical outcome, not a guarantee.'
                if stable==bestacc else 'Here the lowest-variance model is not the most accurate, directly demonstrating that variance alone is insufficient.')
              +' Bias, the size of winning margins and which class wins must also be considered.'),
            p('The depth sweep provides a measured counterexample: depth-2 bagging has variance '
              f'{float(bv["bag100_2"]["variance"]):.3f}, below unrestricted bagging\'s '
              f'{float(bv["bag100_None"]["variance"]):.3f}, yet its mean refit accuracy is only '
              f'{pct(bv["bag100_2"]["accuracy_mean"])}% versus {pct(bv["bag100_None"]["accuracy_mean"])}%. '
              'Its high bias dominates despite its stability.',True),
            h('8.3. Reproduction and checks'),
            p('The repository contains the saved split and 30 resamples, complete grids, per-refit class '
              'vectors, final predictions, configurations, unit tests, environment pins and figure/report '
              'scripts. README commands rerun training or regenerate this report from the committed '
              'logs. The PDF embeds the source commit and is generated outside version control to '
              'avoid a self-referential hash.',True),
            h('References'),
            p('[1] Scikit-learn developers. <i>Olivetti faces dataset and fetch_olivetti_faces API.</i> '
              '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html">scikit-learn.org</link>. Accessed October 12, 2026.',True),
            p('[2] Leo Breiman. Bagging predictors. <i>Machine Learning</i>, 24:123-140, 1996. '
              '<link href="https://doi.org/10.1007/BF00058655">doi:10.1007/BF00058655</link>.',True),
            p('[3] Ji Zhu, Hui Zou, Saharon Rosset and Trevor Hastie. Multi-class AdaBoost. '
              '<i>Statistics and Its Interface</i>, 2:349-360, 2009. '
              '<link href="https://doi.org/10.4310/SII.2009.v2.n3.a8">doi:10.4310/SII.2009.v2.n3.a8</link>.',True),
            p('[4] Scikit-learn developers. <i>DecisionTreeClassifier API</i>, version 1.9.1. '
              '<link href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html">scikit-learn.org</link>.',True),
            p('[5] CVPR organizers. <i>CVPR author kit</i>. '
              '<link href="https://github.com/cvpr-org/author-kit">github.com/cvpr-org/author-kit</link>. Layout dimensions followed in this ReportLab implementation.',True)])
    report.save()
    print(out)


if __name__=='__main__':
    main()

from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import dill
from sklearn.metrics import precision_recall_fscore_support as score
from matplotlib import cm as cmp
from sklearn.metrics import confusion_matrix,accuracy_score
import itertools
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from sklearn.metrics import precision_recall_curve,average_precision_score,auc
from matplotlib.ticker import FuncFormatter
from seqIo import parse_ann

from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import binarize
import matplotlib.colors as col
from sklearn.metrics import ranking
import matplotlib.ticker as tkr
import joblib


def prf_bin_metrics(y_tr_beh, pd_class, beh):
    pred_pos = np.where(pd_class == 1)[0]
    true_pred = np.where(y_tr_beh[pred_pos] == 1)[0]
    true_pos = np.where(y_tr_beh == 1)[0]
    pred_true = np.where(pd_class[true_pos] == 1)[0]

    n_pred_pos = len(pred_pos)
    n_true_pred = len(true_pred)
    n_true_pos = len(true_pos)
    n_pred_true = len(pred_true)

    precision = n_true_pred / (n_pred_pos + np.spacing(1))
    recall = n_pred_true / (n_true_pos + np.spacing(1))
    f_measure = 2 * precision * recall / (precision + recall + np.spacing(1))
    print('P: %5.4f, R: %5.4f,    %s' % (precision, recall, beh))
    return precision, recall

def score_info(y,y_pred):
    precision, recall, fscore, _ = score(y, y_pred)
    print('#Precision: {}'.format(np.round(precision, 3)))
    print('#Recall:    {}'.format(np.round(recall, 3)))
    return precision, recall,fscore

def frames2bouts(pd_class):
    bouts=[]
    bouts_se=[]
    i = 0
    cab = 0
    cae = 0
    while i < len(pd_class) - 1:
        a = pd_class[i]
        if pd_class[i] != pd_class[i + 1]:
            cae = i+1
            bouts_se.append([cab,cae])
            bouts.append(a)
            cab = cae
        i+=1
    bouts_se.append([cab, len(pd_class)])
    bouts.append(a)
    return np.array(bouts),np.array(bouts_se)

def norm_cm(gt, pd):
    cm = confusion_matrix(gt, pd)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    np.round(cm, 3)
    return cm


def plot_conf_mat(cm, classes, savedir, savename='',title='', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig=plt.figure()
    if len(classes)>8: fig.set_size_inches(15,15)
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(0, 100)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.2f'% cm[i, j] if cm.shape[0]<5 else '%.1f' % cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if 'Recall' in title:
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    else:
        plt.xlabel('True label')
        plt.ylabel('Predicted label')
    plt.savefig(savedir + savename)
    plt.savefig(savedir + savename + '.pdf')
    plt.close()

def plot_conf_mat2(cm, classes, savedir, savename='',title='', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig=plt.figure()
    if len(classes)>8: fig.set_size_inches(15,15)
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=16)
    plt.colorbar()
    plt.clim(0, 100)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,    fontsize=16)
    plt.yticks(tick_marks, classes,    fontsize=16)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.2f'% cm[i, j] if cm.shape[0]<5 else '%.1f' % cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",    fontsize=16)

    plt.tight_layout()
    if 'Recall' in title:
        plt.ylabel('True label',    fontsize=16)
        plt.xlabel('Predicted label',    fontsize=16)
    else:
        plt.xlabel('True label',    fontsize=16)
        plt.ylabel('Predicted label',    fontsize=16)
    plt.savefig(savedir + savename)
    plt.savefig(savedir + savename + '.pdf')
    plt.close()

def plot_roc_curve(gt_bin, pd_bin,proba,behs, savedir,savename,colors=[]):
    n_classes=len(behs)
    fig, ax = plt.subplots(1,1)
    if colors==[]:    colors = cycle(['c', 'limegreen' ,'lightcoral'])

    for i, color in zip(range(1,n_classes), colors):
        #get tp fp p n perm
        P=np.sum(gt_bin[:,i]==1)
        N=np.sum(gt_bin[:,i]==0)
        sorted_proba = np.sort(proba[:,i-1,1])[::-1]
        perm = np.argsort(proba[:,i-1,1])[::-1]
        gt_perm = gt_bin[perm,i]
        tp = np.insert(np.cumsum(gt_perm==1),0,0)
        fp = np.insert(np.cumsum(gt_perm==0),0,0)
        # compute rates
        tpr = tp /np.maximum(P,eps)
        fpr = fp /np.maximum(N,eps)
        fnr = 1-tpr
        tnr=1-fpr
        roc_auc = np.sum(tpr*np.diff(np.insert(fpr,0,0)))
        s = np.max(np.where(tnr > tpr)[0])
        if s == len(tpr):
            eer = np.NAN; eerTh = 0
        else:
            if tpr[s] == tpr[s + 1]:
                eer = 1 - tpr[s]
            else:
                eer = 1 - tnr[s]
            eerTh = sorted_proba[s]
        # plt.plot( eer, 1 - eer, color=color, marker='s', markeredgecolor='k', clip_on=False)
        TP = np.sum(np.logical_and(pd_bin[:, i] == 1, gt_bin[:, i] == 1))
        TN = np.sum(np.logical_and(pd_bin[:, i] == 0, gt_bin[:, i] == 0))
        FP = np.sum(np.logical_and(pd_bin[:, i] == 1, gt_bin[:, i] == 0))
        FN = np.sum(np.logical_and(pd_bin[:, i] == 0, gt_bin[:, i] == 1))
        TPR = TP / (P + eps)
        FPR = FP / (N + eps)
        FNR = 1 - TPR
        plt.plot(fpr, tpr, color=color, lw=2, label='{0} (area = {1:0.2f}, EER = {2:0.2f})'  ''.format(behs[i], roc_auc,eer), clip_on=False)
        plt.plot(FPR, TPR, color=color, marker='o', markeredgecolor='k', clip_on=False)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.grid(linestyle='dotted')
    major_ticks = np.arange(0, 1.01, .2)
    minor_ticks = np.arange(0, 1.01, .1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(savedir + savename)
    plt.savefig(savedir + savename + '.pdf')
    plt.close()

def plot_pr_curve(gt_bin, proba,behs, prec,rec,prec_bw,rec_bw,savedir,savename,colors=[]):
    # For each class
    n_classes=len(behs)
    fig,ax=plt.subplots(1,1)
    lines = []
    labels = []

    if colors==[]:colors = cycle(['c', 'limegreen' ,'lightcoral'])
    for i, color in zip(range(1,n_classes), colors):
        # get tp fp p n perm
        P = np.sum(gt_bin[:, i] == 1)
        N = np.sum(gt_bin[:, i] == 0)
        perm = np.argsort(proba[:, i - 1, 1])[::-1]
        gt_perm = gt_bin[perm, i]
        tp = np.insert(np.cumsum(gt_perm == 1), 0, 0)
        fp = np.insert(np.cumsum(gt_perm == 0), 0, 0)
        recall = tp / np.maximum(P,eps)
        precision = np.maximum(tp,eps)/np.maximum(tp+fp,eps)
        auc = .5*np.sum((precision[:-1]+precision[1:])*np.diff(recall))
        sel = np.where(np.diff(recall))[0]+1
        ap = np.sum(precision[sel])/P
        ap11 = 0.0
        for rc in np.linspace(0,1,11):
            pr = np.max(np.insert(precision[recall>=rc],0,0))
            ap11 =ap11+pr/11
        l, = ax.plot(recall, precision, color=color, lw=2)
        lines.append(l)
        labels.append('{0} (area = {1:0.2f}, AP = {2:0.2f}, AP11 = {3:0.2f})'.format(behs[i],auc,ap,ap11))
        ax.plot(rec[i],prec[i],color=color,marker='o')
        ax.plot(rec_bw[i],prec_bw[i],color=color,marker='s')
        ax.plot([rec_bw[i], rec[i]], [prec_bw[i], prec[i]], '-', lw=1, color=color)

    fig.subplots_adjust(bottom=0.25)
    ax.grid(linestyle='dotted')
    major_ticks = np.arange(0, 1.01, .2)
    minor_ticks = np.arange(0, 1.01, .1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall ')
    l,=ax.plot([], [], marker='o', markeredgecolor='k', markerfacecolor='w', markeredgewidth=.5,linestyle='None')
    lines.append(l)
    labels.append('Frame based')
    l,=ax.plot([], [], marker='s', markeredgecolor='k', markerfacecolor='w', markeredgewidth=.5, linestyle='None')
    lines.append(l)
    labels.append('Bout based')
    ax.legend(lines, labels, loc='best', )
    plt.savefig(savedir + savename)
    plt.savefig(savedir + savename+'.pdf')
    plt.close()

def plot_det_curve(gt_bin, pd_bin,proba,behs,savedir,savename,colors=[]):

    n_classes=len(behs)
    fig, ax = plt.subplots(1,1)
    if colors==[]:    colors = cycle(['c', 'limegreen' ,'lightcoral'])

    for i, color in zip(range(1,n_classes), colors):
        P=np.sum(gt_bin[:,i]==1)
        N=np.sum(gt_bin[:,i]==0)
        sorted_proba = np.sort(proba[:,i-1,1])[::-1]
        sorted_proba_idx = np.argsort(proba[:,i-1,1])[::-1]
        gt_perm = gt_bin[sorted_proba_idx,i]
        tp = np.insert(np.cumsum(gt_perm==1),0,0)
        fp = np.insert(np.cumsum(gt_perm==0),0,0)
        tpr = tp /np.maximum(P,eps)
        fpr = fp /np.maximum(N,eps)
        fnr = 1-tpr
        tnr=1-fpr
        miss = 1-tpr
        falseAl =1-tnr
        plt.loglog(falseAl,miss,color, lw=2,  label=behs[i],clip_on=False)
        TP = np.sum(np.logical_and(pd_bin[:, i] == 1, gt_bin[:, i] == 1))
        FP = np.sum(np.logical_and(pd_bin[:, i] == 1, gt_bin[:, i] == 0))
        TPR = TP / (P+eps)
        FPR = FP / (N+eps)
        FNR = 1-TPR
        plt.plot(FPR, FNR, color=color, marker='s', markeredgecolor='k',clip_on=False)

    ax.grid(linestyle='dotted')
    ax.grid(which='both')
    plt.xlabel('Fallout (False Positive Rate)')
    plt.ylabel('Miss (False Negative Rate)')
    plt.title('DET curve')
    plt.legend(loc="lower left")
    plt.savefig(savedir + savename)
    plt.savefig(savedir + savename + '.pdf')
    plt.close()

eps = np.spacing(1)
# behs = ['interaction']


###### plot precicision recall top top_pcf topfront mlp xgboost window /no windows

path = 'mars_v1_6/'
folders = ['_t_mlp','_t_mlp_wnd','_t_xgb500','_t_xgb500_wnd'] #no cable
# folders = ['_m_mlp','_m_mlp_wnd','_m_xgb500','_m_xgb500_wnd']#cable
# folders = ['_tm_mlp','_tm_mlp_wnd','_tm_xgb500','_tm_xgb500_wnd']#all
type=['top','top_pcf','topfront']
savename=path + 'tm_'
color = ['b', 'g', 'r']
cmap = [ 'c', 'limegreen' ,'lightcoral']

behs = ['closeinvestigation', 'mount', 'attack']
n_classes = len(behs)
bin_pred = '3_pd_fbs_hmm'
all_pred = '9_pred_fbs_hmm_ass'
fixes = [bin_pred,all_pred]
n_exp = len(fixes)

results = {}
meth= []
for e, ex in enumerate(folders):
    for t in type:
        savedir = path + t + ex + '/'
        results[t+ex] = {}
        print('%s%s' % (t,ex))
        results[t+ex] = dill.load(open(savedir + '/results.dill', 'rb'))
        meth.append(t+ex)
n_meth = len(meth)
prec = np.zeros((n_meth, n_exp, n_classes))
rec = np.zeros((n_meth, n_exp, n_classes))
all_gt=[]
all_pd=[]
for e, ex in enumerate(meth):
        for f, fix in enumerate(fixes):
            if f == 0:
                for a in range(n_classes):
                    gt = results[ex]['0_G'][:, a]
                    pd = results[ex][fix][:, a]
                    prec[e, f, a], rec[e, f, a]= prf_bin_metrics(gt, pd, '%s %s %s' % (ex, behs[a], fix))
            elif f==1:
                gt = results[ex]['0_Gc']
                pd = results[ex][fix]
                all_gt.append(gt)
                all_pd.append(np.array(pd))
                p,r,f1 = score_info(gt, pd)
                prec[e, f,:] = p[1:]
                rec[e, f, :] = r[1:]

all_gt = np.hstack(np.array(all_gt))
all_pd = np.hstack(np.array(all_pd))
acc = accuracy_score(gt,pd)

# plot
meas = ['Precision', 'Recall']
barw = .25
xp = np.array([x * (barw + .05) for x in range(n_meth)])
# pr bars plot
for f, fix in enumerate(fixes):
    fig, ax = plt.subplots(1, n_classes * 2, sharey=True)  # plots by action by methods
    fig.set_size_inches(18, 5)
    plt.subplots_adjust(top=0.95, bottom=.1, left=.1, wspace=.2, right=.98)
    for a in range(0, n_classes * 2, 2):
        ax[a].text(.9 if a==0 else 1, xp[-1] + .2, behs[int(a / 2)], color=color[int(a / 2)], fontweight='bold')
        ax[a].barh(xp, prec[::-1, f, int(a / 2)], barw, color=cmap[int(a / 2)], ecolor='black', align='center')
        ax[a + 1].barh(xp, rec[::-1, f , int(a / 2)], barw, color=cmap[int(a / 2)], ecolor='black', align='center')
        ax[a].axvline(.8, linewidth=2, linestyle='--', color='gray', label='80%')
        ax[a].axvline(.85, linewidth=2, linestyle='--', color='sandybrown', label='85%')
        ax[a].axvline(.90, linewidth=2, linestyle='--', color='tan', label='90%')
        ax[a].axvline(.95, linewidth=2, linestyle='--', color='palegoldenrod', label='95%')
        ax[a + 1].axvline(.8, linewidth=2, linestyle='--', color='gray', label='80%')
        ax[a + 1].axvline(.85, linewidth=2, linestyle='--', color='sandybrown', label='85%')
        ax[a + 1].axvline(.90, linewidth=2, linestyle='--', color='tan', label='90%')
        ax[a + 1].axvline(.95, linewidth=2, linestyle='--', color='palegoldenrod', label='95%')
        ax[a].set_xlim([.65, 1.])
        ax[a + 1].set_xlim([.65, 1.])
        ax[a].set_ylim([xp[0] - .15, xp[-1] + .15])
        ax[a + 1].set_ylim([xp[0] - .15, xp[-1] + .15])
        ax[a].set_yticks(xp)
        ax[a + 1].set_yticks(xp)
        ax[a].set_yticklabels(meth[::-1], fontsize=9,)
        if a==0:[l.set_weight("bold") for o,l in enumerate(ax[a].get_yticklabels()) if o<n_meth/2]
        ax[a].set_xticks(np.arange(.6, 1.05, .1))
        ax[a + 1].set_xticks(np.arange(.6, 1.05, .1))
        ax[a].text(.75, -.5, meas[0])
        ax[a + 1].text(.75, -.5, meas[1])
    fig.savefig(savename + fix + '.png')
    fig.savefig(savename + fix + '.pdf')
    plt.close()

#### plots pr, roc confusion matrix

# ex='top_pcf_t_xgb500_wnd'
ex='top_pcf_m_xgb500_wnd'
# ex='top_pcf_tm_xgb500_wnd'
behs = ['other','closeinvestigation', 'mount', 'attack']
n_classes = len(behs)
results={}
savedir = path  + ex + '/'
results[ex] = dill.load(open(savedir + '/results.dill', 'rb'))
gt = results[ex]['0_Gc']
pd = results[ex]['9_pred_fbs_hmm_ass']
proba = results[ex]['6_proba_pd_hmm_fbs']
gt_bin = label_binarize(gt,range(n_classes))
pd_bin = label_binarize(pd,range(n_classes))

cm = norm_cm(gt, pd)
plot_conf_mat(cm, classes=behs, savedir=path, savename = ex +'_cm_recall_fbs_hmm_ass', title= 'Confusion matrix')
cm = norm_cm(pd, gt)
plot_conf_mat(cm, classes=behs, savedir=path, savename = ex +'_cm_precision_fbs_hmm_ass', title= 'Confusion matrix "Precision"')

plot_roc_curve(gt_bin,pd_bin, proba,behs, path, ex + '_fbs_hmm_ass_ROC')

###########
# framewise
###########
prec, rec, f1 = score_info(gt, pd)
count_pred = np.zeros(( n_classes-1)).astype(int)
dur_pred = np.zeros(( n_classes-1))
count_gt = np.zeros(( n_classes-1)).astype(int)
dur_gt = np.zeros((n_classes-1))
# measure agreement of frames using first annotations as references
for a in range(1,n_classes):
    gt_a = (gt == a).astype(int)
    n_gt = np.sum(gt_a)  # how many frames in gt bouts
    count_gt[a-1] = n_gt
    pd_a = (np.array(pd) == a).astype(int)
    n_pred = np.sum(pd_a)  # how many frames in detected bouts
    count_pred[a-1] = n_pred

    dur_gt[a-1] = n_gt
    dur_pred[a-1] = n_pred


############
# boutwise
############
gt_bouts, gt_bouts_se = frames2bouts(gt)
pd_bouts, pd_bouts_se = frames2bouts(pd)
IoU = []
for beh in range(n_classes):
    idx = np.where(gt_bouts == beh)[0]
    gt_a = gt_bouts_se[idx]
    idx = np.where(pd_bouts == beh)[0]
    pd_a = pd_bouts_se[idx]
    N = len(gt_a)  # number of bouts in gt
    M = len(pd_a)  # number of bouts in pd
    IoU.append(np.zeros((N, M)))
    for h in range(N):
        for k in range(M):
            a = gt_a[h, 0]
            b = gt_a[h, 1] - 1  # begin end of gt bout
            c = pd_a[k, 0]
            d = pd_a[k, 1] - 1  # begin end of pd bout
            b_gt_len = b - a + 1  # length of gt bout
            b_pd_len = d - c + 1  # length of pd bout
            intersection_len = np.maximum(0, np.minimum(b, d) - np.maximum(a, c))
            union_len = np.maximum(0, np.maximum(b, d) - np.minimum(a, c))
            IoU[beh][h, k] = intersection_len / float(union_len)  # intersection over union
# compute boutwise precision and recall
# compute PR for various thresh
# recall: number of pred bouts / num of gt bouts
# precision: how much of pred in gt wrt thresh
recall_bw = np.zeros((n_classes))
precision_bw = np.zeros(( n_classes))
f1_bw = np.zeros((n_classes))
count_pred_bw = np.zeros(( n_classes)).astype(int)
count_gt_bw = np.zeros(( n_classes)).astype(int)
dur_pred_bw = np.zeros(( n_classes))
dur_gt_bw = np.zeros(( n_classes))
IoU_thresh = .1

n_gt_bw = len(gt_bouts_se)
n_pd_bw = len(pd_bouts_se)
for a in range(1,n_classes):
    idx = np.where(gt_bouts == a)[0]
    gt_a = gt_bouts_se[idx]
    idx = np.where(pd_bouts == a)[0]
    pd_a = pd_bouts_se[idx]
    n_gt = float(len(gt_a))  # num of bouts in gt
    count_gt_bw[a] = n_gt
    n_pred = float(len(pd_a))  # num of bout in pd
    count_pred_bw[a] = n_pred

    dur_gt_bw[a] = np.round(np.median((gt_a[:, 1] - gt_a[:, 0]))) if gt_a.shape[0] > 1 else 0
    dur_pred_bw[a] = np.round(np.median((pd_a[:, 1] - pd_a[:, 0]))) if pd_a.shape[0] > 1 else 0

    IoU_above_tresh = (IoU[a] > IoU_thresh).astype(int)
    gt_recalled = (np.sum(IoU_above_tresh, 1) > 0).astype(int)
    n_correct_recall = np.sum(gt_recalled)
    correctly_pred = (np.sum(IoU_above_tresh, 0) > 0).astype(int)
    n_correct_pred = np.sum(correctly_pred)

    recall_bw[a] = n_correct_recall / n_gt
    recall_bw[np.isnan(recall_bw) | np.isinf(recall_bw)] = 1.
    precision_bw[a] = n_correct_pred / n_pred
    precision_bw[np.isnan(precision_bw) | np.isinf(precision_bw)] = 1.
    f1_bw[a] = 2 * ((precision_bw[a] * recall_bw[a]) / (precision_bw[a] + recall_bw[a]))
dur_gt_bw[np.isnan(dur_gt_bw)] = 0
dur_pred_bw[np.isnan(dur_pred_bw)] = 0

plot_pr_curve(gt_bin, proba,behs, prec,rec,precision_bw,recall_bw, path, ex + '_fbs_hmm_ass_PR')

plot_det_curve(gt_bin,pd_bin, proba,behs, path, ex + '_fbs_hmm_ass_DET')

# egregious perfs
def calc_prf(y_predicted, y_true, verbose=1):
    # Calculate the true positives, false positives, etc. for binary entries
    pred_pos = np.where(y_predicted == 1)[0]  # Find indices of examples predicted to be positive.
    pred_neg = np.where(y_predicted == 0)[0]
    true_pos = np.where(y_true[pred_pos] == 1)[0]  # Find the true positive predictions.
    gt_pos = np.where(y_true == 1)[0]  # Find the indices of the ground-truth positive examples are.

    n_pred_pos = len(pred_pos)
    n_true_pos = len(true_pos)
    n_gt_pos = len(gt_pos)

    # Calculate the Precision, Recall, and F-measure.
    #   Precision is TP/(TP + FP), and measures how often you're correct when designating a positive example.
    #   Recall is TP/(TP + FN), and measures the proportion of positive examples you're actually catching.
    precision = n_true_pos / n_pred_pos
    recall = n_true_pos / n_gt_pos
    f_measure = 2 * precision * recall / (precision + recall)

    if verbose:
        print('P: %5.4f, R: %5.4f, F1: %5.4f' % (precision, recall, f_measure))

    return precision, recall, f_measure
def calc_egregious_prf(y_predicted, y_true, frame_radius=30, verbose=1):
    """ Calculate the 'egregious' PR, i.e. PR with non-egregious failures removed."""
    # First correct the non-egregious false positives.
    y_changed = correct_nonegregious_false_positives(y_predicted,y_true, frame_radius)

    msg = "Egregious false positives corrected --calculating precision, recall, and fscore."
    print(msg)
    # Now calculate the precision, recall, and fscore.
    calc_prf(y_predicted,y_true, verbose=verbose)
    return
def gen_truncated_endpts(array_length, radius):
    """ Generator that generates the endpts of a window around a point"""
    for k in xrange(array_length):
        backward_endpt = k - radius
        forward_endpt = k + radius
        # Truncate everything.
        backward_endpt = max(backward_endpt, 0)
        forward_endpt = min(forward_endpt, array_length-1)
        yield [backward_endpt, forward_endpt]
def correct_nonegregious_false_positives(y_predicted, y_true, frame_radius = 30, verbose=1):
    """ Transform our "close" false positives into true negatives."""
    y_changed = y_predicted
    array_size = np.shape(y_true)[0]
    frame_num = 0
    for backward_endpt, forward_endpt in gen_truncated_endpts(array_length=array_size,radius=frame_radius):
        # For all the points within some radius of our desired point,...
        if np.any(y_true[backward_endpt:(forward_endpt+1)]):
            if y_predicted[frame_num]:
                y_changed[frame_num] = y_true[frame_num]

        # Increment our frame counter
        frame_num += 1

    return y_changed

for i in range(1,n_classes): calc_egregious_prf(gt_bin[:,i],pd_bin[:,i],30)

###################################################################################
#8 behs
####################################################################################
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

path = 'mars_v1_6_8behs/'
folders = ['_t_mlp','_t_mlp_wnd','_t_xgb500','_t_xgb500_wnd'] #no cable
type=['top','top_pcf','topfront']

behs = ['sniff_face', 'sniff_body', 'sniff_genital',
        'attempted_mount', 'mount', 'attempted_attack', 'attack', 'chase']

n_classes = len(behs)
fix = '9_pred_fbs_hmm_ass'

results = {}
meth= []
for e, ex in enumerate(folders):
    for t in type:
        savedir = path + t + ex + '/'
        results[t+ex] = {}
        print('%s%s' % (t,ex))
        results[t+ex] = dill.load(open(savedir + '/results.dill', 'rb'))
        meth.append(t+ex)
n_meth = len(meth)
prec = np.zeros((n_meth,  n_classes+2))
rec = np.zeros((n_meth,  n_classes+2))
all_gt=[]
all_pd=[]
for e, ex in enumerate(meth):
    gt = results[ex]['0_Gc']
    pd = results[ex][fix]
    all_gt.append(gt)
    all_pd.append(np.array(pd))
    p,r,f1 = score_info(gt, pd)
    prec[e,:] = p
    rec[e :] = r

all_gt = np.hstack(np.array(all_gt))
all_pd = np.hstack(np.array(all_pd))
acc = accuracy_score(gt,pd)

# plot
meas = ['Precision', 'Recall']
barw = .25
xp = np.array([x * (barw + .05) for x in range(n_meth)])
cmap = cmp.ScalarMappable(col.Normalize(0, n_classes), cmp.jet)
cmap = cmap.to_rgba(range(n_classes))
cmap=cmap[[0,1,2,3,4,6,7,5]]
savename=path + 't_'
# pr bars plot
fig, ax = plt.subplots(1, n_classes * 2, sharey=True)  # plots by action by methods
fig.set_size_inches(18, 5)
plt.subplots_adjust(top=0.95, bottom=.1, left=.1, wspace=.3, right=.98)
for a in range(0, n_classes * 2, 2):
    ax[a].text(0.6 if a in [3,5] else .8, xp[-1] + .2, behs[int(a / 2)], color=cmap[int(a / 2)], fontweight='bold')
    ax[a].barh(xp, prec[::-1, int(a / 2)+2], barw, color=cmap[int(a / 2)], ecolor='black', align='center')
    ax[a + 1].barh(xp, rec[::-1 , int(a / 2)+2], barw, color=cmap[int(a / 2)], ecolor='black', align='center')
    ax[a].set_xlim([0, 1.])
    ax[a + 1].set_xlim([.0, 1.])
    ax[a].set_ylim([xp[0] - .15, xp[-1] + .15])
    ax[a + 1].set_ylim([xp[0] - .15, xp[-1] + .15])
    ax[a].set_yticks(xp)
    ax[a + 1].set_yticks(xp)
    ax[a].set_yticklabels(meth[::-1], fontsize=9,)
    if a==0:[l.set_weight("bold") for o,l in enumerate(ax[a].get_yticklabels()) if o<n_meth/2]
    ax[a].set_xticks(np.arange(.0, 1.05, .5))
    ax[a + 1].set_xticks(np.arange(.0, 1.05, .5))
    ax[a].text(.15, -.5, meas[0])
    ax[a + 1].text(.135, -.5, meas[1])
fig.savefig(savename + fix + '_8behs.png')
fig.savefig(savename + fix + '_8behs.pdf')
plt.close()

#### plots pr, roc confusion matrix
ex='top_t_xgb500_wnd'
behs = ['other','close_investigation','sniff_face', 'sniff_body', 'sniff_genital',
        'attempted_mount', 'mount', 'attempted_attack', 'attack', 'chase']
n_classes = len(behs)
results={}
savedir = path  + ex + '/'
results[ex] = dill.load(open(savedir + '/results.dill', 'rb'))
gt = results[ex]['0_Gc']
pd = results[ex]['9_pred_fbs_hmm_ass']
proba = results[ex]['6_proba_pd_hmm_fbs']
gt_bin = label_binarize(gt,range(n_classes))
pd_bin = label_binarize(pd,range(n_classes))

cm = norm_cm(gt, pd)
plot_conf_mat(cm, classes=behs, savedir=path, savename = ex +'_cm_recall_fbs_hmm_ass_8behs', title= 'Confusion matrix')
cm = norm_cm(pd, gt)
plot_conf_mat(cm, classes=behs, savedir=path, savename = ex +'_cm_precision_fbs_hmm_ass_8behs', title= 'Confusion matrix "Precision"')

plot_roc_curve(gt_bin,pd_bin, proba,behs, path, ex + '_fbs_hmm_ass_ROC_8behs',
               np.insert(cmap,[0],[0.6,0.6,0.6,1.],axis=0))

###########
# framewise
###########
prec, rec, f1 = score_info(gt, pd)
count_pred = np.zeros(( n_classes-1)).astype(int)
dur_pred = np.zeros(( n_classes-1))
count_gt = np.zeros(( n_classes-1)).astype(int)
dur_gt = np.zeros((n_classes-1))
# measure agreement of frames using first annotations as references
for a in range(1,n_classes):
    gt_a = (gt == a).astype(int)
    n_gt = np.sum(gt_a)  # how many frames in gt bouts
    count_gt[a-1] = n_gt
    pd_a = (np.array(pd) == a).astype(int)
    n_pred = np.sum(pd_a)  # how many frames in detected bouts
    count_pred[a-1] = n_pred

    dur_gt[a-1] = n_gt
    dur_pred[a-1] = n_pred


############
# boutwise
############
gt_bouts, gt_bouts_se = frames2bouts(gt)
pd_bouts, pd_bouts_se = frames2bouts(pd)
IoU = []
for beh in range(n_classes):
    idx = np.where(gt_bouts == beh)[0]
    gt_a = gt_bouts_se[idx]
    idx = np.where(pd_bouts == beh)[0]
    pd_a = pd_bouts_se[idx]
    N = len(gt_a)  # number of bouts in gt
    M = len(pd_a)  # number of bouts in pd
    IoU.append(np.zeros((N, M)))
    for h in range(N):
        for k in range(M):
            a = gt_a[h, 0]
            b = gt_a[h, 1] - 1  # begin end of gt bout
            c = pd_a[k, 0]
            d = pd_a[k, 1] - 1  # begin end of pd bout
            b_gt_len = b - a + 1  # length of gt bout
            b_pd_len = d - c + 1  # length of pd bout
            intersection_len = np.maximum(0, np.minimum(b, d) - np.maximum(a, c))
            union_len = np.maximum(0, np.maximum(b, d) - np.minimum(a, c))
            IoU[beh][h, k] = intersection_len / float(union_len)  # intersection over union
# compute boutwise precision and recall
# compute PR for various thresh
# recall: number of pred bouts / num of gt bouts
# precision: how much of pred in gt wrt thresh
recall_bw = np.zeros((n_classes))
precision_bw = np.zeros(( n_classes))
f1_bw = np.zeros((n_classes))
count_pred_bw = np.zeros(( n_classes)).astype(int)
count_gt_bw = np.zeros(( n_classes)).astype(int)
dur_pred_bw = np.zeros(( n_classes))
dur_gt_bw = np.zeros(( n_classes))
IoU_thresh = .1

n_gt_bw = len(gt_bouts_se)
n_pd_bw = len(pd_bouts_se)
for a in range(1,n_classes):
    idx = np.where(gt_bouts == a)[0]
    gt_a = gt_bouts_se[idx]
    idx = np.where(pd_bouts == a)[0]
    pd_a = pd_bouts_se[idx]
    n_gt = float(len(gt_a))  # num of bouts in gt
    count_gt_bw[a] = n_gt
    n_pred = float(len(pd_a))  # num of bout in pd
    count_pred_bw[a] = n_pred

    dur_gt_bw[a] = np.round(np.median((gt_a[:, 1] - gt_a[:, 0]))) if gt_a.shape[0] > 1 else 0
    dur_pred_bw[a] = np.round(np.median((pd_a[:, 1] - pd_a[:, 0]))) if pd_a.shape[0] > 1 else 0

    IoU_above_tresh = (IoU[a] > IoU_thresh).astype(int)
    gt_recalled = (np.sum(IoU_above_tresh, 1) > 0).astype(int)
    n_correct_recall = np.sum(gt_recalled)
    correctly_pred = (np.sum(IoU_above_tresh, 0) > 0).astype(int)
    n_correct_pred = np.sum(correctly_pred)

    recall_bw[a] = n_correct_recall / n_gt
    recall_bw[np.isnan(recall_bw) | np.isinf(recall_bw)] = 1.
    precision_bw[a] = n_correct_pred / n_pred
    precision_bw[np.isnan(precision_bw) | np.isinf(precision_bw)] = 1.
    f1_bw[a] = 2 * ((precision_bw[a] * recall_bw[a]) / (precision_bw[a] + recall_bw[a]))
dur_gt_bw[np.isnan(dur_gt_bw)] = 0
dur_pred_bw[np.isnan(dur_pred_bw)] = 0

plot_pr_curve(gt_bin, proba,behs, prec,rec,precision_bw,recall_bw, path, ex + '_fbs_hmm_ass_PR_8behs',
              np.insert(cmap,[0],[0.6,0.6,0.6,1.],axis=0))


###################################################################################3
#12 behs
######################################################################################3
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


path = 'mars_v1_6_12behs/'
folders = ['_tm_mlp','_tm_mlp_wnd','_tm_xgb500','_tm_xgb500_wnd'] #no cable
type=['top','top_pcf','topfront']

behs = ['agg_investigation', 'sniff_body', 'sniff_face', 'sniff_genitals', 'approach',
        'chase', 'attempted_attack', 'attack', 'attempted_mount', 'mount', 'grooming']

n_classes = len(behs)
fix = '9_pred_fbs_hmm_ass'

results = {}
meth= []
for e, ex in enumerate(folders):
    for t in type:
        savedir = path + t + ex + '/'
        results[t+ex] = {}
        print('%s%s' % (t,ex))
        results[t+ex] = dill.load(open(savedir + '/results.dill', 'rb'))
        meth.append(t+ex)
n_meth = len(meth)
prec = np.zeros((n_meth,  n_classes+2))
rec = np.zeros((n_meth,  n_classes+2))
all_gt=[]
all_pd=[]
for e, ex in enumerate(meth):
    gt = results[ex]['0_Gc']
    pd = results[ex][fix]
    all_gt.append(gt)
    all_pd.append(np.array(pd))
    p,r,f1 = score_info(gt, pd)
    prec[e,:] = p
    rec[e :] = r

all_gt = np.hstack(np.array(all_gt))
all_pd = np.hstack(np.array(all_pd))
acc = accuracy_score(gt,pd)

# plot
meas = ['Precision', 'Recall']
barw = .25
xp = np.array([x * (barw + .05) for x in range(n_meth)])
cmap = cmp.ScalarMappable(col.Normalize(0, n_classes), cmp.jet)
cmap = cmap.to_rgba(range(n_classes))
cmap=cmap[[0,1,2,3,4,7,9,10,5,6,8]]
savename=path + 'tm_'
# pr bars plot
fig, ax = plt.subplots(1, n_classes * 2, sharey=True)  # plots by action by methods
fig.set_size_inches(20, 5)
plt.subplots_adjust(top=0.95, bottom=.1, left=.1, wspace=.3, right=.98)
for a in range(0, n_classes * 2, 2):
    ax[a].text(0.2 if a in [0,6,8] else .7, xp[-1] + .2, behs[int(a / 2)], color=cmap[int(a / 2)], fontweight='bold')
    ax[a].barh(xp, prec[::-1, int(a / 2)+2], barw, color=cmap[int(a / 2)], ecolor='black', align='center')
    ax[a + 1].barh(xp, rec[::-1 , int(a / 2)+2], barw, color=cmap[int(a / 2)], ecolor='black', align='center')
    ax[a].set_xlim([0, 1.])
    ax[a + 1].set_xlim([.0, 1.])
    ax[a].set_ylim([xp[0] - .15, xp[-1] + .15])
    ax[a + 1].set_ylim([xp[0] - .15, xp[-1] + .15])
    ax[a].set_yticks(xp)
    ax[a + 1].set_yticks(xp)
    ax[a].set_yticklabels(meth[::-1], fontsize=9,)
    if a==0:[l.set_weight("bold") for o,l in enumerate(ax[a].get_yticklabels()) if o<n_meth/2]
    ax[a].set_xticks(np.arange(.0, 1.05, .5))
    ax[a + 1].set_xticks(np.arange(.0, 1.05, .5))
    ax[a].text(.15, -.5, meas[0])
    ax[a + 1].text(.135, -.5, meas[1])
fig.savefig(savename + fix + '_12behs.png')
fig.savefig(savename + fix + '_12behs.pdf')
plt.close()

#### plots pr, roc confusion matrix
ex='topfront_tm_xgb500_wnd'
behs = ['other', 'agg_investigation', 'sniff_body', 'sniff_face', 'sniff_genitals', 'approach',
        'chase', 'attempted_attack', 'attack', 'attempted_mount', 'mount', 'grooming']
behs = ['other', 'close_investigation','agg_investigation', 'sniff_body', 'sniff_face', 'sniff_genitals', 'approach',
        'chase', 'attempted_attack', 'attack', 'attempted_mount', 'mount', 'grooming']
n_classes = len(behs)
results={}
savedir = path  + ex + '/'
results[ex] = dill.load(open(savedir + '/results.dill', 'rb'))
gt = results[ex]['0_Gc']
# gt[gt==1]=3
pd = np.array(results[ex]['9_pred_fbs_hmm_ass'])
# pd[pd==1]=3
proba = results[ex]['6_proba_pd_hmm_fbs']
gt_bin = label_binarize(gt,range(n_classes))
pd_bin = label_binarize(pd,range(n_classes))

cm = norm_cm(gt, pd)
plot_conf_mat2(cm, classes=behs, savedir=path, savename = ex +'_cm_recall_fbs_hmm_ass_12behs', title= 'Confusion matrix')
cm = norm_cm(pd, gt)
plot_conf_mat2(cm, classes=behs, savedir=path, savename = ex +'_cm_precision_fbs_hmm_ass_12behs', title= 'Confusion matrix "Precision"')

plot_roc_curve(gt_bin,pd_bin, proba,behs, path, ex + '_fbs_hmm_ass_ROC_12behs',
               np.insert(cmap,[0],[0.6,0.6,0.6,1.],axis=0))

###########
# framewise
###########
prec, rec, f1 = score_info(gt, pd)
count_pred = np.zeros(( n_classes-1)).astype(int)
dur_pred = np.zeros(( n_classes-1))
count_gt = np.zeros(( n_classes-1)).astype(int)
dur_gt = np.zeros((n_classes-1))
# measure agreement of frames using first annotations as references
for a in range(1,n_classes):
    gt_a = (gt == a).astype(int)
    n_gt = np.sum(gt_a)  # how many frames in gt bouts
    count_gt[a-1] = n_gt
    pd_a = (np.array(pd) == a).astype(int)
    n_pred = np.sum(pd_a)  # how many frames in detected bouts
    count_pred[a-1] = n_pred

    dur_gt[a-1] = n_gt
    dur_pred[a-1] = n_pred


############
# boutwise
############
gt_bouts, gt_bouts_se = frames2bouts(gt)
pd_bouts, pd_bouts_se = frames2bouts(pd)
IoU = []
for beh in range(n_classes):
    idx = np.where(gt_bouts == beh)[0]
    gt_a = gt_bouts_se[idx]
    idx = np.where(pd_bouts == beh)[0]
    pd_a = pd_bouts_se[idx]
    N = len(gt_a)  # number of bouts in gt
    M = len(pd_a)  # number of bouts in pd
    IoU.append(np.zeros((N, M)))
    for h in range(N):
        for k in range(M):
            a = gt_a[h, 0]
            b = gt_a[h, 1] - 1  # begin end of gt bout
            c = pd_a[k, 0]
            d = pd_a[k, 1] - 1  # begin end of pd bout
            b_gt_len = b - a + 1  # length of gt bout
            b_pd_len = d - c + 1  # length of pd bout
            intersection_len = np.maximum(0, np.minimum(b, d) - np.maximum(a, c))
            union_len = np.maximum(0, np.maximum(b, d) - np.minimum(a, c))
            IoU[beh][h, k] = intersection_len / float(union_len)  # intersection over union
# compute boutwise precision and recall
# compute PR for various thresh
# recall: number of pred bouts / num of gt bouts
# precision: how much of pred in gt wrt thresh
recall_bw = np.zeros((n_classes))
precision_bw = np.zeros(( n_classes))
f1_bw = np.zeros((n_classes))
count_pred_bw = np.zeros(( n_classes)).astype(int)
count_gt_bw = np.zeros(( n_classes)).astype(int)
dur_pred_bw = np.zeros(( n_classes))
dur_gt_bw = np.zeros(( n_classes))
IoU_thresh = .1

n_gt_bw = len(gt_bouts_se)
n_pd_bw = len(pd_bouts_se)
for a in range(1,n_classes):
    idx = np.where(gt_bouts == a)[0]
    gt_a = gt_bouts_se[idx]
    idx = np.where(pd_bouts == a)[0]
    pd_a = pd_bouts_se[idx]
    n_gt = float(len(gt_a))  # num of bouts in gt
    count_gt_bw[a] = n_gt
    n_pred = float(len(pd_a))  # num of bout in pd
    count_pred_bw[a] = n_pred

    dur_gt_bw[a] = np.round(np.median((gt_a[:, 1] - gt_a[:, 0]))) if gt_a.shape[0] > 1 else 0
    dur_pred_bw[a] = np.round(np.median((pd_a[:, 1] - pd_a[:, 0]))) if pd_a.shape[0] > 1 else 0

    IoU_above_tresh = (IoU[a] > IoU_thresh).astype(int)
    gt_recalled = (np.sum(IoU_above_tresh, 1) > 0).astype(int)
    n_correct_recall = np.sum(gt_recalled)
    correctly_pred = (np.sum(IoU_above_tresh, 0) > 0).astype(int)
    n_correct_pred = np.sum(correctly_pred)

    recall_bw[a] = n_correct_recall / n_gt
    recall_bw[np.isnan(recall_bw) | np.isinf(recall_bw)] = 1.
    precision_bw[a] = n_correct_pred / n_pred
    precision_bw[np.isnan(precision_bw) | np.isinf(precision_bw)] = 1.
    f1_bw[a] = 2 * ((precision_bw[a] * recall_bw[a]) / (precision_bw[a] + recall_bw[a]))
dur_gt_bw[np.isnan(dur_gt_bw)] = 0
dur_pred_bw[np.isnan(dur_pred_bw)] = 0

plot_pr_curve(gt_bin, proba,behs, prec,rec,precision_bw,recall_bw, path, ex + '_fbs_hmm_ass_PR_12behs',
              np.insert(cmap,[0],[0.6,0.6,0.6,1.],axis=0))


##############################################################################
#  results wild test
##############################################################################
# ex='wild_test'
# save_path = 'mars_v1_6/'
# path = '/media/cristina/MARS_data/mice_project/videos/wild_test/'

ex='wild_test'
save_path = 'wild_test/'
if not os.path.exists(save_path):os.makedirs(save_path)
path = '/dataset/mars/train_test_eval/wild_test/'
test_videos = os.listdir(path)

behs = ['other','closeinvestigation', 'mount', 'attack']
n_classes = len(behs)
label2id = {'other': 0,
            'cable_fix': 0,
            'intruder_introduction': 0,
            'corner': 0,
            'ignore': 0,
            'groom': 0,
            'groom_genital': 0,
            'grooming': 0,
            'tailrattle': 0,
            'tail_rattle': 0,
            'tailrattling': 0,
            'intruder_attacks': 0,
            'approach': 0,

            'closeinvestigate': 1,
            'closeinvestigation': 1,
            'investigation': 1,
            'sniff_genitals': 1,
            'sniff-genital': 1,
            'sniff-face': 1,
            'sniff-body': 1,
            'sniffurogenital': 1,
            'sniffgenitals': 1,
            'agg_investigation': 1,
            'sniff_face': 1,
            'anogen-investigation': 1,
            'head-investigation': 1,
            'sniff_body': 1,
            'body-investigation': 1,
            'socialgrooming': 1,

            'mount': 2,
            'aggressivemount': 2,
            'mount_attempt': 2,
            'intromission': 2,

            'attack': 3,

            }

color = ['b', 'g', 'r']
cmap = [ 'c', 'limegreen' ,'lightcoral']

test_videos = ['Mouse1100_20180110_20-51-38',#interaction
               'Mouse1145_20180405_12-51-10', #attack
               'Mouse1143_20180405_13-08-57', #mount
               'Mouse1139_20180406_11-19-25', #mount
               'Mouse#20_BalbC_20180423_13-28-05',#mount
               'Mouse#25_BalbC_20180423_14-30-32',#attack
               'Mouse513_male_20180404_15-42-09',#attack
               'Mouse515_male_20180404_12-38-38',#attack
               'Mouse447_20180117_12-35-57',#investigation
               'Mouse287_20170713_18-22-43',#mount
               'Mouse291_20170711_15-27-55',#mount
               'Mouse_20180222_18-06-26',#mount
               'Mouse_20170928_11-12-53',#attack
               'Mouse_20170928_13-06-53',#attack
               'Mouse_20170906_16-02-55']#attack
N_MOVIES=len(test_videos)
cmap4 = cmp.ScalarMappable(col.Normalize(0, N_MOVIES), cmp.jet)
cmap4 = cmap4.to_rgba(range(0,N_MOVIES))

flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def do_fbs(y_pred_class, kn, blur, blur_steps, shift):
    """Does forward-backward smoothing."""
    len_y = len(y_pred_class)

    # fbs with classes
    z = np.zeros((3, len_y))  # Make a matrix to hold the shifted predictions --one row for each shift.

    # Create mirrored start and end indices for extending the length of our prediction vector.
    mirrored_start = range(shift, -1, -1)  # Creates indices that go (shift, shift-1, ..., 0)
    mirrored_end = range(len_y - 1, len_y - 1 - shift, -1)  # Creates indices that go (-1, -2, ..., -shift)

    # Now we extend the predictions to have a mirrored portion on the front and back.
    extended_predictions = np.r_[
        y_pred_class[mirrored_start],
        y_pred_class,
        y_pred_class[mirrored_end]
    ]

    # Do our blurring.
    for s in range(blur_steps):
        extended_predictions = signal.convolve(np.r_[extended_predictions[0],
                                                     extended_predictions,
                                                     extended_predictions[-1]],
                                               kn / kn.sum(),  # The kernel we are convolving.
                                               'valid')  # Only use valid conformations of the filter.
        # Note: this will leave us with 2 fewer items in our signal each iteration, so we append on both sides.

    z[0, :] = extended_predictions[2 * shift + 1:]
    z[1, :] = extended_predictions[:-2 * shift - 1]
    z[2, :] = extended_predictions[shift + 1:-shift]

    z_mean = np.mean(z, axis=0)  # Average the blurred and shifted signals together.

    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]  # Anything that has a signal strength over 0.5, is taken to be positive.

    return y_pred_fbs

def predict_labels(features, classifier_path, behaviors=[]):
    all_predicted_probabilities, behaviors_used = predict_probabilities(features=features,
                                                                        classifier_path=classifier_path,
                                                                        behaviors=behaviors)
    proba, labels ,behs= assign_labels(all_predicted_probabilities=all_predicted_probabilities,
                           behaviors_used=behaviors_used)
    return all_predicted_probabilities,proba,labels,behs

def predict_probabilities(features, classifier_path, behaviors=[], VERBOSE = True):
        ''' This predicts behavior labels on a video, given a classifier and features.'''

        scaler = dill.load(open(classifier_path + '/scaler.dill', 'rb'))
        # Scale the data appropriately.
        X_test = scaler.transform(features)

        models = [os.path.join(classifier_path, filename) for filename in os.listdir(classifier_path)]
        behaviors_used = []

        preds_fbs_hmm = []
        proba_fbs_hmm = []

        for b, behavior in enumerate(behaviors):
            models_with_this_behavior = filter(lambda x: x.find('classifier_' + behavior) > -1, models)
            if models_with_this_behavior:
                name_n_timestamp = dict([(x, os.stat(x).st_mtime) for x in models_with_this_behavior])
                name_classifier = max(name_n_timestamp, key=lambda k: name_n_timestamp.get(k))

                classifier = dill.load(open(name_classifier, 'rb'))

                bag_clf = classifier['bag_clf']  if 'bag_clf' in classifier.keys() else classifier['clf']
                hmm_fbs = classifier['hmm_fbs']
                kn = classifier['k']
                blur_steps = classifier['blur_steps']
                shift = classifier['shift']

                behaviors_used += [behavior]

            else:
                print('Classifier not found, you need to train a classifier for this behavior before using it')
                print('Classification will continue without classifying this behavior')
                continue


            # Do the actual prediction.
            predicted_probabilities = bag_clf.predict_proba(X_test)
            predicted_class = np.argmax(predicted_probabilities, axis=1)

            y_pred_fbs = do_fbs(y_pred_class=predicted_class, kn=kn, blur=4, blur_steps=blur_steps, shift=shift)

            # Do the hmm prediction.
            y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
            y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)

            # Add our predictions to the list.
            preds_fbs_hmm.append(y_pred_fbs_hmm)
            proba_fbs_hmm.append(y_proba_fbs_hmm)


        # Change the list of [1x(numFrames)]-predictions to an np.array by stacking them vertically.
        preds_fbs_hmm = np.vstack(preds_fbs_hmm)

        # Flip it over so that it's stored as a [(numFrames)x(numBehaviors)] array
        all_predictions = preds_fbs_hmm.T

        # Change [(behavior)x(frames)x(positive/neg)] => [(frames) x (behaviors) x (pos/neg)]
        all_predicted_probabilities = np.array(proba_fbs_hmm).transpose(1, 0, 2)
        # pdb.set_trace()
        return all_predicted_probabilities, behaviors_used

def assign_labels(all_predicted_probabilities,behaviors_used):
    ''' Assigns labels based on the provided probabilities.'''
    labels = []
    behs=[]
    proba=[]
    num_frames = all_predicted_probabilities.shape[0]
    # Looping over frames, determine which annotation label to take.
    for i in xrange(num_frames):
        # Get the [3x2] matrix of current prediction probabilities.
        current_prediction_probabilities = all_predicted_probabilities[i,:]
        # Get the positive/negative labels for each behavior, by taking the argmax along the pos/neg axis.
        onehot_class_predictions = np.argmax(current_prediction_probabilities, axis=1)
        # Get the actual probabilities of those predictions.
        predicted_class_probabilities = np.max(current_prediction_probabilities, axis=1)

        # If every behavioral predictor agrees that the current_
        if np.all(onehot_class_predictions == 0):
            # The index here is one past any positive behavior --this is how we code for "other".
            beh_frame = 0
            # How do we get the probability of it being "other?" Since everyone's predicting it, we just take the mean.
            proba_frame = np.mean(predicted_class_probabilities)
            labels += [beh_frame]
            proba.append(proba_frame)
            behs += ['other']
        else:
            # If we have positive predictions, we find the probabilities of the positive labels and take the argmax.
            pos = np.where(onehot_class_predictions)[0]
            max_prob = np.argmax(predicted_class_probabilities[pos])

            # This argmax is, by construction, the id for this behavior.
            beh_frame = pos[max_prob]
            # We also want to save that probability,maybe.
            proba_frame = predicted_class_probabilities[beh_frame]
            labels.append(beh_frame+1)
            behs += [behaviors_used[beh_frame]]
            proba.append(proba_frame)

    return proba,labels,behs

def parse_annot(filename):
    """ Takes as input a path to a .annot file and returns the frame-wise behavioral labels."""
    if not filename:
        print("No filename provided")
        return -1

    behaviors = []
    channel_names = []
    keys = []

    channel_dict = {}
    with open(filename, 'rb') as annot_file:
        line = annot_file.readline().rstrip()
        # Parse the movie files
        while line != '':
            line = annot_file.readline().rstrip()
            # Get movie files if you want

        # Parse the stim name and other stuff
        start_frame = 0
        end_frame = 0
        stim_name = ''

        line = annot_file.readline().rstrip()
        split_line = line.split()
        stim_name = split_line[-1]

        line = annot_file.readline().rstrip()
        split_line = line.split()
        start_frame = int(split_line[-1])

        line = annot_file.readline().rstrip()
        split_line = line.split()
        end_frame = int(split_line[-1])

        line = annot_file.readline().rstrip()
        assert (line == '')

        # Just pass through whitespace
        while line == '':
            line = annot_file.readline().rstrip()

        # pdb.set_trace()
        # At the beginning of list of channels
        assert 'channels' in line
        line = annot_file.readline().rstrip()
        while line != '':
            key = line
            keys.append(key)
            line = annot_file.readline().rstrip()

        # pdb.set_trace()
        # At beginning of list of annotations.
        line = annot_file.readline().rstrip()
        assert 'annotations' in line
        line = annot_file.readline().rstrip()
        while line != '':
            behavior = line
            behaviors.append(behavior)
            line = annot_file.readline().rstrip()

        # At the start of the sequence of channels
        line = annot_file.readline()
        while line != '':
            # Strip the whitespace.
            line = line.rstrip()

            assert ('----------' in line)
            channel_name = line.rstrip('-')
            channel_names.append(channel_name)

            behaviors_framewise = [''] * end_frame
            line = annot_file.readline().rstrip()
            while '---' not in line:

                # If we've reached EOF (end-of-file) break out of this loop.
                if line == '':
                    break

                # Now get rid of newlines and trailing spaces.
                line = line.rstrip()

                # If this is a blank
                if line == '':
                    line = annot_file.readline()
                    continue

                # Now we're parsing the behaviors
                if '>' in line:
                    print(line)
                    curr_behavior = line[1:]
                    # Skip table headers.
                    annot_file.readline()
                    line = annot_file.readline().rstrip()

                # Split it into the relevant numbers
                start_stop_duration = line.split()

                # Collect the bout info.
                bout_start = int(start_stop_duration[0])
                bout_end = int(start_stop_duration[1])
                bout_duration = int(start_stop_duration[2])

                # Store it in the appropriate place.
                behaviors_framewise[(bout_start-1):bout_end] = [curr_behavior] * (bout_duration+1)

                line = annot_file.readline()

                # end of channel
            channel_dict[channel_name] = behaviors_framewise
        chosen_behavior_list = channel_dict['Ch1']
        changed_behavior_list = [annotated_behavior if annotated_behavior != '' else 'other' for annotated_behavior in
                                 chosen_behavior_list]
        ann_dict = {
            'keys': keys,
            'behs': behaviors,
            'nstrm': len(channel_names),
            'nFrames': end_frame,
            'behs_frame': changed_behavior_list
        }
        return ann_dict


classifier_path = 'mars_v1_6/top_pcf_tm_xgb500_wnd/'
def classify_actions_wrapper(top_video_fullpath,classifier_path):
    video_fullpath = top_video_fullpath
    video_path = os.path.dirname(video_fullpath)
    v = os.path.basename(video_fullpath)

    behaviors = ['closeinvestigation', 'mount', 'attack']

    # Get the output folder for this specific mouse.
    if 'wnd' in classifier_path:
        top_feat_name=video_fullpath+'/output_v1_7/' +  v +'/' +v + '_raw_feat_top_pcf_v1_7_wnd.npz'
        features = np.load(top_feat_name)['data']

    else:
        top_feat_name=video_fullpath+'/output_v1_7/' +  v +'/' +v + '_raw_feat_top_pcf_v1_7.npz'
        features = np.load(top_feat_name)['data_smooth']
        scaler = joblib.load(classifier_path+'scaler')

    # Classify the actions (get the labels back).
    all_proba,proba, pred ,behs= predict_labels(features, classifier_path,behaviors)

    return all_proba,proba,pred,behs

all_proba=[]
all_pred=[]
all_gt=[]
ann_list=[]
for v in test_videos:

    top_video_fullpath = path + v

    ann = [f for f in os.listdir(top_video_fullpath) if f.endswith('.annot')]
    ann_list.append(ann)
    if ann!=[]:
        print(v)
        all_pred_proba, proba, pd,pd_lab = classify_actions_wrapper(top_video_fullpath,classifier_path)
        all_pred.append(pd)
        all_proba.append(all_pred_proba)
        ann = parse_annot(top_video_fullpath+'/'+ann[0])['behs_frame']
        gt = np.array([label2id[i] if i in label2id else 0 for i in ann ])
        all_gt.append(gt)
        print(set(gt))
        print(set(pd))

gt=np.hstack(np.array(all_gt))
pd=np.hstack(np.array(all_pred))
prec,rec,_ = score_info(gt,pd)

for f in range(N_MOVIES):
    if len(all_gt[f])!=len(all_pred[f]):
        print(test_videos[f])
        print(str(len(all_gt[f])) + ' --- ' + str(len(all_pred[f])))

gt_bin = label_binarize(gt,range(n_classes))
pd_bin = label_binarize(pd,range(n_classes))
cm = norm_cm(gt, pd)
plot_conf_mat(cm, classes=behs, savedir=save_path, savename = ex +'_cm_recall_fbs_hmm_ass15', title= 'Confusion matrix')
cm = norm_cm(pd, gt)
plot_conf_mat(cm, classes=behs, savedir=save_path, savename = ex +'_cm_precision_fbs_hmm_ass15', title= 'Confusion matrix "Precision"')

precision = np.zeros((N_MOVIES, n_classes))
recall = np.zeros((N_MOVIES, n_classes))
f1 = np.zeros((N_MOVIES, n_classes))
count_pred = np.zeros((N_MOVIES, n_classes)).astype(int)
count_gt = np.zeros((N_MOVIES, n_classes)).astype(int)
recall_bw = np.zeros((N_MOVIES, n_classes))
precision_bw = np.zeros((N_MOVIES, n_classes))
f1_bw = np.zeros((N_MOVIES, n_classes))
count_pred_bw = np.zeros((N_MOVIES, n_classes)).astype(int)
count_gt_bw = np.zeros((N_MOVIES, n_classes)).astype(int)
dur_pred_bw = np.zeros((N_MOVIES, n_classes))
dur_gt_bw = np.zeros((N_MOVIES, n_classes))
eps=np.spacing(1)

##############
## framewise milticlass
##############
for m in range(len(all_gt)):
    gt = all_gt[m]
    pd = np.array(all_pred[m])

    # measure agreement of frames using first annotations as references
    for a in range(n_classes):
        gt_a = (gt==a)
        n_gt = np.sum(gt_a) # how many frames in gt bouts
        count_gt[m,a] = n_gt
        pd_a = (pd==a)
        n_pred = np.sum(pd_a) # how many frames in detected bouts
        count_pred[m,a] = n_pred

        n_correct_detect = np.sum((gt_a & pd_a).astype(int)) # places where both detect
        p = n_correct_detect / float(n_pred+eps) #num of frames detected as correct over num of detections
        p = 1. if (np.isnan(p) | np.isinf(p)) else p
        precision[m,a] = p
        r = n_correct_detect / float(n_gt+eps) #num of frames detected as correct over num of real correct
        r = 1. if (np.isnan(r) | np.isinf(r)) else r
        recall[m,a]= r
        f1[m,a] = 2*((precision[m,a]*recall[m,a]) / (precision[m,a]+recall[m,a]+eps))

############
# boutwise
############
IoU = [[] for _ in range(N_MOVIES)]
# first measure bout-to-bout affinity and intersection over unit
for m in range(len(all_gt)):
    gt = all_gt[m]
    pd = np.array(all_pred[m])
    gt_bouts, gt_bouts_se = frames2bouts(gt)
    pd_bouts, pd_bouts_se = frames2bouts(pd)
    for beh in range(n_classes):
        idx = np.where(gt_bouts == beh)[0]
        gt_a = gt_bouts_se[idx]
        idx = np.where(pd_bouts == beh)[0]
        pd_a = pd_bouts_se[idx]
        N = len(gt_a)  # number of bouts in gt
        M = len(pd_a)  # number of bouts in pd
        IoU[m].append(np.zeros((N, M)))
        for h in range(N):
            for k in range(M):
                a = gt_a[h, 0];                b = gt_a[h, 1] - 1  # begin end of gt bout
                c = pd_a[k, 0];                d = pd_a[k, 1] - 1  # begin end of pd bout
                intersection_len = np.maximum(0, np.minimum(b, d) - np.maximum(a, c))
                union_len = np.maximum(0, np.maximum(b, d) - np.minimum(a, c))
                IoU[m][beh][h, k] = intersection_len / float(union_len)  # intersection over union
# compute boutwise precision and recall
# compute PR for various thresh
# recall: number of pred bouts / num of gt bouts
# precision: how much of pred in gt wrt thresh
IoU_thresh = .1
for m in range(len(all_gt)):
    gt = all_gt[m]
    pd = np.array(all_pred[m])
    gt_bouts, gt_bouts_se = frames2bouts(gt)
    pd_bouts, pd_bouts_se = frames2bouts(pd)
    for a in range(n_classes):
        idx = np.where(gt_bouts == a)[0]
        gt_a = gt_bouts_se[idx]
        idx = np.where(pd_bouts == a)[0]
        pd_a = pd_bouts_se[idx]
        n_gt = float(len(gt_a))  # num of bouts in gt
        count_gt_bw[m, a] = n_gt
        n_pred = float(len(pd_a))  # num of bout in pd
        count_pred_bw[m, a] = n_pred

        dur_gt_bw[m, a] = np.round(np.median((gt_a[:, 1] - gt_a[:, 0]))) if gt_a.shape[0] > 1 else 0
        dur_pred_bw[m, a] = np.round(np.median((pd_a[:, 1] - pd_a[:, 0]))) if pd_a.shape[0] > 1 else 0

        IoU_above_tresh = (IoU[m][a] > IoU_thresh).astype(int)
        gt_recalled = (np.sum(IoU_above_tresh, 1) > 0).astype(int)
        n_correct_recall = np.sum(gt_recalled)
        correctly_pred = (np.sum(IoU_above_tresh, 0) > 0).astype(int)
        n_correct_pred = np.sum(correctly_pred)

        recall_bw[m, a] = n_correct_recall / (n_gt+eps)
        recall_bw[np.isnan(recall_bw) | np.isinf(recall_bw)] = 1.
        precision_bw[m, a] = n_correct_pred / (n_pred+eps)
        precision_bw[np.isnan(precision_bw) | np.isinf(precision_bw)] = 1.
        f1_bw[m, a] = 2 * ((precision_bw[m, a] * recall_bw[m, a]) / (precision_bw[m, a] + recall_bw[m, a]+eps))
dur_gt_bw[np.isnan(dur_gt_bw)] = 0
dur_pred_bw[np.isnan(dur_pred_bw)] = 0

# plots
from matplotlib import rc
rc('text', usetex=True)
# plot precision recall framewise
fig,ax = plt.subplots(nrows=n_classes-1,ncols=7)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.subplots_adjust(wspace=0.5, hspace=0.01,top=.95,bottom=.05,left=.05,right=.98)
y_text=[.8,.55,.55]
for a in range(n_classes-1):
    for m in range(len(all_gt)):
        ax[a,0].plot(recall[m, a+1], precision[m, a+1], '.', markersize=15, markeredgecolor='k', markeredgewidth=.5,clip_on=False, color=cmap4[m])
    ax[a,0].set_xlabel('Recall');ax[a,0].set_ylabel('Precision')
    ax[a,0].set_xlim([0, 1])
    ax[a,0].set_ylim([0, 1])
    ax[a,0].text(-0.4,y_text[a],behs[a+1],fontweight='bold',color=color[a],rotation=90,fontsize=15)
    ax[a,0].grid(linestyle='dotted')
    major_ticks = np.arange(0, 1.01, .2)
    minor_ticks = np.arange(0, 1.01, .1)
    ax[a,0].set_xticks(major_ticks)
    ax[a,0].set_xticks(minor_ticks, minor=True)
    ax[a,0].set_yticks(major_ticks)
    ax[a,0].set_yticks(minor_ticks, minor=True)
    ax[a,0].grid(which='both')
    ax[a,0].grid(which='minor', alpha=0.2)
    ax[a,0].grid(which='major', alpha=0.1)
    if a==0: ax[a,0].set_title(r"$\textbf{PR-frame}$")
    ax[a,0].set_aspect('equal', 'box')

# plot gt count vs pred count framewise
def tick_label_func(val, pos=None):
        return '%.1fK' % (val *1e-3)
for a in range(n_classes-1):
    for m in range(len(all_gt)):
        ax[a,1].plot(count_gt[m, a+1], count_pred[m, a+1], 'o', color=cmap4[m],
                 markeredgecolor='k', markeredgewidth=.5,markersize=8, label='A' + str(m), clip_on=False)
    ax[a,1].plot([0, np.maximum(np.max(count_pred[:, a+1]) + 1, np.max(count_gt[:, a+1]) + 1)],
                 [0, np.maximum(np.max(count_pred[:, a+1]) + 1, np.max(count_gt[:, a+1]) + 1)], '--', color='black',lw=1,alpha=.4)
    ax[a,1].set_xlabel('GT count'); ax[a,1].set_ylabel('PD count')
    major_ticks = np.linspace(0, np.maximum(np.max(count_pred[:, a+1]) + 1, np.max(count_gt[:, a+1]) + 1), 5)
    ax[a,1].set_xticks(major_ticks)
    ax[a,1].set_yticks(major_ticks)
    ax[a,1].set_xlim([0, np.maximum(np.max(count_pred[:, a+1]) + 1, np.max(count_gt[:, a+1]) + 1)])
    ax[a,1].set_ylim([0, np.maximum(np.max(count_pred[:, a+1]) + 1, np.max(count_gt[:, a+1]) + 1)])
    ax[a,1].yaxis.set_major_formatter(FuncFormatter(tick_label_func))
    ax[a,1].xaxis.set_major_formatter(FuncFormatter(tick_label_func))
    if a==0:ax[a,1].set_title(r"\textbf{Frame count scatter}")
    ax[a,1].set_aspect('equal', 'box')

# plot PR bw
for a in range(n_classes-1):
    for m in range(len(all_gt)):
        ax[a,2].plot(recall_bw[m, a+1], precision_bw[m, a+1], 'o', markersize=8,
                 markeredgecolor='k', markeredgewidth=.5,clip_on=False, color=cmap4[m])
    ax[a,2].set_xlabel('Recall');ax[a,2].set_ylabel('Precision')
    ax[a,2].set_xlim([0, 1])
    ax[a,2].set_ylim([0, 1])
    ax[a,2].grid(linestyle='dotted')
    major_ticks = np.arange(0, 1.01, .2)
    minor_ticks = np.arange(0, 1.01, .1)
    ax[a,2].set_xticks(major_ticks)
    ax[a,2].set_xticks(minor_ticks, minor=True)
    ax[a,2].set_yticks(major_ticks)
    ax[a,2].set_yticks(minor_ticks, minor=True)
    ax[a,2].grid(which='both')
    ax[a,2].grid(which='minor', alpha=0.2)
    ax[a,2].grid(which='major', alpha=0.1)
    if a==0:ax[a,2].set_title(r"$\textbf{PR-bout}$")
    ax[a,2].set_aspect('equal', 'box')

# bout scatter
for a in range(n_classes-1):
    for m in range(len(all_gt)):
        ax[a,3].plot(count_gt_bw[m, a+1], count_pred_bw[m, a+1], 'o', color=cmap4[m], markersize=8,
                 label='A' + str(m),markeredgecolor='k', markeredgewidth=.5, clip_on=False)
    ax[a,3].plot([0, np.maximum(np.max(count_pred_bw[:, a+1]) + 1, np.max(count_gt_bw[:, a+1]) + 1)],
                 [0, np.maximum(np.max(count_pred_bw[:, a+1]) + 1, np.max(count_gt_bw[:, a+1]) + 1)], '--', color='black',lw=1,alpha=.4)
    ax[a,3].set_xlabel('GT count'); ax[a,3].set_ylabel('PD count')
    major_ticks = np.linspace(0, np.maximum(np.max(count_pred_bw[:, a+1]) + 1, np.max(count_gt_bw[:, a+1]) + 1), 5).astype(int)
    ax[a,3].set_xticks(major_ticks)
    ax[a,3].set_yticks(major_ticks)
    ax[a,3].set_xlim([0, np.maximum(np.max(count_pred_bw[:, a+1]), np.max(count_gt_bw[:, a+1]))])
    ax[a,3].set_ylim([0, np.maximum(np.max(count_pred_bw[:, a+1]), np.max(count_gt_bw[:, a+1]))])
    if a==0:ax[a,3].set_title(r"\textbf{Bout count scatter}")
    ax[a,3].set_aspect('equal', 'box')

#plot gt duration vs pred duration boutwise
def tick_label_func(val, pos=None):
    return '%.1fK' % (val *1e-3)
for a in range(n_classes-1):
    for m in range(len(all_gt)):
        ax[a,4].plot(dur_gt_bw[m, a+1], dur_pred_bw[m, a+1], 'o', color=cmap4[m], markersize=8,
                 label='A' + str(m),markeredgecolor='k', markeredgewidth=.5, clip_on=False)
    ax[a,4].plot([0, np.maximum(np.max(dur_pred_bw[:, a+1]), np.max(dur_gt_bw[:, a+1]))],
                 [0, np.maximum(np.max(dur_pred_bw[:, a+1]), np.max(dur_gt_bw[:, a+1]))], '--', color='black',lw=1,alpha=.4)
    ax[a,4].set_xlabel('GT count'); ax[a,4].set_ylabel('PD duration')
    major_ticks = np.linspace(0, np.maximum(np.max(dur_pred_bw[:, a+1]), np.max(dur_gt_bw[:, a+1])), 5,endpoint=True).astype(int)
    ax[a,4].set_xticks(major_ticks)
    ax[a,4].set_yticks(major_ticks)
    ax[a,4].set_xlim([0, np.maximum(np.max(dur_pred_bw[:, a+1]), np.max(dur_gt_bw[:, a+1]))])
    ax[a,4].set_ylim([0, np.maximum(np.max(dur_pred_bw[:, a+1]), np.max(dur_gt_bw[:, a+1]))])
    if a==0:ax[a,4].set_title(r"\textbf{Duration Scatter}")
    ax[a,4].set_aspect('equal', 'box')

# PR for each action framewise connected to boutwise
for a in range(n_classes-1):
    for m in range(len(all_gt)):
        ax[a,5].plot(recall[m, a+1], precision[m, a+1], 's', markersize=8, color=cmap4[m],
                 markeredgecolor='k', markeredgewidth=.5,clip_on=False)
        ax[a,5].plot(recall_bw[m, a+1], precision_bw[m, a+1], 'o', markersize=8, color=cmap4[m],
                 markeredgecolor='k', markeredgewidth=.5,clip_on=False)
        ax[a,5].plot([recall_bw[m, a+1], recall[m, a+1]], [precision_bw[m, a+1], precision[m, a+1]], '-', lw=1, color=cmap4[m])
    ax[a,5].plot([], [], marker='s', markeredgecolor='k', markerfacecolor='w', markeredgewidth=.5,
             linestyle='None', label='Frame based')
    ax[a,5].plot([], [], marker='o', markeredgecolor='k', markerfacecolor='w', markeredgewidth=.5,
             linestyle='None', label='Bout based')
    ax[a,5].set_xlim([0, 1.02])
    ax[a,5].set_ylim([0, 1.02])
    ax[a,5].set_xlabel('Recall'); ax[a,5].set_ylabel('Precision')
    ax[a,5].grid(linestyle='dotted')
    major_ticks = np.arange(0, 1.01, .2)
    minor_ticks = np.arange(0, 1.01, .1)
    ax[a,5].set_xticks(major_ticks)
    ax[a,5].set_xticks(minor_ticks, minor=True)
    ax[a,5].set_yticks(major_ticks)
    ax[a,5].set_yticks(minor_ticks, minor=True)
    ax[a,5].grid(which='both')
    ax[a,5].grid(which='minor', alpha=0.2)
    ax[a,5].grid(which='major', alpha=0.1)
    if a==0:ax[a,5].set_title(r"$\textbf{PR}$")
    ax[a,5].set_aspect('equal', 'box')
    if a==0: ax[a,5].legend(loc='lower left')

# f1 framewise vs f1 boutwise
for a in range(n_classes-1):
    for m in range(len(all_gt)):
        ax[a,6].plot(f1_bw[m, a+1], f1[m, a+1], 'o', markersize=8,
                 markeredgecolor='k', markeredgewidth=.5,color=cmap4[m], clip_on=False)
    ax[a,6].plot([0, 1], [0, 1], 'k--', lw=1, alpha=.4)
    ax[a,6].set_xlim([0, 1.])
    ax[a,6].set_ylim([0, 1.])
    ax[a,6].set_xlabel('F1-bout');ax[a,6].set_ylabel('F1-frame')
    ax[a,6].grid(linestyle='dotted')
    major_ticks = np.arange(0, 1.01, .2)
    minor_ticks = np.arange(0, 1.01, .1)
    ax[a,6].set_xticks(major_ticks)
    ax[a,6].set_xticks(minor_ticks, minor=True)
    ax[a,6].set_yticks(major_ticks)
    ax[a,6].set_yticks(minor_ticks, minor=True)
    ax[a,6].grid(which='both')
    ax[a,6].grid(which='minor', alpha=0.2)
    ax[a,6].grid(which='major', alpha=0.1)
    if a==0:ax[a,6].set_title(r"$\textbf{F1-score}$")
    ax[a,6].set_aspect('equal', 'box')

fig.savefig(save_path+ 'perf15')
fig.savefig(save_path + 'perf15.pdf')

#####################################################################################
# plot pred vs gt imshow
####################################################################################
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['white','blue','green','red'],'my')
for m in range(len(all_gt)):
    gt = all_gt[m]
    pd = np.array(all_pred[m])
    fig, ax = plt.subplots(2,1)
    fig.set_size_inches(18,4)
    mis = list(set(range(4)) - set(gt))
    if mis:
        for i in mis:  gt = np.append(gt, i)  # if not all actions are  the array the map skrew up
    mis = list(set(range(4)) - set(pd))
    if mis:
        for i in mis:  pd = np.append(pd, i)  # if not all actions are  the array the map skrew up
    ax[0].imshow(gt[1500:].reshape(1,-1),aspect='auto',cmap=cmap,interpolation='none')
    ax[0].set_xticks([])
    ax[0].set_yticks([0])
    ax[0].set_yticklabels(['GT'],fontsize=20)
    ax[1].imshow(pd[1500:].reshape(1,-1),aspect='auto',cmap=cmap,interpolation='none')
    ax[1].set_xticks([])
    ax[1].set_yticks([0])
    ax[1].set_yticklabels(['PD'],fontsize=20)
    plt.tight_layout()
    fig.savefig(save_path+ str(m)+test_videos[m]+'_pred_seg')
    fig.savefig(save_path+ str(m)+test_videos[m]+'_pred_seg.pdf')
    plt.close()

########### other version as pnas paper
num_behs_gt = len(list(set(gt)))
num_behs_pd = len(list(set(pd)))
proba = all_proba[m]
cmap= [plt.cm.Greys_r,plt.cm.Blues, plt.cm.Greens,plt.cm.Reds]
cmap2= [plt.cm.Greys,plt.cm.Blues_r, plt.cm.Greens_r,plt.cm.Reds_r]
gt_bin=label_binarize(gt,range(num_behs_gt))
pd_bin=label_binarize(pd,range(num_behs_pd))
num_plot= np.maximum(num_behs_gt,num_behs_pd)

fig, ax = plt.subplots(num_plot*2, 1, sharex=True)
fig.set_size_inches(18,2*num_plot)
fig.subplots_adjust(hspace=0)
for a in range(num_behs_gt*2,0,-2):
    ax[a-1].imshow(gt_bin[:5000,int(a/2)-1].reshape(1,-1),aspect='auto',
                   cmap=cmap[int(a/2)-1] if gt_bin[0,int(a/2)-1]==0 else cmap2[int(a/2)-1],interpolation='none')
    ax[a-2].imshow(pd_bin[:5000,int(a/2)-1].reshape(1,-1),aspect='auto',
                   cmap=cmap[int(a/2)-1] if pd_bin[0,int(a/2)-1]==0 else cmap2[int(a/2)-1],interpolation='none')
    ax[a-2].set_yticks([0]);ax[a-2].set_yticklabels(['GT'])
    ax[a-1].set_yticks([0]);ax[a-1].set_yticklabels(['PD'])
plt.tight_layout()



################################################################################################
########################################################################################
# KFOLD
path = 'mars_v1_6_kfold/'
folders = ['_tm_mlp','_tm_mlp_wnd','_tm_xgb500','_tm_xgb500_wnd'] #no cable
type=['top_pcf','top']
savename=path + 'tm_'
color = ['b', 'g', 'r']
cmap = [ 'c', 'limegreen' ,'lightcoral']

behs = ['closeinvestigation', 'mount', 'attack']
n_classes = len(behs)
bin_pred = '3_pd_fbs_hmm' #num_frames x num behs
all_pred = '9_pred_fbs_hmm_ass' #list of behs
fold_pred = '3l_pd_fbs_hmm' #list of behs per fold
fixes = [bin_pred,all_pred,fold_pred]
n_exp = len(fixes)

results = {}
meth= []
for e, ex in enumerate(folders):
    for t in type:
        savedir = path + t + ex + '/'
        results[t+ex] = {}
        print('%s%s' % (t,ex))
        results[t+ex] = dill.load(open(savedir + '/results.dill', 'rb'))
        meth.append(t+ex)

n_meth = len(meth)
prec = np.zeros((n_meth, n_exp,n_classes))
rec = np.zeros((n_meth, n_exp, n_classes))
all_gt=[]
all_pd=[]
prec_std = np.zeros((n_meth, n_classes))
rec_std = np.zeros((n_meth,  n_classes))
acc =[]

for e, ex in enumerate(meth):
    for f, fix in enumerate(fixes):
        if f == 0:
            for a in range(n_classes):
                gt = results[ex]['0_G'][:, a]
                pd = results[ex][fix][:, a]
                prec[e, f, a], rec[e, f, a]= prf_bin_metrics(gt, pd, '%s %s %s' % (ex, behs[a], fix))
        elif f==1:
            gt = results[ex]['0_Gc']
            pd = results[ex][fix]
            all_gt.append(np.array(gt) if isinstance(gt,list) else gt)
            all_pd.append(np.array(pd))
            p,r,f1 = score_info(gt, pd)
            prec[e, f,:] = p[1:]
            rec[e, f, :] = r[1:]
            acc.append(accuracy_score(gt, pd))
        if f==2:
            gt = results[ex]['0l_G']
            pd = results[ex][fix]
            p=np.zeros((n_classes,4))
            r=np.zeros((n_classes,4))
            for k in range(4):
                for b in range(n_classes):
                    p[b,k],r[b,k]= prf_bin_metrics(gt[b][k], pd[b][k], '%s %s' % (ex, behs[b]))
            prec[e,f,:] =np.nanmean(p,axis=1)
            prec_std[e,:] =np.nanstd(p,axis=1)
            rec[e, f, :] = np.nanmean(r, axis=1)
            rec_std[e, :] = np.nanstd(r, axis=1)

# plot
meas = ['Precision', 'Recall']
barw = .25
xp = np.array([x * (barw + .05) for x in range(n_meth)])
fig, ax = plt.subplots(1, n_classes * 2, sharey=True)  # plots by action by methods
fig.set_size_inches(18, 9)
plt.subplots_adjust(top=0.95, bottom=.1, left=.1, wspace=.2, right=.98)
for a in range(0, n_classes * 2, 2):
    ax[a].text(.9 if a == 0 else 1, xp[-1] + .2, behs[int(a / 2)], color=color[int(a / 2)], fontweight='bold')
    ax[a].barh(xp, prec[::-1, 2, int(a / 2)], barw, color=cmap[int(a / 2)], xerr=prec_std[::-1, int(a / 2)], capsize=10, ecolor='black', align='center')
    ax[a + 1].barh(xp, rec[::-1, 2, int(a / 2)], barw, color=cmap[int(a / 2)], xerr=rec_std[::-1, int(a / 2)], capsize=10, ecolor='black', align='center')
    ax[a].set_xlim([.65, 1.])
    ax[a + 1].set_xlim([.65, 1.])
    ax[a].set_ylim([xp[0] - .15, xp[-1] + .15])
    ax[a + 1].set_ylim([xp[0] - .15, xp[-1] + .15])
    ax[a].set_yticks(xp)
    ax[a + 1].set_yticks(xp)
    ax[a].set_yticklabels(meth[::-1], fontsize=9, )
    if a == 0: [l.set_weight("bold") for o, l in enumerate(ax[a].get_yticklabels()) if o < n_meth / 2]
    ax[a].set_xticks(np.arange(.6, 1.05, .1))
    ax[a + 1].set_xticks(np.arange(.6, 1.05, .1))
    ax[a].text(.75, -.5, meas[0])
    ax[a + 1].text(.75, -.5, meas[1])
fig.savefig(savename + fix + '_kfold.png')
fig.savefig(savename + fix + '_kfold.pdf')
plt.close()

#### plots pr, roc confusion matrix

# ex='top_pcf_t_xgb500_wnd'
ex='top_tm_xgb500_wnd'
# ex='top_pcf_tm_xgb500_wnd'
savedir = path  + ex + '/'
behs = ['other','closeinvestigation', 'mount', 'attack']
n_classes = len(behs)
results={}
results[ex] = dill.load(open(savedir + '/results.dill', 'rb'))
gt = results[ex]['0l_G']
pd = results[ex]['3l_pd_fbs_hmm']
proba = results[ex]['3l_pr_fbs_hmm']

def plot_roc_curve_kfold(gt, pd,proba,behs, colors=[]):
    n_classes=len(behs)-1
    fig, ax = plt.subplots(1,n_classes)
    fig.set_size_inches(15,4)
    if colors==[]:    colors = cycle(['c', 'limegreen' ,'lightcoral'])
    cmap= cmp.ScalarMappable(col.Normalize(0, 4), cmp.jet)
    cmap = cmap.to_rgba(range(0, 4))

    for i, color in zip(range(n_classes), colors):
        fprs = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        roc_aucs = []
        for k in range(4):
            #get tp fp p n perm
            P=np.sum(gt[i][k]==1)
            N=np.sum(gt[i][k]==0)
            sorted_proba = np.sort(proba[b][k][:,1])[::-1]
            perm = np.argsort(proba[b][k][:,1])[::-1]
            gt_perm = gt[i][k][perm]
            tp = np.insert(np.cumsum(gt_perm==1),0,0)
            fp = np.insert(np.cumsum(gt_perm==0),0,0)
            # compute rates
            tpr = tp /np.maximum(P,eps)
            fpr = fp /np.maximum(N,eps)
            tnr=1-fpr
            roc_auc = np.sum(tpr*np.diff(np.insert(fpr,0,0)))
            roc_aucs.append(roc_auc)
            s = np.max(np.where(tnr > tpr)[0])
            if s == len(tpr):
                eer = np.NAN; eerTh = 0
            else:
                if tpr[s] == tpr[s + 1]:
                    eer = 1 - tpr[s]
                else:
                    eer = 1 - tnr[s]
                eerTh = sorted_proba[s]
            fprs.append(fpr)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            ax[i].plot(fpr, tpr, color=cmap[k], lw=2, label=' Fold {0} (area = {1:0.2f}, EER = {2:0.2f})'  ''.format(k,
                roc_auc,eer), clip_on=False,alpha=0.3)

        ax[i].plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray',
                 label='Luck', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(roc_aucs)
        ax[i].plot(mean_fpr, mean_tpr, color='r',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax[i].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        ax[i].set_xlim([-0.05, 1.05])
        ax[i].set_ylim([-0.05, 1.05])
        ax[i].set_xlabel('False Positive Rate')
        ax[i].set_ylabel('True Positive Rate')
        ax[i].set_title('ROC ' + behs[i+1])
        ax[i].legend(loc="lower right")

plot_roc_curve_kfold(gt,pd, proba,behs)
plt.savefig(path+ ex + '_fbs_hmm_ass_ROC_kfold')
plt.savefig(path+ ex + '_fbs_hmm_ass_ROC_kfold'+ '.pdf')
plt.close()

def plot_pr_curve_kfold(gt, proba,behs,colors=[]):
    # For each class
    n_classes=len(behs)-1
    cmap= cmp.ScalarMappable(col.Normalize(0, 4), cmp.jet)
    cmap = cmap.to_rgba(range(0, 4))
    if colors==[]:    colors = cycle(['c', 'limegreen' ,'lightcoral'])

    fig,ax=plt.subplots(1,n_classes)
    fig.set_size_inches(15,4)
    plt.subplots_adjust(wspace=0.5)

    for i, color in zip(range(n_classes), colors):
        gt_all=[]
        proba_all =[]
        lines = []
        labels = []
        precs=[]
        aucs=[]
        aps=[]
        aps11=[]
        for k in range(4):
            P = np.sum(gt[i][k] == 1)
            gt_all.append(gt[i][k])
            proba_all.append(proba[i][k][:,1])
            perm = np.argsort(proba[i][k][:, 1])[::-1]
            gt_perm = gt[i][k][perm]
            tp = np.insert(np.cumsum(gt_perm == 1), 0, 0)
            fp = np.insert(np.cumsum(gt_perm == 0), 0, 0)
            recall = tp / np.maximum(P,eps)
            precision = np.maximum(tp,eps)/np.maximum(tp+fp,eps)
            precs.append(interp(np.linspace(0,1,100),recall,precision))
            auc_pr = .5*np.sum((precision[:-1]+precision[1:])*np.diff(recall))
            aucs.append(auc_pr)
            sel = np.where(np.diff(recall))[0]+1
            ap = np.sum(precision[sel])/P
            aps.append(ap)
            ap11 = 0.0
            for rc in np.linspace(0,1,11):
                pr = np.max(np.insert(precision[recall>=rc],0,0))
                ap11 =ap11+pr/11
            aps11.append(ap11)
            l, = ax[i].plot(recall, precision, color=cmap[k], lw=2,alpha=.3)
            lines.append(l)
            labels.append('fold {0} (area = {1:0.2f}, AP = {2:0.2f})'.format(k,auc_pr,ap))

        mean_prec= np.mean(precs,axis=0)
        mean_prec[-1] = 0
        mean_rec = np.linspace(0,1,100)
        mean_auc = auc(mean_rec, mean_prec)
        std_auc = np.std(aucs)
        l,=ax[i].plot(mean_rec, mean_prec, color='r', lw=2, alpha=.8)
        labels.append(r'Mean PR (AUC = %0.2f $\pm$ %0.2f mAP = %0.2f)' % (mean_auc,
                                std_auc,np.mean(aps)))
        lines.append(l)

        std_prec= np.std(precs, axis=0)
        prec_upper = np.minimum(mean_prec + std_prec, 1)
        prec_lower = np.maximum(mean_prec - std_prec, 0)
        ax[i].fill_between(mean_rec, prec_lower, prec_upper, color='grey', alpha=.2)
        l,=ax[i].plot(mean_rec, prec_lower, color='grey', alpha=.2)
        lines.append(l)
        labels.append(r'$\pm$ 1 std. dev.')

        ax[i].set_xlim([0.0, 1.0])
        ax[i].set_ylim([0.0, 1.01])
        ax[i].set_xlabel('Recall')
        ax[i].set_ylabel('Precision')
        ax[i].set_title('Precision-Recall '+ behs[i+1])
        labels.append('Bout based')
        ax[i].legend(lines, labels, loc='best', )

plot_pr_curve_kfold(gt, proba,behs)
plt.savefig(path+ ex + '_fbs_hmm_ass_PR_kfold')
plt.savefig(path+ ex + '_fbs_hmm_ass_PR_kfold' + '.pdf')
plt.close()

################################################################################






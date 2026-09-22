"""Render thesis figures from saved results and predictor data; never refit models."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/thesis';OUT=DATA/'figures'
LABELS={'lcavol':'Log cancer volume','lweight':'Log prostate weight','age':'Age','lbph':'Log BPH amount','svi':'Seminal vesicle invasion','lcp':'Log capsular penetration','gleason':'Gleason score','pgg45':'Gleason 4 or 5 percentage'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)
def plot_r_squared(summary):
    """Compare mean held-out fold R², using the same scores as the results table."""
    ordered=summary.sort_values('r2_mean',ascending=False).reset_index(drop=True)
    labels=ordered.model.replace({'Linear Regression':'OLS','Random Forest':'Random forest','Gradient Boosting':'Gradient boosting','Decision Tree':'Decision tree'})
    fig,ax=plt.subplots(figsize=(7.1,3.9))
    y=np.arange(len(ordered))
    colors=['#999999' if name=='Mean baseline' else '#386f95' for name in ordered.model]
    ax.barh(y,ordered.r2_mean,color=colors,height=.66,zorder=3)
    ax.set_yticks(y,labels)
    ax.invert_yaxis()
    ax.axvline(0,color='#333333',lw=.9,zorder=4)
    ax.set_xlim(min(-.17,ordered.r2_mean.min()-.08),max(.63,ordered.r2_mean.max()+.08))
    ax.set_xlabel('Mean held-out fold R² (higher is better)')
    ax.grid(axis='x',color='#dddddd',lw=.65,zorder=0)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0)
    for i,value in enumerate(ordered.r2_mean):
        ax.text(value+(.009 if value>=0 else -.009),i,f'{value:.3f}',
                ha='left' if value>=0 else 'right',va='center',fontsize=10)
    evaluations=int(ordered.evaluations.iloc[0])
    ax.set_title(f'R² across {evaluations} held-out evaluations per model',fontsize=12,pad=12)
    fig.tight_layout()
    save(fig,'05_r_squared_comparison')

def plot_feature_correlations(data_path):
    """Describe the eight predictors; exclude the outcome and split marker."""
    data=pd.read_csv(data_path)
    features=list(LABELS)
    corr=data[features].corr(method='pearson')
    corr.to_csv(DATA/'predictor_correlations.csv',float_format='%.12g')
    fig,ax=plt.subplots(figsize=(6.6,5.45))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(features)),features,rotation=45,ha='right')
    ax.set_yticks(range(len(features)),features)
    for i in range(len(features)):
        for j in range(len(features)):
            value=corr.iloc[i,j]
            ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=10,
                    color='white' if abs(value)>.55 else '#111111')
    ax.set_xticks(np.arange(-.5,len(features),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(features),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.7)
    ax.tick_params(which='minor',bottom=False,left=False)
    ax.set_title('Correlations among the eight predictors',pad=12,fontsize=12)
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04,ticks=[-1,-.5,0,.5,1])
    cb.set_label('Pearson correlation (r)')
    fig.tight_layout()
    save(fig,'06_feature_correlations')

def main(results_dir=DATA,data_path=ROOT/'data/prostate.csv'):
    global DATA, OUT
    DATA=Path(results_dir);OUT=DATA/'figures'
    s=pd.read_csv(DATA/'model_cv_summary.csv').sort_values('rmse_mean',ascending=False)
    f=pd.read_csv(DATA/'fold_metrics.csv')
    fig,ax=plt.subplots(figsize=(7.3,4.6),layout='constrained')
    for i,r in enumerate(s.itertuples()):
        values=f.loc[f.model==r.model,'rmse'].to_numpy()
        jitter=np.random.default_rng(42+i).uniform(-.13,.13,len(values))
        ax.scatter(values,i+jitter,s=10,color='#b4b9be',alpha=.65,zorder=1)
        ax.errorbar(r.rmse_mean,i,xerr=r.rmse_sd,fmt='o',color='#155f73',capsize=4,zorder=3)
    ax.set_yticks(range(len(s)),s.model);ax.set_xlabel('Held-out RMSE in log PSA units (lower is better)')
    ax.set_title('Model performance across 50 held-out folds',loc='left',weight='bold')
    ax.grid(axis='x',alpha=.2);fig.supxlabel('Dots: individual folds. Blue: mean ± SD. SD is not a confidence interval.',fontsize=9)
    save(fig,'01_model_performance')
    p=pd.read_csv(DATA/'oof_predictions.csv');p=p[(p.model=='Linear Regression')&(p['repeat']==1)]
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.5),layout='constrained')
    lo=min(p.observed.min(),p.predicted.min());hi=max(p.observed.max(),p.predicted.max())
    axes[0].scatter(p.observed,p.predicted,color='#155f73',s=20,alpha=.7);axes[0].plot([lo,hi],[lo,hi],'--',color='black',lw=1)
    axes[0].set(xlabel='Observed log PSA',ylabel='Predicted log PSA',title='Predictions')
    axes[1].scatter(p.predicted,p.observed-p.predicted,color='#155f73',s=20,alpha=.7);axes[1].axhline(0,ls='--',color='black',lw=1)
    axes[1].set(xlabel='Predicted log PSA',ylabel='Observed minus predicted',title='Residuals')
    fig.suptitle('Linear regression: first cross-validation repeat',weight='bold')
    fig.supxlabel('One held-out prediction per patient (n = 97); no training predictions shown.',fontsize=9)
    save(fig,'02_ols_predictions_residuals')
    raw=pd.read_csv(DATA/'permutation_importance_raw.csv')
    fold=raw.groupby(['feature','fold']).importance.mean().reset_index()
    order=fold.groupby('feature').importance.mean().sort_values().index
    fig,ax=plt.subplots(figsize=(7.3,4.3),layout='constrained')
    for i,feature in enumerate(order):
        vals=fold[fold.feature==feature].importance.to_numpy()
        ax.scatter(vals,np.repeat(i,len(vals)),color='#c18b3a',s=22,alpha=.8)
        ax.scatter(vals.mean(),i,color='#155f73',s=60,marker='D',zorder=3)
    ax.set_yticks(range(len(order)),[LABELS[x] for x in order]);ax.axvline(0,color='black',lw=1,ls='--')
    ax.set_xlabel('Increase in held-out RMSE after feature shuffling')
    ax.set_title('Predictive reliance of the fixed linear model',loc='left',weight='bold')
    fig.supxlabel('Gold: five fold means (30 shuffles each). Blue: overall mean. Not causal effects.',fontsize=9)
    save(fig,'03_permutation_importance')
    co=pd.read_csv(DATA/'coefficients_by_fold.csv')
    order=co.groupby('feature').coefficient_per_training_sd.mean().sort_values().index
    fig,ax=plt.subplots(figsize=(7.3,4.3),layout='constrained')
    for i,feature in enumerate(order):
        vals=co[co.feature==feature].coefficient_per_training_sd
        ax.scatter(vals,np.repeat(i,len(vals)),s=14,color='#b4b9be',alpha=.65)
        ax.scatter(vals.mean(),i,color='#155f73',s=55,marker='D')
    ax.set_yticks(range(len(order)),[LABELS[x] for x in order]);ax.axvline(0,color='black',ls='--',lw=1)
    ax.set_xlabel('Log PSA coefficient per training-fold predictor SD')
    ax.set_title('Linear coefficient variation across fitted folds',loc='left',weight='bold')
    fig.supxlabel('Gray: 50 fitted coefficients. Blue: mean. Conditional associations, not causal effects.',fontsize=9)
    save(fig,'04_coefficient_stability')
    plot_r_squared(s)
    plot_feature_correlations(data_path)
    print('Saved six figures in PNG, PDF, and SVG:',OUT)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=DATA)
    parser.add_argument('--data-path',type=Path,default=ROOT/'data/prostate.csv')
    args=parser.parse_args()
    main(args.results_dir,args.data_path)

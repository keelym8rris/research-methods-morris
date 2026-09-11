"""Repeated cross-validation for the thesis. Run from any directory; no network required.
Fixed exploratory settings, common outer folds, natural-log PSA errors.
"""
from pathlib import Path
import argparse, hashlib, json, platform, warnings
from datetime import date
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.data_loader import get_predictors_and_target
from src.research_analysis import build_models, held_out_permutation_importance
ROOT=Path(__file__).resolve().parent
FEATURES=['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']
SEED=42

def thesis_models():
    models=build_models(SEED)
    # Use fixed settings and train-fold scaling for the neural model.
    models.pop('Neural Network')
    models['MLP (L-BFGS)']=TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(),MLPRegressor(
            hidden_layer_sizes=(16,),activation='relu',solver='lbfgs',alpha=1.0,
            max_iter=4000,max_fun=50000,random_state=SEED,early_stopping=False)),
        transformer=StandardScaler())
    models['Ridge']=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    models['Lasso']=make_pipeline(StandardScaler(),Lasso(alpha=0.05,max_iter=10000))
    models['Mean baseline']=DummyRegressor(strategy='mean')
    models['Random Forest'].set_params(n_jobs=1)
    return models

def metrics(y,pred):
    return {'rmse':float(mean_squared_error(y,pred)**0.5),
            'mae':float(mean_absolute_error(y,pred)),'r2':float(r2_score(y,pred))}

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data_path=ROOT/'data/prostate.csv';data=pd.read_csv(data_path)
    X,y=get_predictors_and_target(data)
    if list(X.columns)!=FEATURES:raise ValueError('Unexpected predictor schema')
    if len(data)!=97 or not np.isfinite(data.select_dtypes('number')).all().all():
        raise ValueError('Expected 97 complete numeric records')
    if data.duplicated().any():raise ValueError('Duplicate records require review')
    split_records=[];predictions=[];fold_metrics=[];coefs=[];warning_records=[]
    splits=list(RepeatedKFold(n_splits=5,n_repeats=10,random_state=SEED).split(X))
    models=thesis_models()
    for k,(train,test) in enumerate(splits):
        repeat,fold=k//5+1,k%5+1
        split_records.append({'repeat':repeat,'fold':fold,'train_rows':(train+1).tolist(),'test_rows':(test+1).tolist()})
        assert not set(train)&set(test)
        for name,estimator in models.items():
            model=clone(estimator)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(X.iloc[train],y.iloc[train])
            for w in caught:warning_records.append({'model':name,'repeat':repeat,'fold':fold,'category':w.category.__name__,'message':str(w.message)})
            pred=model.predict(X.iloc[test])
            fold_metrics.append({'model':name,'repeat':repeat,'fold':fold,'n_test':len(test),**metrics(y.iloc[test],pred)})
            for idx,value in zip(test,pred):
                predictions.append({'model':name,'repeat':repeat,'fold':fold,'row_id':int(idx+1),'observed':float(y.iloc[idx]),'predicted':float(value)})
            if name=='Linear Regression':
                for feature,coef in zip(FEATURES,model.named_steps['model'].coef_):
                    coefs.append({'repeat':repeat,'fold':fold,'feature':feature,'coefficient_per_training_sd':float(coef)})
        print('Completed repeat',repeat,'fold',fold,flush=True)
    folds=pd.DataFrame(fold_metrics);preds=pd.DataFrame(predictions)
    summary=folds.groupby('model').agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'),r2_mean=('r2','mean'),r2_sd=('r2','std'),evaluations=('rmse','size')).reset_index().sort_values('rmse_mean')
    repeat_metrics=[]
    for (name,repeat),g in preds.groupby(['model','repeat']):
        assert len(g)==97 and g.row_id.nunique()==97
        repeat_metrics.append({'model':name,'repeat':int(repeat),**metrics(g.observed,g.predicted)})
    # Explain fixed OLS, not a model selected independently from these folds.
    raw,importance=held_out_permutation_importance(X,y,models['Linear Regression'])
    cf=pd.DataFrame(coefs)
    cfsummary=cf.groupby('feature').coefficient_per_training_sd.agg(['mean','std','min','max']).reset_index()
    opt={'index':False,'float_format':'%.12g'}
    tables={'fold_metrics':folds,'model_cv_summary':summary,'oof_predictions':preds,'repeat_metrics':pd.DataFrame(repeat_metrics),'permutation_importance_raw':raw,'permutation_importance_summary':importance,'coefficients_by_fold':cf,'coefficient_summary':cfsummary,'descriptive_statistics':data[FEATURES+['lpsa']].describe().T.reset_index(names='variable')}
    for name,frame in tables.items():frame.to_csv(output/(name+'.csv'),**opt)
    metadata={'analysis_date':date.today().isoformat(),'dataset_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'patients':len(data),'predictors':FEATURES,'excluded_columns':['train','lpsa'],'missing_values':int(data.isna().sum().sum()),'duplicate_rows':int(data.duplicated().sum()),'target':'natural-log PSA (lpsa), not cancer risk','validation':'5 folds x 10 repeats; identical splits for all models','seed':SEED,'ranking':'mean fold RMSE; exploratory comparison, no statistical superiority claim','importance_model':'fixed OLS; first five folds overlap evaluation; descriptive, not independent validation','settings':{name:repr(model) for name,model in models.items()},'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'warnings':warning_records}
    (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'fold_assignments.json').write_text(json.dumps(split_records,indent=2)+'\n')
    print(summary.to_string(index=False));print('Warnings:',len(warning_records))
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'results/thesis');run(p.parse_args().output_dir)

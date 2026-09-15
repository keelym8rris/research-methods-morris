"""Independent evidence checks; run with python tests/verify_results.py."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data_loader import get_predictors_and_target
from run_thesis_analysis import thesis_models

def matches_data_hash(content,recorded_hash):
    """Allow only LF/CRLF conversion from Git checkout; preserve all other bytes."""
    lf=content.replace(b'\r\n',b'\n')
    return recorded_hash in {
        hashlib.sha256(value).hexdigest()
        for value in (content,lf,lf.replace(b'\n',b'\r\n'))
    }

def main(results_dir=ROOT/'results/thesis'):
    out=Path(results_dir);data=pd.read_csv(ROOT/'data/prostate.csv')
    pred=pd.read_csv(out/'oof_predictions.csv');folds=pd.read_csv(out/'fold_metrics.csv')
    splits=json.loads((out/'fold_assignments.json').read_text())
    X,y=get_predictors_and_target(data)
    model_names=set(thesis_models())
    metadata=json.loads((out/'run_metadata.json').read_text())
    assert matches_data_hash((ROOT/'data/prostate.csv').read_bytes(),metadata['dataset_sha256'])
    assert list(X.columns)==['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']
    assert len(pred)==7760 and len(folds)==400
    assert set(pred.model)==set(folds.model)==model_names
    assert not pred.duplicated(['model','repeat','row_id']).any()
    assert not folds.duplicated(['model','repeat','fold']).any()
    expected_splits={(repeat,fold) for repeat in range(1,11) for fold in range(1,6)}
    assert len(splits)==50
    assert {(s['repeat'],s['fold']) for s in splits}==expected_splits
    assert set(zip(folds['repeat'],folds.fold))==expected_splits
    for split in splits:
        tr=set(split['train_rows']);te=set(split['test_rows'])
        assert tr.isdisjoint(te) and tr|te==set(range(1,98))
        g=pred[(pred['repeat']==split['repeat'])&(pred.fold==split['fold'])]
        assert set(g.model)==model_names
        for name,group in g.groupby('model'):
            assert set(group.row_id)==te
            truth=data.iloc[group.row_id.to_numpy()-1].lpsa.to_numpy()
            np.testing.assert_allclose(truth,group.observed,atol=1e-10)
            err=truth-group.predicted.to_numpy()
            expected=[np.sqrt(np.mean(err**2)),np.mean(abs(err)),1-np.sum(err**2)/np.sum((truth-truth.mean())**2)]
            actual=folds[(folds.model==name)&(folds['repeat']==split['repeat'])&(folds.fold==split['fold'])][['rmse','mae','r2']].iloc[0]
            np.testing.assert_allclose(expected,actual,atol=1e-9)
            assert folds[(folds.model==name)&(folds['repeat']==split['repeat'])&(folds.fold==split['fold'])].n_test.iloc[0]==len(te)
            if name=='Mean baseline':np.testing.assert_allclose(group.predicted,data.iloc[np.array(sorted(tr))-1].lpsa.mean(),atol=1e-10)
    for _,g in pred.groupby(['model','repeat']):assert len(g)==97 and g.row_id.nunique()==97
    summary=pd.read_csv(out/'model_cv_summary.csv').set_index('model').sort_index()
    assert len(summary)==8 and set(summary.index)==model_names
    for metric in ['rmse','mae','r2']:
        expected=folds.groupby('model')[metric].agg(['mean','std']).sort_index()
        np.testing.assert_allclose(summary[[metric+'_mean',metric+'_sd']],expected,rtol=1e-9,atol=1e-9)
    np.testing.assert_array_equal(summary.evaluations,folds.groupby('model').size().sort_index())
    repeats=pd.read_csv(out/'repeat_metrics.csv').set_index(['model','repeat'])
    assert len(repeats)==80 and repeats.index.is_unique
    assert set(repeats.index)=={(name,repeat) for name in model_names for repeat in range(1,11)}
    for key,g in pred.groupby(['model','repeat']):
        err=g.observed-g.predicted
        expected=[np.sqrt(np.mean(err**2)),np.mean(abs(err)),1-np.sum(err**2)/np.sum((g.observed-g.observed.mean())**2)]
        np.testing.assert_allclose(repeats.loc[key,['rmse','mae','r2']],expected,rtol=1e-9,atol=1e-9)
    # Directly check preprocessing against a known training subset.
    split=splits[0];train=np.array(split['train_rows'])-1;test=np.array(split['test_rows'])-1
    model=thesis_models()['Linear Regression'];model.fit(X.iloc[train],y.iloc[train])
    np.testing.assert_allclose(model.named_steps['scale'].mean_,X.iloc[train].mean())
    saved=pred[(pred.model=='Linear Regression')&(pred['repeat']==1)&(pred.fold==1)].sort_values('row_id')
    np.testing.assert_allclose(model.predict(X.iloc[test]),saved.predicted,atol=1e-9)
    # Perturb held-out data: fitted model/scaler remain unchanged.
    initial=model.named_steps['scale'].mean_.copy();_ = model.predict(X.iloc[test]*1000)
    np.testing.assert_array_equal(initial,model.named_steps['scale'].mean_)
    print('PASS: data hash, predictor schema, 400 fold metrics, 7760 prediction memberships, model and repeat summaries, training-mean baseline, train-only scaling.')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results/thesis')
    main(parser.parse_args().results_dir)

"""Independent evidence checks; run with python tests/verify_audited_outputs.py."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data_loader import get_predictors_and_target
from run_audited_analysis import audited_models

def main():
    out=ROOT/'results/audited';data=pd.read_csv(ROOT/'data/prostate.csv')
    pred=pd.read_csv(out/'oof_predictions.csv');folds=pd.read_csv(out/'fold_metrics.csv')
    splits=json.loads((out/'fold_assignments.json').read_text())
    X,y=get_predictors_and_target(data)
    assert list(X.columns)==['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']
    assert len(pred)==7760 and len(folds)==400
    for split in splits:
        tr=set(split['train_rows']);te=set(split['test_rows'])
        assert tr.isdisjoint(te) and tr|te==set(range(1,98))
        g=pred[(pred['repeat']==split['repeat'])&(pred.fold==split['fold'])]
        for name,group in g.groupby('model'):
            assert set(group.row_id)==te
            truth=data.iloc[group.row_id.to_numpy()-1].lpsa.to_numpy()
            np.testing.assert_allclose(truth,group.observed,atol=1e-10)
            err=truth-group.predicted.to_numpy()
            expected=[np.sqrt(np.mean(err**2)),np.mean(abs(err)),1-np.sum(err**2)/np.sum((truth-truth.mean())**2)]
            actual=folds[(folds.model==name)&(folds['repeat']==split['repeat'])&(folds.fold==split['fold'])][['rmse','mae','r2']].iloc[0]
            np.testing.assert_allclose(expected,actual,atol=1e-9)
            if name=='Mean baseline':np.testing.assert_allclose(group.predicted,data.iloc[np.array(sorted(tr))-1].lpsa.mean(),atol=1e-10)
    for _,g in pred.groupby(['model','repeat']):assert len(g)==97 and g.row_id.nunique()==97
    # Directly check preprocessing against a known training subset.
    split=splits[0];train=np.array(split['train_rows'])-1;test=np.array(split['test_rows'])-1
    model=audited_models()['Linear Regression'];model.fit(X.iloc[train],y.iloc[train])
    np.testing.assert_allclose(model.named_steps['scale'].mean_,X.iloc[train].mean())
    saved=pred[(pred.model=='Linear Regression')&(pred['repeat']==1)&(pred.fold==1)].sort_values('row_id')
    np.testing.assert_allclose(model.predict(X.iloc[test]),saved.predicted,atol=1e-9)
    # Perturb held-out data: fitted model/scaler remain unchanged.
    initial=model.named_steps['scale'].mean_.copy();_ = model.predict(X.iloc[test]*1000)
    np.testing.assert_array_equal(initial,model.named_steps['scale'].mean_)
    print('PASS: predictor schema, 400 fold metrics recalculated, 7760 prediction memberships, training-mean baseline, train-only scaling.')
if __name__=='__main__':main()

"""HG 交易的两个校验：按入场日聚合的 t，以及同股票随机入场对照。先跑 hg_study.py --pool csi300 生成交易明细。"""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
import hg_study as H, pandas as pd, numpy as np
rng=np.random.default_rng(0)
codes=H.universe('csi300')
for tf in ('60m','1D(全史)'):
    t=pd.read_csv(H.CACHE/f'trades_csi300_{tf}.csv',parse_dates=['entry_time'])
    # 按入场日聚合：同日多笔取均值，再算 t
    day=t.groupby(t.entry_time.dt.normalize())['R'].mean()
    print(f"{tf}: 逐笔 t={t.R.mean()/t.R.std()*np.sqrt(len(t)):.2f}  按日聚合 {len(day)} 天 mean={day.mean():.3f} t={day.mean()/day.std()*np.sqrt(len(day)):.2f}")
    # 随机对照：同股票随机 bar 入场，止损距离 = 该股实际交易的 risk/ATR
    out=[]
    for code,g in t.groupby('code'):
        h=H.load_60m(code,False)
        d=H.prepare(h,True) if tf=='60m' else H.prepare(H.load_1d(code),False)
        d.attrs['limit']=0.2 if code[2:5] in ('688','300','301') else 0.1
        m=(g.risk_pct*d.close.iloc[g.sig.clip(upper=len(d)-1)].to_numpy()/g.atr).to_numpy()
        lo_bar=max(200, g.sig.min()-5) if tf=='60m' else g.sig.min()
        for _ in range(3):
            for mi in m:
                k=int(rng.integers(20,len(d)-2)) if tf=='60m' else int(rng.integers(lo_bar, len(d)-2))
                a=d.atr.iat[k]; e=d.open.iat[k+1]
                s=dict(side=1,sig=k,lo=e-mi*a+0.1*a,hi=np.nan,mid=np.nan,wrb_lo=np.nan,wrb_hi=np.nan,atr=a)
                r=H.simulate(d,s,2.0,'A',True)
                if r: out.append(r['R'])
    out=np.array(out)
    print(f"   随机对照 n={len(out)} AvgR={out.mean():.3f} Win={(out>0).mean():.0%}  → HG 超额 {t.R.mean()-out.mean():+.3f}R")

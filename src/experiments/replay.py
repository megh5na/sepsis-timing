from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.parse import FEATURE_COLUMNS, load_patient_tensor
from src.decision.stopping import margin, onset_distribution
from src.experiments.evaluate_runs import EVAL
from src.experiments.run_phase2 import PROCESSED, RUNS, get_bundle

SEED = 0
SHOW = ["HR", "MAP", "Resp", "O2Sat", "Temp"]
OUT = Path("results/phase2/replay")


def pick_patients(ctx_test, a1, a2, utilities1, utilities2) -> list[tuple[str, int]]:
    T, septic = ctx_test.lengths, ctx_test.is_septic
    al1, al2 = a1 < T, a2 < T
    d = utilities2 - utilities1
    cand = {
        "septic: P2 alarms earlier than P1": np.flatnonzero(septic & al1 & al2 & (a2 < a1)),
        "septic: P2 alarms later than P1": np.flatnonzero(septic & al1 & al2 & (a2 > a1)),
        "septic: both alarm within an hour of each other": np.flatnonzero(septic & al1 & al2 & (np.abs(a1 - a2) <= 1)),
        "septic: neither rule alarms": np.flatnonzero(septic & ~al1 & ~al2),
        "non-septic: both rules raise a false alarm": np.flatnonzero(~septic & al1 & al2),
        "non-septic: no alarm (correct)": np.flatnonzero(~septic & ~al1 & ~al2),
    }
    picks = []
    for title, idx in cand.items():
        if len(idx) == 0:
            continue
        if "earlier" in title:
            i = idx[np.argmax(d[idx])]
        elif "later" in title:
            i = idx[np.argmin(d[idx])]
        else:
            i = idx[len(idx) // 2]
        picks.append((title, int(i)))
    return picks


def build():
    bundle, _ = get_bundle()
    test = bundle.test
    F_all = np.load(RUNS / "P" / f"seed{SEED}" / "preds_test.npz")["F"].astype(np.float64)
    ev = np.load(EVAL / f"P_seed{SEED}.npz")
    meta = json.loads((RUNS / "P" / f"seed{SEED}" / "meta.json").read_text())
    from src.eval.fast_utility import UtilityTable
    tab = UtilityTable.from_split(test)
    o1, o2 = tab.observed(ev["p1_test"]), tab.observed(ev["p2_test"])
    picks = pick_patients(test, ev["p1_test"], ev["p2_test"], o1, o2)

    patients = []
    for title, i in picks:
        T = int(test.lengths[i]); off = test.offsets[i]
        Fi = F_all[off:off + T]
        p, p_none = onset_distribution(Fi)
        X = load_patient_tensor(PROCESSED, test.ids[i])
        vit = {c: [None if np.isnan(v) else round(float(v), 2) for v in X[:, FEATURE_COLUMNS.index(c)]] for c in SHOW}
        patients.append({
            "id": test.ids[i], "title": title, "T": T,
            "t_sepsis": int(test.t_sepsis[i]) if test.is_septic[i] else None,
            "p1_alarm": int(ev["p1_test"][i]) if ev["p1_test"][i] < T else None,
            "p2_alarm": int(ev["p2_test"][i]) if ev["p2_test"][i] < T else None,
            "F6": [round(float(v), 4) for v in Fi[:, 5]], "margin": [round(float(v), 4) for v in margin(Fi)],
            "p": [[round(float(v), 4) for v in row] for row in p], "p_none": [round(float(v), 4) for v in p_none],
            "utility_p1": round(float(o1[i]), 3), "utility_p2": round(float(o2[i]), 3),
            "vitals": vit,
        })
    payload = {"tau": meta["tau_p1"], "K": int(F_all.shape[1]), "vitals": SHOW, "patients": patients}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay.html").write_text(HTML.replace("__DATA__", json.dumps(payload)))
    print("wrote", OUT / "replay.html", "with", len(patients), "stays")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sepsis alarm replay</title>
<style>
:root{--bg:#F6F8F7;--panel:#fff;--ink:#1C2521;--soft:#5B6862;--line:#D7DED9;--teal:#0E6B6E;--amber:#C9862E;--rose:#A4453F}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Georgia,'Times New Roman',serif}
.banner{background:var(--rose);color:#fff;padding:10px 20px;font:600 13px/1.4 system-ui,sans-serif;letter-spacing:.02em}
main{max-width:1100px;margin:0 auto;padding:18px 20px 40px}
h1{font-size:22px;margin:6px 0 2px}p.sub{color:var(--soft);margin:0 0 14px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0 14px;font-family:system-ui,sans-serif;font-size:13px}
select,button{font:inherit;padding:6px 10px;border:1px solid var(--line);background:#fff;border-radius:4px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
input[type=range]{flex:1;min-width:200px}
.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:14px}@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.card h2{font:600 12px system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:var(--soft);margin:0 0 6px}
canvas{display:block}
.grid>*{min-width:0}.card{overflow:hidden}
.status{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.pill{flex:1;min-width:230px;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 14px;font-family:system-ui,sans-serif}
.pill b{display:block;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--soft)}
.pill span{font-size:18px;font-weight:600}.on{border-color:var(--rose);background:#F8EAE8}.on span{color:var(--rose)}
.note{font:12.5px/1.5 system-ui,sans-serif;color:var(--soft);margin-top:12px}
</style></head><body>
<div class="banner">REPLAY of historical, de-identified ICU stays from the PhysioNet/CinC 2019 test split. This is not a live clinical system and makes no claim about deployment.</div>
<main>
<h1>Sepsis alarm replay</h1>
<p class="sub">One stay at a time, hour by hour. Only data up to the current hour is shown. P1 alarms when F[t,6] crosses the validation-tuned threshold; P2 alarms when waiting no longer raises expected utility. Both read the same trained model.</p>
<div class="bar"><label>Stay <select id="sel"></select></label><button id="play">Play</button><button id="step">Step</button><button id="reset">Reset</button>
<label>Speed <select id="speed"><option value="900">slow</option><option value="450" selected>normal</option><option value="150">fast</option></select></label>
<input id="slider" type="range" min="0" value="0"><span id="hr"></span></div>
<div class="status"><div class="pill" id="s1"><b>P1 threshold rule</b><span id="s1t">watching</span></div><div class="pill" id="s2"><b>P2 expected-utility rule</b><span id="s2t">watching</span></div><div class="pill" id="so"><b>Ground truth (revealed in hindsight)</b><span id="sot">onset not yet reached</span></div></div>
<div class="grid"><div class="card"><h2>Vital signs (dots = measured, line = last known value)</h2><canvas id="vit" height="360"></canvas></div>
<div><div class="card" style="margin-bottom:14px"><h2>Belief about onset timing at this hour</h2><canvas id="bel" height="170"></canvas></div>
<div class="card"><h2>P1 score F[t,6] and P2 margin</h2><canvas id="risk" height="170"></canvas></div></div></div>
<p class="note" id="info"></p>
</main>
<script>
const D=__DATA__;let cur=0,t=0,timer=null;
const $=id=>document.getElementById(id);
const sel=$('sel');D.patients.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=p.id+' - '+p.title;sel.appendChild(o)});
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function fit(c){const w=Math.max(200,c.parentElement.clientWidth-28),h=+(c.dataset.h||(c.dataset.h=c.getAttribute('height'))),d=window.devicePixelRatio||1;c.width=w*d;c.height=h*d;c.style.width=w+'px';c.style.height=h+'px';const g=c.getContext('2d');g.setTransform(d,0,0,d,0,0);return[g,w,h]}
function P(){return D.patients[cur]}
function vline(g,x,y0,y1,col,dash,w){g.strokeStyle=col;g.lineWidth=w||1.5;g.setLineDash(dash||[]);g.beginPath();g.moveTo(x,y0);g.lineTo(x,y1);g.stroke();g.setLineDash([])}
function drawVitals(){const[g,W,H]=fit($('vit'));const p=P(),n=D.vitals.length,rh=H/n,L=44,R=W-8;g.font='11px system-ui';
 D.vitals.forEach((v,k)=>{const y0=k*rh+6,y1=(k+1)*rh-14,vals=p.vitals[v];const obs=vals.filter(x=>x!==null);if(!obs.length){g.fillStyle=css('--soft');g.fillText(v+': never measured',L,y0+12);return}
 let lo=Math.min(...obs),hi=Math.max(...obs);if(hi-lo<1e-6){hi=lo+1}const pad=(hi-lo)*.1;lo-=pad;hi+=pad;
 const X=i=>L+(R-L)*i/(p.T-1||1),Y=x=>y1-(y1-y0)*(x-lo)/(hi-lo);
 g.fillStyle=css('--soft');g.fillText(v,4,y0+12);g.fillText(hi.toFixed(0),4,y0+26);g.fillText(lo.toFixed(0),4,y1);
 g.strokeStyle=css('--line');g.strokeRect(L,y0,R-L,y1-y0);
 let last=null;g.strokeStyle=css('--teal');g.lineWidth=1.3;g.beginPath();let started=false;
 for(let i=0;i<=t;i++){if(vals[i]!==null)last=vals[i];if(last===null)continue;const x=X(i),y=Y(last);if(!started){g.moveTo(x,y);started=true}else g.lineTo(x,y)}g.stroke();
 g.fillStyle=css('--teal');for(let i=0;i<=t;i++)if(vals[i]!==null){g.beginPath();g.arc(X(i),Y(vals[i]),2.2,0,7);g.fill()}
 vline(g,X(t),y0,y1,css('--ink'),[2,2],1);
 if(p.p1_alarm!==null&&t>=p.p1_alarm)vline(g,X(p.p1_alarm),y0,y1,css('--amber'),[],2);
 if(p.p2_alarm!==null&&t>=p.p2_alarm)vline(g,X(p.p2_alarm),y0,y1,css('--rose'),[5,3],2);
 if(p.t_sepsis!==null&&t>=p.t_sepsis)vline(g,X(p.t_sepsis),y0,y1,css('--ink'),[],2)});
 g.fillStyle=css('--soft');g.fillText('hour 0',L,H-1);g.fillText('hour '+(p.T-1),R-46,H-1)}
function drawBelief(){const[g,W,H]=fit($('bel'));const p=P(),K=D.K,pk=p.p[t].concat([p.p_none[t]]),bw=(W-30)/(K+1);let mx=Math.max(...pk,.05);g.font='11px system-ui';
 pk.forEach((v,i)=>{const h=(H-34)*v/mx,x=24+i*bw;g.fillStyle=i===K?css('--soft'):css('--teal');g.fillRect(x+2,H-20-h,bw-4,h);g.fillStyle=css('--soft');g.fillText(i===K?'none':String(i+1),x+bw/2-6,H-6)});
 g.fillStyle=css('--soft');g.fillText('P(onset in k hours from now)  max '+mx.toFixed(2),24,12)}
function drawRisk(){const[g,W,H]=fit($('risk'));const p=P(),L=34,R=W-8,y0=8,y1=H-18;g.font='11px system-ui';const X=i=>L+(R-L)*i/(p.T-1||1);
 const mg=p.margin.slice(0,t+1),f6=p.F6.slice(0,t+1);const top=Math.max(.1,D.tau*2,...f6);const Yf=v=>y1-(y1-y0)*v/top;
 g.strokeStyle=css('--line');g.strokeRect(L,y0,R-L,y1-y0);g.fillStyle=css('--soft');g.fillText(top.toFixed(2),2,y0+8);g.fillText('0',20,y1);
 g.strokeStyle=css('--amber');g.setLineDash([4,3]);g.beginPath();g.moveTo(L,Yf(D.tau));g.lineTo(R,Yf(D.tau));g.stroke();g.setLineDash([]);g.fillStyle=css('--amber');g.fillText('tau '+D.tau.toFixed(3),R-70,Yf(D.tau)-4);
 g.strokeStyle=css('--teal');g.lineWidth=1.5;g.beginPath();f6.forEach((v,i)=>{i?g.lineTo(X(i),Yf(v)):g.moveTo(X(i),Yf(v))});g.stroke();
 g.fillStyle=css('--teal');g.fillText('F[t,6]',L+4,y0+12);
 g.fillStyle=css('--rose');g.fillText('P2 margin now: '+mg[t].toFixed(3)+(mg[t]>=-1e-12?'  (alarm)':'  (wait)'),L+60,y0+12);
 vline(g,X(t),y0,y1,css('--ink'),[2,2],1)}
function status(){const p=P();const a1=p.p1_alarm!==null&&t>=p.p1_alarm,a2=p.p2_alarm!==null&&t>=p.p2_alarm;
 $('s1').className='pill'+(a1?' on':'');$('s2').className='pill'+(a2?' on':'');
 $('s1t').textContent=a1?'ALARM at hour '+p.p1_alarm:'watching';$('s2t').textContent=a2?'ALARM at hour '+p.p2_alarm:'watching';
 const rev=p.t_sepsis!==null&&t>=p.t_sepsis;$('sot').textContent=p.t_sepsis===null?'this patient never became septic':(rev?'onset was hour '+p.t_sepsis:'onset not yet reached');
 $('hr').textContent='hour '+t+' of '+(p.T-1);$('slider').max=p.T-1;$('slider').value=t;
 $('info').textContent='Final offline result for this stay: P1 alarm '+(p.p1_alarm===null?'never':'hour '+p.p1_alarm)+', P2 alarm '+(p.p2_alarm===null?'never':'hour '+p.p2_alarm)+'. Normalised-utility numerator for this stay: P1 '+p.utility_p1+', P2 '+p.utility_p2+'. Black solid line = true onset (shown only once reached), amber = P1 alarm, dashed rose = P2 alarm.'}
function draw(){status();drawVitals();drawBelief();drawRisk()}
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play'}
function advance(){if(t<P().T-1){t++;draw()}else stop()}
$('play').onclick=()=>{if(timer){stop();return}if(t>=P().T-1)t=0;$('play').textContent='Pause';timer=setInterval(advance,+$('speed').value);draw()};
$('speed').onchange=()=>{if(timer){stop();$('play').click()}};
$('step').onclick=()=>{stop();advance()};$('reset').onclick=()=>{stop();t=0;draw()};
$('slider').oninput=e=>{stop();t=+e.target.value;draw()};
sel.onchange=()=>{stop();cur=+sel.value;t=0;draw()};window.onresize=draw;draw();
</script></body></html>"""


if __name__ == "__main__":
    build()

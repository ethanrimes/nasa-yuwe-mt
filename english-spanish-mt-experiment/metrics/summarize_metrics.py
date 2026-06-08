import json, os
tiers = [10000, 50000, 100000, 500000, 1000000]
for t in tiers:
    p = f'metrics/T{t}_trainer_state.json'
    if not os.path.exists(p): continue
    d = json.load(open(p))
    h = d['log_history']
    train = [(x.get('epoch',0), x.get('step',0), x['loss']) for x in h if 'loss' in x and 'eval_loss' not in x]
    eval_ = [(x.get('epoch',0), x.get('step',0), x['eval_loss']) for x in h if 'eval_loss' in x and 'eval_avg_bleu' not in x]
    bleu  = [(x.get('epoch',0), x.get('step',0), x.get('eval_avg_bleu'), x.get('eval_en2es_bleu'), x.get('eval_es2en_bleu')) for x in h if 'eval_avg_bleu' in x]
    last = h[-1] if h else {}
    fe = last.get('epoch','?'); fs = last.get('step','?')
    print(f'=== T{t} (final ep {fe}, {fs} steps) ===')
    if train:
        sample = train[::max(1,len(train)//6)] + [train[-1]]
        seen=set()
        for e,s,l in sample:
            if s in seen: continue
            seen.add(s)
            print(f'  train  ep {e:5.2f} step {s:6d}: loss={l:.4f}')
    if eval_:
        for e,s,l in eval_:
            print(f'  eval   ep {e:5.2f} step {s:6d}: eval_loss={l:.4f}')
    if bleu:
        for e,s,avg,en2,es2 in bleu:
            if avg is None: continue
            print(f'  BLEU   ep {e:5.2f} step {s:6d}: avg={avg:5.2f}  en2es={en2:5.2f}  es2en={es2:5.2f}')
        valid = [b for b in bleu if b[2] is not None]
        if valid:
            best = max(valid, key=lambda x: x[2])
            last_step = bleu[-1][1]
            pct = best[1]/last_step*100 if last_step else 0
            print(f'  >> BEST avg BLEU={best[2]:.2f} @ ep {best[0]:.2f} step {best[1]} ({pct:.0f}% through training)')
    print()

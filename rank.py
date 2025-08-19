#!/usr/bin/env python3
"""
pair_rank_tool.py

A Flask-based tool for efficient pairwise ranking of images with TrueSkill and network inference.
Users compare image pairs, TrueSkill updates μ/σ, virtual matches propagate via transitive chains,
with live μ–σ scatter and comparison network graph, virtual thumbnails, undo, and full export/resets.

Dependencies:
    pip install Flask colorlog matplotlib trueskill networkx
"""
import os
import math
import json
import threading
import webbrowser
from pathlib import Path
import statistics
import random
from collections import Counter

import trueskill
import networkx as nx
from flask import Flask, redirect, render_template_string, request, url_for, send_file, session
import logging
import matplotlib.pyplot as plt

# ---------- Transitive Closure Helper ----------

def compute_tc(G):
    reach = {u: set() for u in G.nodes()}
    for u in G.nodes():
        for v in nx.descendants(G, u):
            reach[u].add(v)
    TC = nx.DiGraph()
    TC.add_nodes_from(G.nodes())
    for u, targets in reach.items():
        for v in targets:
            TC.add_edge(u, v)
    return TC

# Logging setup
try:
    import colorlog
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s',
        log_colors={'DEBUG':'cyan','INFO':'green','WARNING':'yellow','ERROR':'red'}
    ))
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)

# TrueSkill environment
env = trueskill.TrueSkill(draw_probability=0)
INITIAL_MU = env.mu
INITIAL_SIGMA = env.sigma

app = Flask(__name__)
app.secret_key = 'replace-with-random-secret'
STATE_FILE = 'pair_rank_state.json'
VALID_IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.gif'}

# ---------- Persistence & Refresh ----------

def load_state(json_dir):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        logger.info('Loaded state from %s', STATE_FILE)
    else:
        state = {'items': [], 'comparisons': []}
    app.config['json_dir'] = json_dir
    refresh_items(state, json_dir)
    save_state(state)
    return state


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info('Saved state to %s', STATE_FILE)


def refresh_items(state, json_dir):
    existing = {it['path'] for it in state['items']}
    next_id = max((it['id'] for it in state['items']), default=0) + 1
    for jf in sorted(Path(json_dir).glob('*.json')):
        with open(jf, 'r') as f:
            data = json.load(f)
        group = f"Model: {data.get('model','')} | Prompt: {data.get('user_prompt','')} | LORA: {data.get('lora','') or 'None'}"
        for cand in data.get('candidates', []):
            img = cand.get('file')
            if img and Path(img).suffix.lower() in VALID_IMAGE_EXT and img not in existing:
                state['items'].append({
                    'id': next_id,
                    'path': img,
                    'group': group,
                    'meta_file': str(jf),
                    'mu': INITIAL_MU,
                    'sigma': INITIAL_SIGMA
                })
                existing.add(img)
                next_id += 1

# ---------- TrueSkill Helpers ----------

def win_probability(mu1, sigma1, mu2, sigma2):
    beta = env.beta
    denom = (2*beta*beta + sigma1*sigma1 + sigma2*sigma2) ** 0.5
    return env.cdf((mu1 - mu2) / denom)


def update_ratings(winner, loser, state):
    w = next(it for it in state['items'] if it['id']==winner)
    l = next(it for it in state['items'] if it['id']==loser)
    r_w = trueskill.Rating(mu=w['mu'], sigma=w['sigma'])
    r_l = trueskill.Rating(mu=l['mu'], sigma=l['sigma'])
    new_w, new_l = env.rate_1vs1(r_w, r_l)
    w['mu'], w['sigma'] = new_w.mu, new_w.sigma
    l['mu'], l['sigma'] = new_l.mu, new_l.sigma

# ---------- Virtual Matches ----------

def virtual_matches(winner, loser, state):
    items = {it['id']: it for it in state['items']}
    G = nx.DiGraph()
    G.add_nodes_from(items)
    for comp in state['comparisons']:
        G.add_edge(comp['winner'], comp['loser'])
    try:
        TC = nx.transitive_closure_dag(G)
    except nx.NetworkXUnfeasible:
        TC = compute_tc(G)
    virtuals = []
    for c in items:
        if c != winner and TC.has_edge(loser, c):
            p = win_probability(items[winner]['mu'], items[winner]['sigma'], items[c]['mu'], items[c]['sigma'])
            if random.random() < p:
                state['comparisons'].append({'winner': winner, 'loser': c, 'virtual': True})
                update_ratings(winner, c, state)
                virtuals.append((winner, c))
    for d in items:
        if d != loser and TC.has_edge(d, winner):
            p = win_probability(items[d]['mu'], items[d]['sigma'], items[loser]['mu'], items[loser]['sigma'])
            if random.random() < p:
                state['comparisons'].append({'winner': d, 'loser': loser, 'virtual': True})
                update_ratings(d, loser, state)
                virtuals.append((d, loser))
    return virtuals

# ---------- Pair Selection ----------

def group_items(state, group):
    return {it['id']: it for it in state['items'] if it['group'] == group}


def next_pair(state, group):
    items = group_items(state, group)
    G = nx.DiGraph()
    G.add_nodes_from(items)
    for comp in state['comparisons']:
        if comp['winner'] in items and comp['loser'] in items:
            G.add_edge(comp['winner'], comp['loser'])
    try:
        TC = nx.transitive_closure_dag(G)
    except nx.NetworkXUnfeasible:
        TC = compute_tc(G)
    sorted_ids = sorted(items, key=lambda i: items[i]['sigma'], reverse=True)
    for i in range(len(sorted_ids)):
        for j in range(i+1, len(sorted_ids)):
            u, v = sorted_ids[i], sorted_ids[j]
            if not TC.has_edge(u, v) and not TC.has_edge(v, u):
                return u, v
    return None, None

# ---------- Templates ----------
HOME_TMPL = '''
<h1>Select Group</h1>
<ul>
  {% for g in groups %}
    <li><a href="{{ url_for('rank_pair', group=g) }}">{{ g }}</a></li>
  {% endfor %}
</ul>
'''

RANK_TMPL = '''
<h2>Group: {{ group }}</h2>
{% if virtuals %}
  <div style="margin-bottom:10px;">
    {% for a,b in virtuals %}
      <img src="/img/{{ a }}" height="50">&gt;<img src="/img/{{ b }}" height="50">
    {% endfor %}
  </div>
{% endif %}
<p>Min σ={{ min_s|round(2) }}, Max σ={{ max_s|round(2) }}, Var μ={{ var_mu|round(2) }}</p>
<p>Images with no votes yet: {{ needing }}</p>
<p>
  <a href="{{ url_for('undo', group=group) }}">Undo</a> | 
  <a href="{{ url_for('results', group=group) }}">Results</a> | 
  <a href="{{ url_for('show_graph', group=group) }}">Graph</a>
</p>
<div style="display:flex;gap:20px;">
  <form id="left-form" method="post" action="{{ url_for('vote', group=group) }}">
    <input type="hidden" name="winner" value="{{ u }}">
    <input type="hidden" name="loser" value="{{ v }}">
    <button type="submit"><img src="/img/{{ u }}" height="300"></button>
  </form>
  <form id="right-form" method="post" action="{{ url_for('vote', group=group) }}">
    <input type="hidden" name="winner" value="{{ v }}">
    <input type="hidden" name="loser" value="{{ u }}">
    <button type="submit"><img src="/img/{{ v }}" height="300"></button>
  </form>
</div>
<input id="focus" style="position:absolute;opacity:0;"><script>
  document.getElementById('focus').focus();
  document.addEventListener('keydown', function(e) {
    if (e.key === '1') document.getElementById('left-form').submit();
    if (e.key === '2') document.getElementById('right-form').submit();
  });
</script>
'''

RESULTS_TMPL = '''
<h2>TrueSkill Ranking: {{ group }}</h2>
<p>Min σ={{ min_s|round(2) }}, Max σ={{ max_s|round(2) }}, Var μ={{ var_mu|round(2) }}</p>
<ol>
{% for it in order %}
  <li>
    <a href="{{ url_for('reset', item_id=it.id, group=group) }}">
      <img src="/img/{{ it.id }}" height="80">
    </a>
    {{ it.path }} (μ={{ it.mu|round(1) }}, σ={{ it.sigma|round(1) }})
  </li>
{% endfor %}
</ol>
<p>
  <a href="{{ url_for('export', group=group) }}">Export</a> | 
  <a href="{{ url_for('rank_pair', group=group) }}">Continue</a>
</p>
'''

GRAPH_TMPL = '''
<h2>TrueSkill & Comparison Graph: {{ group }}</h2>
<img src="/{{ graph_path }}" style="max-width:100%;">
<p><a href="{{ url_for('rank_pair', group=group) }}">Back</a></p>
'''

# ---------- Routes ----------
@app.route('/')
def home():
    state = app.config['state']
    groups = sorted({it['group'] for it in state['items']})
    return render_template_string(HOME_TMPL, groups=groups)

@app.route('/rank')
def rank_pair():
    state = app.config['state']
    refresh_items(state, app.config['json_dir'])
    save_state(state)
    group = request.args.get('group') or sorted({i['group'] for i in state['items']})[0]
    items = group_items(state, group)
    mus = [it['mu'] for it in items.values()]
    sigs = [it['sigma'] for it in items.values()]
    min_s, max_s = min(sigs), max(sigs)
    var_mu = statistics.pvariance(mus)
    votes = Counter()
    for c in state['comparisons']:
        votes[c['winner']] += 1
        votes[c['loser']] += 1
    needing = sum(1 for i in items if votes[i] == 0)
    u, v = next_pair(state, group)
    virtuals_list = session.pop('virtuals', [])
    if u is None:
        return redirect(url_for('results', group=group))
    return render_template_string(RANK_TMPL, group=group, min_s=min_s, max_s=max_s,
                                  var_mu=var_mu, needing=needing, u=u, v=v,
                                  virtuals=virtuals_list)

@app.route('/vote', methods=['POST'])
def vote():
    state = app.config['state']
    w, l = int(request.form['winner']), int(request.form['loser'])
    state['comparisons'].append({'winner': w, 'loser': l, 'virtual': False})
    update_ratings(w, l, state)
    virt = virtual_matches(w, l, state)
    save_state(state)
    session['virtuals'] = virt
    return redirect(url_for('rank_pair', group=request.args.get('group')))

@app.route('/undo')
def undo():
    state = app.config['state']
    if state['comparisons']:
        state['comparisons'].pop()
        save_state(state)
    return redirect(url_for('rank_pair', group=request.args.get('group')))

@app.route('/img/<int:item_id>')
def serve_img(item_id):
    path = next(it['path'] for it in app.config['state']['items'] if it['id'] == item_id)
    return send_file(path)

@app.route('/results')
def results():
    state = app.config['state']
    group = request.args.get('group')
    mus = [it['mu'] for it in state['items'] if it['group'] == group]
    sigs = [it['sigma'] for it in state['items'] if it['group'] == group]
    min_s, max_s = min(sigs), max(sigs)
    var_mu = statistics.pvariance(mus)
    order = sorted([it for it in state['items'] if it['group'] == group],
                   key=lambda it: it['mu'], reverse=True)
    return render_template_string(RESULTS_TMPL, group=group, min_s=min_s,
                                  max_s=max_s, var_mu=var_mu, order=order)

@app.route('/reset/<int:item_id>')
def reset(item_id):
    state = app.config['state']
    for it in state['items']:
        if it['id'] == item_id:
            it['mu'], it['sigma'] = INITIAL_MU, INITIAL_SIGMA
    state['comparisons'] = [c for c in state['comparisons']
                             if c['winner'] != item_id and c['loser'] != item_id]
    save_state(state)
    return redirect(url_for('rank_pair', group=request.args.get('group')))

@app.route('/show_graph')
def show_graph():
    state = app.config['state']
    group = request.args.get('group')
    items = group_items(state, group)
    ids   = list(items.keys())
    sigs  = [items[i]['sigma'] for i in ids]
    mus   = [items[i]['mu']    for i in ids]

    # 1) Build rank_map (1…N) and seed positions
    ranked   = sorted(items.items(), key=lambda kv: kv[1]['mu'], reverse=True)
    rank_map = {item_id: idx+1 for idx, (item_id, _) in enumerate(ranked)}
    # seed at (mu, sigma)
    

    # 2) Build comparison graph and edge weights (inverse pooled z-score)
    G = nx.DiGraph()
    G.add_nodes_from(ids)
    for c in state['comparisons']:
        if c['winner'] in ids and c['loser'] in ids:
            G.add_edge(c['winner'], c['loser'])

    eps = 1e-6
    for u, v in G.edges():
        μ1, σ1 = items[u]['mu'], items[u]['sigma']
        μ2, σ2 = items[v]['mu'], items[v]['sigma']
        avgσ    = (σ1 + σ2) / 2
        Δμ      = abs(μ1 - μ2)
        raw_w   = avgσ / (Δμ + eps)
        G[u][v]['weight'] = raw_w

    # Normalize weights into [0,1]
    raw_vals = [G[u][v]['weight'] for u, v in G.edges()]
    max_w    = max(raw_vals) if raw_vals else 1
    for u, v in G.edges():
        G[u][v]['weight_norm'] = G[u][v]['weight'] / max_w

    # 3) Compute sizing
    scale_node = 500
    scale_edge = 4
    node_sizes  = [ (1/items[n]['sigma']) * scale_node  for n in ids ]
    edge_widths = [ G[u][v]['weight_norm'] * scale_edge for u, v in G.edges() ]

    seed_pos = {n: (items[n]['sigma'], items[n]['mu']) for n in G.nodes()}

    pos = nx.fruchterman_reingold_layout(
        G,
        weight='weight_norm',
        pos=seed_pos,
        # fixed=list(G.nodes()),      # keep nodes roughly at seed
        k=1/math.sqrt(len(G)),      # optimal distance
        iterations=50
    )

    # 5) Plot both panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # TrueSkill scatter
    ax1.scatter(sigs, mus)
    for idx, it_id in enumerate(ids):
        ax1.annotate(str(rank_map[it_id]), (sigs[idx], mus[idx]))
    ax1.set(xlabel='Sigma', ylabel='Mu', title='TrueSkill')

    # Comparison graph
    nx.draw(
        G,
        pos,
        ax=ax2,
        labels=rank_map,
        with_labels=True,
        node_size=node_sizes,
        node_color=mus,            # numeric → colormapped
        cmap=plt.cm.viridis,
        width=edge_widths,
        arrows=True
    )
    ax2.set_title('Comparison Graph')

    # 6) Save and render
    os.makedirs('static', exist_ok=True)
    graph_path = f'static/graph_{group}.png'
    fig.savefig(graph_path, bbox_inches='tight')
    plt.close(fig)
    return render_template_string(GRAPH_TMPL, group=group, graph_path=graph_path)


@app.route('/export')
def export():
    state = app.config['state']
    group = request.args.get('group')
    order = sorted([it for it in state['items'] if it['group']==group], key=lambda it: it['mu'], reverse=True)
    fmap = {it['id']: it['meta_file'] for it in state['items'] if it['group']==group}
    for jf in set(fmap.values()):
        with open(jf, 'r') as f:
            data = json.load(f)
        data['ranked_candidates'] = [it['id'] for it in order if fmap[it['id']]==jf]
        with open(jf, 'w') as f:
            json.dump(data, f, indent=2)
    return redirect(url_for('results', group=group))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    state = load_state(args.json_dir)
    app.config['state'] = state
    url = f"http://{args.host}:{args.port}/"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port)

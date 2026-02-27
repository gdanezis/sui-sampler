#!/usr/bin/env python3
"""
Dataflow Graph Visualization for Sui PTBs.

Reads sui-sampler JSON from stdin and outputs an interactive HTML force-directed
graph showing how MoveCall functions connect across transactions — which functions
feed data to which other functions, and how often.

Usage:
    ./target/release/sui-sampler --sample-count 50 | python3 examples/dataflow_graph.py
    open dataflow_graph.html
"""

import json
import sys
from collections import defaultdict


def extract_programmable_transaction(transaction):
    """Extract ProgrammableTransaction from transaction data."""
    if 'data' in transaction and isinstance(transaction['data'], list):
        for data_item in transaction['data']:
            if 'intent_message' in data_item:
                intent_message = data_item['intent_message']
                if 'value' in intent_message:
                    value = intent_message['value']
                    if 'V1' in value and 'kind' in value['V1']:
                        kind = value['V1']['kind']
                        if 'ProgrammableTransaction' in kind:
                            return kind['ProgrammableTransaction']
    return None


def extract_arg_refs(arg):
    """Extract argument references from a command argument.

    Returns a list of (type, index) or (type, index, sub_index) tuples.
    Types: 'Input', 'Result', 'NestedResult'.
    """
    if isinstance(arg, dict):
        if 'Input' in arg:
            return [('Input', arg['Input'])]
        if 'Result' in arg:
            return [('Result', arg['Result'])]
        if 'NestedResult' in arg:
            return [('NestedResult', arg['NestedResult'][0], arg['NestedResult'][1])]
    # "GasCoin" or other literals — ignore
    return []


def get_command_args(command):
    """Extract all argument references consumed by a command."""
    refs = []
    if 'MoveCall' in command:
        for arg in command['MoveCall'].get('arguments', []):
            refs.extend(extract_arg_refs(arg))
    elif 'SplitCoins' in command:
        coin_ref, amount_refs = command['SplitCoins']
        refs.extend(extract_arg_refs(coin_ref))
        for a in amount_refs:
            refs.extend(extract_arg_refs(a))
    elif 'MergeCoins' in command:
        target_ref, source_refs = command['MergeCoins']
        refs.extend(extract_arg_refs(target_ref))
        for s in source_refs:
            refs.extend(extract_arg_refs(s))
    elif 'TransferObjects' in command:
        obj_refs, recipient_ref = command['TransferObjects']
        for o in obj_refs:
            refs.extend(extract_arg_refs(o))
        refs.extend(extract_arg_refs(recipient_ref))
    elif 'MakeMoveVec' in command:
        _type_tag, element_refs = command['MakeMoveVec']
        for e in element_refs:
            refs.extend(extract_arg_refs(e))
    return refs


# Sui framework package prefixes to exclude from graph nodes
SYSTEM_PACKAGES = (
    '0x0000000000000000000000000000000000000000000000000000000000000001',
    '0x0000000000000000000000000000000000000000000000000000000000000002',
    '0x0000000000000000000000000000000000000000000000000000000000000003',
)


def get_move_call_label(command):
    """Return 'module::function' label and package ID if command is a MoveCall, else None.

    Returns (None, None) for system packages (0x1, 0x2, 0x3).
    """
    if 'MoveCall' not in command:
        return None, None
    mc = command['MoveCall']
    package = mc.get('package', '')
    module = mc.get('module', '')
    function = mc.get('function', '')
    if not package.startswith('0x'):
        package = f'0x{package}'
    if package in SYSTEM_PACKAGES:
        return None, None
    label = f"{module}::{function}"
    return label, package


def trace_origins(cmd_index, commands, cmd_args_cache, visited=None):
    """Trace through non-MoveCall commands to find originating MoveCall indices.

    Returns a set of command indices that are MoveCall producers.
    """
    if visited is None:
        visited = set()
    if cmd_index in visited:
        return set()
    visited.add(cmd_index)

    origins = set()
    args = cmd_args_cache[cmd_index]
    for ref in args:
        if ref[0] == 'Input':
            continue  # Inputs handled separately
        producer_idx = ref[1]
        if producer_idx < 0 or producer_idx >= len(commands):
            continue
        label, _ = get_move_call_label(commands[producer_idx])
        if label is not None:
            origins.add(producer_idx)
        else:
            # Builtin — trace transitively
            origins.update(trace_origins(producer_idx, commands, cmd_args_cache, visited))
    return origins


def process_transaction(commands, edge_counts, node_counts, node_packages):
    """Process a single transaction's commands to extract dataflow edges."""
    # Pre-cache args for each command
    cmd_args_cache = {}
    for i, cmd in enumerate(commands):
        cmd_args_cache[i] = get_command_args(cmd)

    # Identify MoveCall commands
    move_call_indices = []
    for i, cmd in enumerate(commands):
        label, pkg = get_move_call_label(cmd)
        if label is not None:
            move_call_indices.append(i)
            node_counts[label] += 1
            node_packages[label] = pkg

    # Build input consumer map: input_index -> [MoveCall cmd indices that consume it]
    input_consumers = defaultdict(list)
    for i in move_call_indices:
        for ref in cmd_args_cache[i]:
            if ref[0] == 'Input':
                input_consumers[ref[1]].append(i)

    # For each MoveCall, trace Result/NestedResult edges
    for i in move_call_indices:
        label_b, _ = get_move_call_label(commands[i])
        for ref in cmd_args_cache[i]:
            if ref[0] in ('Result', 'NestedResult'):
                producer_idx = ref[1]
                if producer_idx >= len(commands):
                    continue
                label_a, _ = get_move_call_label(commands[producer_idx])
                if label_a is not None:
                    # Direct MoveCall -> MoveCall
                    if label_a != label_b:
                        edge_counts[(label_a, label_b)] += 1
                else:
                    # Trace through builtin
                    origins = trace_origins(producer_idx, commands, cmd_args_cache)
                    for origin_idx in origins:
                        label_a, _ = get_move_call_label(commands[origin_idx])
                        if label_a is not None and label_a != label_b:
                            edge_counts[(label_a, label_b)] += 1

    # Shared-input edges: for each input consumed by multiple MoveCalls
    for input_idx, consumers in input_consumers.items():
        if len(consumers) < 2:
            continue
        for ci in range(len(consumers)):
            for cj in range(ci + 1, len(consumers)):
                idx_a, idx_b = consumers[ci], consumers[cj]
                label_a, _ = get_move_call_label(commands[idx_a])
                label_b, _ = get_move_call_label(commands[idx_b])
                if label_a and label_b and label_a != label_b:
                    edge_counts[(label_a, label_b)] += 1


def extract_checkpoint_timestamp(checkpoint):
    """Extract timestamp_ms from checkpoint_summary.data."""
    try:
        return checkpoint['checkpoint_summary']['data']['timestamp_ms']
    except (KeyError, TypeError):
        return None


def analyze_data(data):
    """Analyze all transactions and return edge counts, node counts, package map, and metadata."""
    edge_counts = defaultdict(int)
    node_counts = defaultdict(int)
    node_packages = {}
    pkg_tx_counts = defaultdict(int)  # package -> number of transactions it appears in

    checkpoints = data.get('checkpoints', [])
    tx_count = 0
    timestamps = []

    for checkpoint in checkpoints:
        ts = extract_checkpoint_timestamp(checkpoint)
        if ts is not None:
            timestamps.append(ts)
        for tx_data in checkpoint.get('transactions', []):
            transaction = tx_data.get('transaction', {})
            ptx = extract_programmable_transaction(transaction)
            if ptx is None:
                continue
            commands = ptx.get('commands', [])
            if not commands:
                continue
            tx_count += 1
            # Track which packages appear in this transaction
            tx_packages = set()
            for cmd in commands:
                _, pkg = get_move_call_label(cmd)
                if pkg is not None:
                    tx_packages.add(pkg)
            for pkg in tx_packages:
                pkg_tx_counts[pkg] += 1
            process_transaction(commands, edge_counts, node_counts, node_packages)

    ts_min = min(timestamps) if timestamps else None
    ts_max = max(timestamps) if timestamps else None
    meta = {
        "tx_count": tx_count,
        "checkpoint_count": len(checkpoints),
        "ts_min": ts_min,
        "ts_max": ts_max,
        "pkg_tx_counts": dict(pkg_tx_counts),
    }
    return edge_counts, node_counts, node_packages, meta


def generate_html(edge_counts, node_counts, node_packages, meta, filename="dataflow_graph.html"):
    """Generate interactive D3.js force-directed graph HTML."""
    tx_count = meta["tx_count"]

    # Build node list
    nodes = []
    node_index = {}
    for i, (label, count) in enumerate(sorted(node_counts.items(), key=lambda x: -x[1])):
        pkg = node_packages.get(label, '')
        node_index[label] = i
        nodes.append({
            "id": label,
            "count": count,
            "package": pkg,
        })

    # Build edge list (only edges where both endpoints exist)
    links = []
    for (src, tgt), count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        if src in node_index and tgt in node_index:
            links.append({
                "source": node_index[src],
                "target": node_index[tgt],
                "count": count,
            })

    # Top packages by fraction of transactions they appear in
    pkg_tx = meta["pkg_tx_counts"]
    top_packages = sorted(pkg_tx.items(), key=lambda x: -x[1])[:15]
    # Store as [package, tx_presence_count] — fraction computed in JS using tx_count

    # Format date range for display
    date_range_html = ""
    if meta["ts_min"] is not None and meta["ts_max"] is not None:
        date_range_html = f'<div class="stat date-range" id="date-range" data-ts-min="{meta["ts_min"]}" data-ts-max="{meta["ts_max"]}"></div>'

    graph_data = json.dumps({"nodes": nodes, "links": links}, separators=(',', ':'))
    top_packages_json = json.dumps(top_packages, separators=(',', ':'))
    tx_count_js = tx_count

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sui PTB Dataflow Graph</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0e17; color: #c8d6e5; overflow: hidden; }}
  #container {{ width: 100vw; height: 100vh; position: relative; }}
  svg {{ width: 100%; height: 100%; }}
  #info {{
    position: absolute; top: 16px; left: 16px;
    background: rgba(10, 14, 23, 0.92); border: 1px solid #1e2a3a;
    border-radius: 8px; padding: 14px 18px; max-width: 400px;
    font-size: 13px; line-height: 1.5; pointer-events: none;
  }}
  #info h2 {{ font-size: 16px; color: #e8ecf1; margin-bottom: 6px; }}
  #info .stat {{ color: #7f8fa6; }}
  #info .date-range {{ margin-top: 4px; font-size: 12px; color: #636e72; }}
  #legend {{
    position: absolute; bottom: 16px; left: 16px;
    background: rgba(10, 14, 23, 0.92); border: 1px solid #1e2a3a;
    border-radius: 8px; padding: 12px 16px; max-width: 380px;
    font-size: 12px; line-height: 1.6; max-height: 40vh; overflow-y: auto;
  }}
  #legend h3 {{ font-size: 13px; color: #e8ecf1; margin-bottom: 6px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .legend-item a {{ color: #7f8fa6; text-decoration: none; }}
  .legend-item a:hover {{ color: #c8d6e5; text-decoration: underline; }}
  .legend-pct {{ color: #55efc4; font-variant-numeric: tabular-nums; min-width: 42px; text-align: right; flex-shrink: 0; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }}
  #tooltip {{
    position: absolute; pointer-events: none; display: none;
    background: rgba(10, 14, 23, 0.95); border: 1px solid #2e3a4a;
    border-radius: 6px; padding: 10px 14px; font-size: 12px;
    line-height: 1.5; max-width: 420px; z-index: 10;
  }}
  #tooltip .tt-label {{ font-weight: 600; color: #e8ecf1; font-size: 13px; }}
  #tooltip .tt-pkg {{ color: #7f8fa6; font-size: 11px; word-break: break-all; }}
  #tooltip .tt-count {{ color: #55efc4; margin-top: 4px; }}
  #tooltip .tt-hint {{ color: #636e72; font-size: 10px; margin-top: 4px; }}
  .link {{ fill: none; }}
  marker {{ overflow: visible; }}
  circle {{ cursor: pointer; }}
  #filter {{
    position: absolute; top: 16px; right: 16px;
    background: rgba(10, 14, 23, 0.92); border: 1px solid #1e2a3a;
    border-radius: 8px; padding: 12px 16px; font-size: 12px;
    display: flex; flex-direction: column; gap: 6px; min-width: 180px;
  }}
  #filter label {{ color: #7f8fa6; }}
  #filter .filter-val {{ color: #55efc4; font-weight: 600; }}
  #filter input[type=range] {{
    width: 100%; accent-color: #55efc4; cursor: pointer;
  }}
  #filter .filter-stats {{ color: #636e72; font-size: 11px; }}
</style>
</head>
<body>
<div id="container">
  <svg></svg>
  <div id="info">
    <h2>Sui PTB Dataflow Graph</h2>
    <div class="stat">{len(nodes)} functions &middot; {len(links)} edges &middot; {tx_count} txns &middot; {meta["checkpoint_count"]} checkpoints</div>
    {date_range_html}
  </div>
  <div id="legend"><h3>Top Packages</h3></div>
  <div id="tooltip"></div>
  <div id="filter">
    <label>Min count: <span class="filter-val" id="filter-val">0</span></label>
    <input type="range" id="filter-slider" min="0" max="30" step="1" value="0">
    <div class="filter-stats" id="filter-stats"></div>
  </div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const graph = {graph_data};
const topPackages = {top_packages_json};
const txCount = {tx_count_js};

// Format date range from timestamps embedded in DOM
(function() {{
  const el = document.getElementById('date-range');
  if (!el) return;
  const tsMin = parseInt(el.dataset.tsMin);
  const tsMax = parseInt(el.dataset.tsMax);
  const fmt = (ts) => {{
    const d = new Date(ts);
    return d.toISOString().replace('T', ' ').replace(/\\.\\d{{3}}Z/, ' UTC');
  }};
  if (tsMin === tsMax) {{
    el.textContent = fmt(tsMin);
  }} else {{
    el.textContent = fmt(tsMin) + '  \\u2192  ' + fmt(tsMax);
  }}
}})();

// Assign maximally distinct hues to top packages via golden angle spacing
const topPkgColors = {{}};
topPackages.forEach(([pkg], i) => {{
  const hue = (i * 137.508) % 360;  // golden angle
  topPkgColors[pkg] = `hsl(${{hue}}, 90%, 55%)`;
}});

// Color: use pre-assigned palette for top packages, hash fallback for the rest
function pkgColor(pkg) {{
  if (topPkgColors[pkg]) return topPkgColors[pkg];
  let h = 0;
  for (let i = 0; i < pkg.length; i++) h = ((h << 5) - h + pkg.charCodeAt(i)) | 0;
  const hue = ((h % 360) + 360) % 360;
  return `hsl(${{hue}}, 70%, 50%)`;
}}

// Build explorer URL for a package
function explorerUrl(pkg) {{
  return `https://suiscan.xyz/mainnet/object/${{pkg}}`;
}}

// Populate legend
const legendEl = document.getElementById('legend');
topPackages.forEach(([pkg, count]) => {{
  const d = document.createElement('div');
  d.className = 'legend-item';
  const pct = ((count / txCount) * 100).toFixed(1);
  const short = pkg.length > 16 ? pkg.slice(0, 6) + '...' + pkg.slice(-4) : pkg;
  const url = explorerUrl(pkg);
  d.innerHTML = `<span class="legend-swatch" style="background:${{pkgColor(pkg)}}"></span><span class="legend-pct">${{pct}}%</span><a href="${{url}}" target="_blank" title="${{pkg}}">${{short}}</a>`;
  legendEl.appendChild(d);
}});

const svg = d3.select('svg');
const width = window.innerWidth;
const height = window.innerHeight;
svg.attr('viewBox', [0, 0, width, height]);

// Zoom — track current scale for label visibility
let currentZoom = 1;
const g = svg.append('g');
// Zoom-adjusted node radius: shrinks by 1/sqrt(zoom) so nodes reveal gaps
function zoomedRadius(d) {{
  return nodeRadius(d.count) / Math.sqrt(Math.max(currentZoom, 0.1));
}}

const zoomBehavior = d3.zoom().scaleExtent([0.1, 8]).on('zoom', (e) => {{
  g.attr('transform', e.transform);
  currentZoom = e.transform.k;
  updateLabelVisibility();
  node.attr('r', d => zoomedRadius(d));
  labels.attr('dy', d => -zoomedRadius(d) - 4);
}});
svg.call(zoomBehavior);

// Arrow marker — small and proportional to nodes
svg.append('defs').append('marker')
  .attr('id', 'arrow')
  .attr('viewBox', '0 -3 6 6')
  .attr('refX', 12)
  .attr('refY', 0)
  .attr('markerWidth', 4)
  .attr('markerHeight', 4)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,-3L6,0L0,3')
  .attr('fill', '#4a6fa5');

// Scales — use log of counts for link width/opacity so high-count edges don't dominate
const maxEdgeLog = Math.log1p(d3.max(graph.links, d => d.count) || 1);
const maxNode = d3.max(graph.nodes, d => d.count) || 1;
const linkWidth = d3.scaleLinear().domain([0, maxEdgeLog]).range([0.5, 4]).clamp(true);
const linkOpacity = d3.scaleLinear().domain([0, maxEdgeLog]).range([0.15, 0.7]).clamp(true);
const nodeRadius = d3.scaleSqrt().domain([1, maxNode]).range([2, 32]).clamp(true);
// Lightness boost: low-count nodes stay at base color, high-count nodes push toward white
const nodeLightness = d3.scaleLog().domain([1, maxNode]).range([0, 0.4]).clamp(true);

// Force simulation — run many iterations for a well-settled layout
const simulation = d3.forceSimulation(graph.nodes)
  .force('link', d3.forceLink(graph.links).distance(180).strength(0.2))
  .force('charge', d3.forceManyBody().strength(-600).distanceMax(800))
  .force('x', d3.forceX(width / 2).strength(0.003))
  .force('y', d3.forceY(height / 2).strength(0.003))
  .force('collide', d3.forceCollide().radius(d => nodeRadius(d.count) + 18).iterations(3))
  .alphaDecay(0.01)
  .velocityDecay(0.3);

// Links
const link = g.append('g')
  .selectAll('line')
  .data(graph.links)
  .join('line')
  .attr('class', 'link')
  .attr('stroke', '#4a6fa5')
  .attr('stroke-width', d => linkWidth(Math.log1p(d.count)))
  .attr('stroke-opacity', d => linkOpacity(Math.log1p(d.count)))
  .attr('marker-end', 'url(#arrow)');

// Nodes
let isDragging = false;
const node = g.append('g')
  .selectAll('circle')
  .data(graph.nodes)
  .join('circle')
  .attr('r', d => nodeRadius(d.count))
  .attr('fill', d => {{
    // Parse base hsl and boost lightness for high-count nodes
    const base = pkgColor(d.package);
    const m = base.match(/hsl\\(([\\d.]+),\\s*([\\d.]+)%,\\s*([\\d.]+)%\\)/);
    if (!m) return base;
    const h = +m[1], s = +m[2], l = +m[3];
    const boost = nodeLightness(d.count);
    // Lerp lightness toward 95 and desaturate toward white
    const nl = l + (80 - l) * boost;
    const ns = s * (1 - boost * 0.3);
    return `hsl(${{h}}, ${{ns}}%, ${{nl}}%)`;
  }})
  .attr('stroke', '#0a0e17')
  .attr('stroke-width', 1.5)
  .on('click', (e, d) => {{
    if (!isDragging) window.open(explorerUrl(d.package), '_blank');
  }})
  .call(d3.drag()
    .on('start', (e, d) => {{ isDragging = false; if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on('drag', (e, d) => {{ isDragging = true; d.fx = e.x; d.fy = e.y; }})
    .on('end', (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

// Labels — all nodes get labels; size, color, and visibility are zoom-dependent
const labels = g.append('g')
  .selectAll('text')
  .data(graph.nodes)
  .join('text')
  .text(d => d.id)
  .attr('text-anchor', 'middle')
  .attr('dy', d => -nodeRadius(d.count) - 4)
  .style('pointer-events', 'none')
  .style('text-shadow', '0 0 4px #0a0e17, 0 0 4px #0a0e17');

// Filter state — initialized here so updateLabelVisibility can reference it
let minCountFilter = 0;
const visibleNodes = new Set();
graph.nodes.forEach(d => visibleNodes.add(d.index));  // all visible initially

// Each node gets a "rank" from 0 (top count) to 1 (lowest count).
// Its "appear zoom" is the zoom level at which it first becomes visible.
const sortedCounts = graph.nodes.map(d => d.count).sort((a, b) => b - a);
const rankOf = (count) => {{
  const idx = sortedCounts.indexOf(count);
  return graph.nodes.length > 1 ? idx / (graph.nodes.length - 1) : 0;
}};
// Pre-assign each node its appearance zoom: top nodes appear at zoom 0.6, bottom at 3.0
graph.nodes.forEach(d => {{
  d._rank = rankOf(d.count);
  d._appearZoom = 0.6 + d._rank * 2.4;  // 0.6 .. 3.0
}});

// Target screen sizes: labels appear at 8px on screen, grow to 12px on screen
function updateLabelVisibility() {{
  const z = Math.max(currentZoom, 0.1);
  labels
    .style('display', d => (visibleNodes.has(d.index) && currentZoom >= d._appearZoom) ? null : 'none')
    .attr('font-size', d => {{
      // t goes 0→1 over 1.5x of zoom past appearance
      const t = Math.min(1, Math.max(0, (currentZoom - d._appearZoom) / 1.5));
      // Target screen px: 12 at appear, 18 fully mature
      const screenPx = 12 + t * 6;
      // Convert to SVG units by dividing by zoom
      return screenPx / z;
    }})
    .attr('fill', d => {{
      // High-count nodes (low rank) → white, low-count → grey
      const c = Math.round(100 + (1 - d._rank) * 155);
      return `rgb(${{c}},${{c}},${{c}})`;
    }});
}}
updateLabelVisibility();

// Tooltip
const tooltip = d3.select('#tooltip');
node.on('mouseover', (e, d) => {{
  tooltip.style('display', 'block')
    .html(`<div class="tt-label">${{d.id}}</div><div class="tt-pkg">${{d.package}}</div><div class="tt-count">Called ${{d.count}} time${{d.count !== 1 ? 's' : ''}}</div><div class="tt-hint">Click to open on Suiscan</div>`);
}}).on('mousemove', (e) => {{
  tooltip.style('left', (e.pageX + 14) + 'px').style('top', (e.pageY - 14) + 'px');
}}).on('mouseout', () => tooltip.style('display', 'none'));

// Filter by minimum count — also updates simulation layout
function applyFilter(minCount) {{
  minCountFilter = minCount;
  // Determine which nodes pass
  visibleNodes.clear();
  graph.nodes.forEach(d => {{ if (d.count >= minCount) visibleNodes.add(d.index); }});

  const filteredNodes = graph.nodes.filter(d => visibleNodes.has(d.index));
  const filteredLinks = graph.links.filter(d => {{
    const si = typeof d.source === 'object' ? d.source.index : d.source;
    const ti = typeof d.target === 'object' ? d.target.index : d.target;
    return d.count >= minCount && visibleNodes.has(si) && visibleNodes.has(ti);
  }});

  // Hide/show nodes and labels
  node.style('display', d => visibleNodes.has(d.index) ? null : 'none');
  labels.style('display', d => {{
    if (!visibleNodes.has(d.index)) return 'none';
    return currentZoom >= d._appearZoom ? null : 'none';
  }});
  link.style('display', d => {{
    const si = typeof d.source === 'object' ? d.source.index : d.source;
    const ti = typeof d.target === 'object' ? d.target.index : d.target;
    return (d.count >= minCount && visibleNodes.has(si) && visibleNodes.has(ti)) ? null : 'none';
  }});

  // Update simulation with filtered data and reheat
  simulation.nodes(filteredNodes);
  simulation.force('link').links(filteredLinks);
  simulation.alpha(0.5).restart();

  document.getElementById('filter-stats').textContent = `${{filteredNodes.length}} nodes, ${{filteredLinks.length}} edges`;
}}

const slider = document.getElementById('filter-slider');
const filterVal = document.getElementById('filter-val');
slider.addEventListener('input', () => {{
  const v = parseInt(slider.value);
  filterVal.textContent = v;
  applyFilter(v);
}});
applyFilter(0);

// Tick
simulation.on('tick', () => {{
  link
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y);
  node
    .attr('cx', d => d.x)
    .attr('cy', d => d.y);
  labels
    .attr('x', d => d.x)
    .attr('y', d => d.y);
}});
</script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)
    print(f"Wrote {filename} ({len(nodes)} nodes, {len(links)} edges from {tx_count} txns across {meta['checkpoint_count']} checkpoints)", file=sys.stderr)


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    edge_counts, node_counts, node_packages, meta = analyze_data(data)
    generate_html(edge_counts, node_counts, node_packages, meta)


if __name__ == '__main__':
    main()

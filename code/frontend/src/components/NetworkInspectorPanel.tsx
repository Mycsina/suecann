import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react';

// ── Types ──────────────────────────────────────────────────────────

interface GenomeData {
  node_ids: number[];
  node_types: number[];   // 0=input 1=bias 2=hidden 3=output
  node_acts: number[];    // 0=IDENTITY 1=NOT 2=THRESHOLD
  node_aggs: number[];    // 0=SUM 1=MIN(AND) 2=MAX(OR)
  conn_srcs: number[];
  conn_dsts: number[];
  conn_signs: number[];   // +1 or -1
  conn_enabled: number[]; // 0 or 1
}

export interface NetworkEval {
  beliefs: number[];
  outputs: number[];
  activations: number[];
  brain_type: string;
}

interface Props {
  genome: GenomeData;
  evalData: NetworkEval | null;
  playerName: string;
  onClose: () => void;
}

// ── Constants ──────────────────────────────────────────────────────

const INPUT_COUNT = 35;
const BIAS_ID = 35;
const OUTPUT_START = 36;
const OUTPUT_COUNT = 3;

const OUTPUT_LABELS = ['MAX_FORCE', 'EFFICIENT_WIN', 'EQUITY_BUILDER'];

// Strategic meaning of each intent (matching the styled resolver)
const OUTPUT_DESCRIPTIONS: string[] = [
  'MAX_FORCE — Elite + control. When trump-long: lead a low trump to draw opponents\' high trumps. Otherwise: play aggressively.',
  'EFFICIENT_WIN — Exactly Elite. Play the cheapest winner, else cheap cut, else cheapest legal sacrifice. The strong default.',
  'EQUITY_BUILDER — Elite + tempo. Lead shortest side-suit to build a void; duck cheap tricks (≤2 pts) when not last; preserve trump.',
];

// Full feature names matching the Rust FEATURE_NAMES array (35 features, redesigned)
const FEATURE_NAMES: string[] = [
  'Has_Led_Suit', 'Has_Trump', 'Led_Suit_Count', 'Trump_Count',
  'Hand_Point_Density', 'Am_I_Leading', 'Am_I_Last_To_Play', 'Is_Partner_Winning',
  'Trick_Point_Value', 'Has_Trick_Been_Cut', 'Partner_Void_Led', 'Partner_Void_Trump',
  'Any_Opp_Void_Led', 'Any_Opp_Void_Trump', 'Led_Suit_Ace_Played', 'Led_Suit_7_Played',
  'Trump_Ace_Played', 'Holds_Boss_Led', 'Holds_Boss_Trump', 'Can_Beat_Winner',
  'Min_Winning_Cost', 'Min_Sacrifice_Cost', 'Game_Pts_Remaining', 'Trick_Number',
  'Trumps_Remaining', 'Score_Delta', 'My_Void_Count', 'Longest_Side_Suit',
  'Shortest_Side_Suit', 'Side0_Depletion', 'Side1_Depletion', 'Side2_Depletion',
  'Points_Secured_Us', 'Known_Void_Suits_Count', 'Depleted_Suits_Count',
];

// Human-readable descriptions for each belief input (matching current 35-feature design)
const FEATURE_DESCRIPTIONS: string[] = [
  'Do I hold at least one card of the led suit? (bool)',
  'Do I hold at least one trump card? (bool)',
  'Cards I hold in the led suit ÷ 10.0 (float)',
  'Trump cards I hold ÷ 10.0 (float)',
  'Points in my hand ÷ unplayed game points (float)',
  'Am I the first player to act in this trick? (bool)',
  'Am I the last player to act in this trick (4th seat)? (bool)',
  'Is my partner currently winning the trick? (bool)',
  'Total point value of cards already played into this trick ÷ 44.0 (float)',
  'Has someone already played trump on a non-trump lead? (bool)',
  'Is my partner known to be void in the led suit? (bool)',
  'Is my partner known to be void in trump? (bool)',
  'Is either opponent known to be void in the led suit? (bool)',
  'Is either opponent known to be void in trump? (bool)',
  'Has the Ace of the led suit already been played? (bool)',
  'Has the 7 (manilha) of the led suit already been played? (bool)',
  'Has the Ace of trumps already been played? (bool)',
  'Do I hold the highest unplayed card in the led suit? (bool)',
  'Do I hold the highest unplayed card in trump? (bool)',
  'Can any of my legal cards beat the current trick winner? (bool)',
  'Points of my cheapest winning card ÷ 11.0 (0 if cannot win)',
  'Points of my cheapest legal card ÷ 11.0 (sacrifice cost)',
  'Unplayed card points remaining ÷ 120.0 (float)',
  'Current trick index ÷ 9.0 (float)',
  'Unplayed trump cards ÷ 10.0 (float)',
  '(our pts − opp pts + 120) ÷ 240.0; 0.5 = tied (float)',
  'Number of suits I am void in ÷ 3.0 (float)',
  'Max cards I hold in any non-trump, non-led suit ÷ 10.0 (float)',
  'Min cards I hold in any non-trump, non-led suit ÷ 10.0 (float)',
  'Fraction of side-suit 0 cards already played (float)',
  'Fraction of side-suit 1 cards already played (float)',
  'Fraction of side-suit 2 cards already played (float)',
  'Game points our team has already secured ÷ 120.0 (float)',
  'Suits where any player is known void ÷ 4.0 (float)',
  'Fully-depleted suits (all 10 cards played) ÷ 4.0 (float)',
];

// Short labels derived from feature names for the SVG
const INPUT_SHORT: string[] = FEATURE_NAMES.map((n) => {
  return n
    .replace(/_/g, '')
    .replace('Played', 'Out')
    .replace('Remaining', 'Left')
    .replace('Partner', 'Ptnr')
    .replace('Opponent', 'Opp')
    .replace('Power', 'Pwr')
    .replace('Point', 'Pt')
    .replace('Density', 'Dens')
    .replace('Depletion', 'Depl')
    .replace('Secured', 'Sec')
    .replace('Suits', 'Suits')
    .replace('Winning', 'Win')
    .replace('Leading', 'Lead')
    .replace('Number', 'Num')
    .slice(0, 9);
});

const ACT_NAMES = ['ID', 'NOT', 'THR'];
const AGG_NAMES = ['SUM', 'MIN', 'MAX'];

const SVG_WIDTH = 800;
const SVG_HEIGHT = 580;
const MARGIN = { top: 28, right: 95, bottom: 12, left: 12 };
const NODE_R = 7;
const OUTPUT_R = 12;
const INPUT_R = 5.5;

const MIN_PANEL_W = 420;
const MIN_PANEL_H = 340;
const DEFAULT_PANEL_W = 600;
const DEFAULT_PANEL_H = 520;

// ── Layout engine (matches Rust compile_rules::export_topology) ─────

interface Layout {
  positions: Map<number, { x: number; y: number }>;
  edges: { src: number; dst: number; sign: number; enabled: boolean }[];
  nodeMeta: Map<number, { type: number; act: number; agg: number; depth: number }>;
  maxDepth: number;
  reachable: Set<number>;
  layerGroups: Map<number, number[]>;
}

function computeLayout(genome: GenomeData): Layout {
  const { node_ids, node_types, node_acts, node_aggs, conn_srcs, conn_dsts, conn_signs, conn_enabled } = genome;

  // Node metadata lookup
  const nodeMeta = new Map<number, { type: number; act: number; agg: number; depth: number }>();
  for (let i = 0; i < node_ids.length; i++) {
    nodeMeta.set(node_ids[i], { type: node_types[i], act: node_acts[i], agg: node_aggs[i], depth: 0 });
  }

  // Build adjacency maps
  const outgoing = new Map<number, number[]>(); // src → [dst]
  const incoming = new Map<number, { src: number; sign: number; enabled: boolean }[]>();
  for (const nid of node_ids) {
    outgoing.set(nid, []);
    incoming.set(nid, []);
  }

  const edges: { src: number; dst: number; sign: number; enabled: boolean }[] = [];
  for (let i = 0; i < conn_srcs.length; i++) {
    const src = conn_srcs[i];
    const dst = conn_dsts[i];
    const sign = conn_signs[i];
    const enabled = conn_enabled[i] === 1;
    edges.push({ src, dst, sign, enabled });
    if (enabled) {
      outgoing.get(src)?.push(dst);
    }
    incoming.get(dst)?.push({ src, sign, enabled });
  }

  // ── Step 1: Get reachable nodes (BFS backwards from outputs) ──
  const reachable = new Set<number>();
  const queue: number[] = [];
  for (let o = OUTPUT_START; o < OUTPUT_START + OUTPUT_COUNT; o++) {
    if (nodeMeta.has(o)) {
      reachable.add(o);
      queue.push(o);
    }
  }

  while (queue.length > 0) {
    const curr = queue.pop()!;
    const incs = incoming.get(curr) || [];
    for (const { src, enabled } of incs) {
      if (enabled && !reachable.has(src)) {
        reachable.add(src);
        queue.push(src);
      }
    }
  }

  // ── Step 2: Topological order of reachable nodes (Kahn's algorithm) ──
  const inDegree = new Map<number, number>();
  for (const nid of node_ids) inDegree.set(nid, 0);
  for (const e of edges) {
    if (e.enabled && reachable.has(e.src) && reachable.has(e.dst)) {
      inDegree.set(e.dst, (inDegree.get(e.dst) || 0) + 1);
    }
  }

  const topoOrder: number[] = [];
  const topoQueue: number[] = [];
  for (const nid of node_ids) {
    if (reachable.has(nid) && (inDegree.get(nid) || 0) === 0) {
      topoQueue.push(nid);
    }
  }
  // Sort queue so inputs/bias come before hidden (deterministic layout)
  topoQueue.sort((a, b) => {
    const aIsBase = a <= BIAS_ID;
    const bIsBase = b <= BIAS_ID;
    if (aIsBase !== bIsBase) return aIsBase ? -1 : 1;
    return a - b;
  });

  while (topoQueue.length > 0) {
    const nid = topoQueue.shift()!;
    topoOrder.push(nid);
    for (const dst of (outgoing.get(nid) || [])) {
      if (!reachable.has(dst)) continue;
      const deg = (inDegree.get(dst) || 1) - 1;
      inDegree.set(dst, deg);
      if (deg === 0) topoQueue.push(dst);
    }
  }

  // ── Step 3: Compute depths (matching Rust compute_depths) ──
  const depths = new Map<number, number>();
  for (const nid of node_ids) {
    if (nid <= BIAS_ID) depths.set(nid, 0);
    else depths.set(nid, 0);
  }

  for (const nid of topoOrder) {
    if (nid < OUTPUT_START || nid >= OUTPUT_START + OUTPUT_COUNT) {
      // Hidden nodes or other non-outputs
      if (nid <= BIAS_ID) continue; // Already set to 0
      const incs = incoming.get(nid) || [];
      const enabledIncs = incs.filter((x) => x.enabled && reachable.has(x.src));
      if (enabledIncs.length === 0) {
        depths.set(nid, 1);
      } else {
        let maxD = 0;
        for (const { src } of enabledIncs) {
          maxD = Math.max(maxD, (depths.get(src) ?? 0));
        }
        depths.set(nid, 1 + maxD);
      }
    }
  }

  // Force all output nodes to the same max depth
  const maxHidden = Math.max(
    0,
    ...Array.from(reachable)
      .filter((n) => n >= OUTPUT_START + OUTPUT_COUNT || n < OUTPUT_START)
      .filter((n) => n > BIAS_ID)
      .map((n) => depths.get(n) ?? 0),
  );

  for (let o = OUTPUT_START; o < OUTPUT_START + OUTPUT_COUNT; o++) {
    if (reachable.has(o)) depths.set(o, maxHidden + 1);
  }

  const maxDepth = Math.max(0, ...Array.from(depths.values()));

  // Store depth in nodeMeta
  for (const [nid, d] of depths) {
    const m = nodeMeta.get(nid);
    if (m) nodeMeta.set(nid, { ...m, depth: d });
  }

  // ── Step 4: Group by depth and position ──
  const layerGroups = new Map<number, number[]>();
  for (const nid of reachable) {
    const d = depths.get(nid) ?? 0;
    if (!layerGroups.has(d)) layerGroups.set(d, []);
    layerGroups.get(d)!.push(nid);
  }

  // Sort within each layer: inputs by ID, then hidden by ID, outputs at bottom
  for (const [, nodes] of layerGroups) {
    nodes.sort((a, b) => {
      // Inputs/bias first (sorted by ID)
      const aIsBase = a <= BIAS_ID;
      const bIsBase = b <= BIAS_ID;
      if (aIsBase !== bIsBase) return aIsBase ? -1 : 1;
      // Outputs at the bottom
      const aOut = a >= OUTPUT_START && a < OUTPUT_START + OUTPUT_COUNT;
      const bOut = b >= OUTPUT_START && b < OUTPUT_START + OUTPUT_COUNT;
      if (aOut !== bOut) return aOut ? 1 : -1;
      return a - b;
    });
  }

  // Position nodes
  const positions = new Map<number, { x: number; y: number }>();
  const cw = SVG_WIDTH - MARGIN.left - MARGIN.right;
  const ch = SVG_HEIGHT - MARGIN.top - MARGIN.bottom;

  for (const [depth, nodes] of layerGroups) {
    const x = MARGIN.left + (maxDepth === 0 ? 0.5 : depth / maxDepth) * cw;
    const spacing = ch / (nodes.length + 1);
    nodes.forEach((nid, idx) => {
      positions.set(nid, { x, y: MARGIN.top + spacing * (idx + 1) });
    });
  }

  return { positions, edges, nodeMeta, maxDepth, reachable, layerGroups };
}

// ── Component ──────────────────────────────────────────────────────

const NetworkInspectorPanel: React.FC<Props> = ({ genome, evalData, playerName, onClose }) => {
  const layout = useMemo(() => computeLayout(genome), [genome]);

  // Drag state
  const [panelPos, setPanelPos] = useState({ x: window.innerWidth - DEFAULT_PANEL_W - 20, y: 60 });
  const [panelSize, setPanelSize] = useState({ w: DEFAULT_PANEL_W, h: DEFAULT_PANEL_H });
  const [collapsed, setCollapsed] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const dragRef = useRef({ startX: 0, startY: 0, origX: 0, origY: 0 });
  const resizeRef = useRef({ startX: 0, startY: 0, origW: 0, origH: 0 });

  // Zoom/pan state for the SVG
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: SVG_WIDTH, h: SVG_HEIGHT });
  const [isPanning, setIsPanning] = useState(false);
  const panRef = useRef({ startX: 0, startY: 0, origVbx: 0, origVby: 0 });
  const svgRef = useRef<SVGSVGElement>(null);
  const svgContainerRef = useRef<HTMLDivElement>(null);

  // Zoom step
  const zoomLevel = SVG_WIDTH / viewBox.w;

  // Drag handlers
  const onHeaderMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return; // Don't drag when clicking buttons
    setIsDragging(true);
    dragRef.current = { startX: e.clientX, startY: e.clientY, origX: panelPos.x, origY: panelPos.y };
    e.preventDefault();
  }, [panelPos]);

  const onResizeMouseDown = useCallback((e: React.MouseEvent) => {
    setIsResizing(true);
    resizeRef.current = { startX: e.clientX, startY: e.clientY, origW: panelSize.w, origH: panelSize.h };
    e.preventDefault();
    e.stopPropagation();
  }, [panelSize]);

  useEffect(() => {
    if (!isDragging && !isResizing) return;
    const onMove = (e: MouseEvent) => {
      if (isDragging) {
        setPanelPos({
          x: Math.max(0, dragRef.current.origX + (e.clientX - dragRef.current.startX)),
          y: Math.max(0, dragRef.current.origY + (e.clientY - dragRef.current.startY)),
        });
      }
      if (isResizing) {
        setPanelSize({
          w: Math.max(MIN_PANEL_W, resizeRef.current.origW + (e.clientX - resizeRef.current.startX)),
          h: Math.max(MIN_PANEL_H, resizeRef.current.origH + (e.clientY - resizeRef.current.startY)),
        });
      }
    };
    const onUp = () => { setIsDragging(false); setIsResizing(false); };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [isDragging, isResizing]);

  // Pan handlers (separate from window drag/resize)
  useEffect(() => {
    if (!isPanning) return;
    const onMove = (e: MouseEvent) => {
      const dx = (e.clientX - panRef.current.startX) / zoomLevel;
      const dy = (e.clientY - panRef.current.startY) / zoomLevel;
      setViewBox((prev) => ({
        ...prev,
        x: panRef.current.origVbx - dx,
        y: panRef.current.origVby - dy,
      }));
    };
    const onUp = () => setIsPanning(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [isPanning, zoomLevel]);

  // Zoom with mouse wheel
  const onSvgWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const svgEl = svgRef.current;
    const containerEl = svgContainerRef.current;
    if (!svgEl || !containerEl) return;

    const rect = containerEl.getBoundingClientRect();
    // Cursor position relative to the SVG container, in CSS pixels
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    // Convert to SVG coordinates
    const scaleX = viewBox.w / rect.width;
    const scaleY = viewBox.h / rect.height;
    const svgX = viewBox.x + mx * scaleX;
    const svgY = viewBox.y + my * scaleY;

    const zoomFactor = e.deltaY < 0 ? 0.8 : 1.25; // zoom in / zoom out
    const newW = Math.max(SVG_WIDTH / 8, Math.min(SVG_WIDTH, viewBox.w * zoomFactor));
    const newH = Math.max(SVG_HEIGHT / 8, Math.min(SVG_HEIGHT, viewBox.h * zoomFactor));

    // Keep the point under the cursor fixed
    const newX = svgX - mx * (newW / rect.width);
    const newY = svgY - my * (newH / rect.height);

    setViewBox({
      x: Math.max(0, Math.min(SVG_WIDTH - newW, newX)),
      y: Math.max(0, Math.min(SVG_HEIGHT - newH, newY)),
      w: newW,
      h: newH,
    });
  }, [viewBox]);

  // Start panning (left-click on SVG background)
  const onSvgMouseDown = useCallback((e: React.MouseEvent) => {
    // Only pan on left-click and when not clicking on a node
    if (e.button !== 0) return;
    const target = e.target as SVGElement;
    if (target.closest('.network-node-group')) return; // Don't pan when clicking nodes
    setIsPanning(true);
    panRef.current = { startX: e.clientX, startY: e.clientY, origVbx: viewBox.x, origVby: viewBox.y };
    e.preventDefault();
  }, [viewBox]);

  const resetZoom = useCallback(() => {
    setViewBox({ x: 0, y: 0, w: SVG_WIDTH, h: SVG_HEIGHT });
  }, []);

  const activations = evalData?.activations ?? [];
  const beliefs = evalData?.beliefs ?? [];
  const outputs = evalData?.outputs ?? [];

  const actColor = (v: number, isOut: boolean) => {
    const a = Math.max(0, Math.min(1, v));
    if (isOut) return `rgba(212,175,55,${(0.18 + a * 0.82).toFixed(2)})`;
    return `rgba(60,122,137,${(0.12 + a * 0.88).toFixed(2)})`;
  };

  // Structural connectivity must stay visible regardless of activation; the
  // activation only *emphasizes* an edge (raises opacity), never hides it.
  const edgeAlpha = (srcAct: number) => 0.35 + srcAct * 0.5;

  // Count reachable edges (enabled edges between reachable nodes)
  const reachableEdges = layout.edges.filter(
    (e) => layout.reachable.has(e.src) && layout.reachable.has(e.dst),
  );

  return (
    <div
      className={`inspector-panel ${collapsed ? 'collapsed' : ''} ${isDragging ? 'dragging' : ''}`}
      style={{ left: panelPos.x, top: panelPos.y, width: panelSize.w, height: collapsed ? 'auto' : panelSize.h }}
    >
      {/* Header — draggable */}
      <div className="inspector-header" onMouseDown={onHeaderMouseDown}>
        <span className="inspector-title">🧠 WANN Inspector — {playerName}</span>
        <span className={`inspector-brain-badge ${evalData?.brain_type === 'lead' ? 'lead' : 'follow'}`}>
          {evalData?.brain_type === 'lead' ? 'Lead Brain' : 'Follow Brain'}
        </span>
        <span className="inspector-node-count">
          {layout.reachable.size} nodes · {reachableEdges.filter((e) => e.enabled).length} edges
        </span>
        <button
          type="button"
          className="inspector-minimize-btn"
          onClick={() => setCollapsed(!collapsed)}
          aria-label={collapsed ? 'Expand' : 'Collapse'}
          title={collapsed ? 'Expand panel' : 'Collapse panel'}
        >
          {collapsed ? '□' : '−'}
        </button>
        <button type="button" className="inspector-close-btn" onClick={onClose} aria-label="Close inspector">
          ✕
        </button>
      </div>

      {!collapsed && (
        <>
          {/* SVG Canvas */}
          <div
            className={`inspector-svg-container ${isPanning ? 'panning' : ''}`}
            ref={svgContainerRef}
            onWheel={onSvgWheel}
          >
            {/* Zoom controls */}
            <div className="svg-zoom-controls">
              <button type="button" className="svg-zoom-btn" onClick={() => {
                const cw = viewBox.w * 0.8;
                const ch = viewBox.h * 0.8;
                setViewBox((prev) => ({
                  x: prev.x + (prev.w - cw) / 2,
                  y: prev.y + (prev.h - ch) / 2,
                  w: Math.max(SVG_WIDTH / 8, cw),
                  h: Math.max(SVG_HEIGHT / 8, ch),
                }));
              }} title="Zoom in">+</button>
              <span className="svg-zoom-level" title="Reset zoom" onClick={resetZoom}>
                {Math.round(zoomLevel * 100)}%
              </span>
              <button type="button" className="svg-zoom-btn" onClick={() => {
                const cw = viewBox.w * 1.25;
                const ch = viewBox.h * 1.25;
                if (cw >= SVG_WIDTH && ch >= SVG_HEIGHT) { resetZoom(); return; }
                setViewBox((prev) => ({
                  x: Math.max(0, prev.x - (cw - prev.w) / 2),
                  y: Math.max(0, prev.y - (ch - prev.h) / 2),
                  w: Math.min(SVG_WIDTH, cw),
                  h: Math.min(SVG_HEIGHT, ch),
                }));
              }} title="Zoom out">−</button>
              {zoomLevel > 1.05 && (
                <button type="button" className="svg-zoom-reset" onClick={resetZoom} title="Reset zoom">
                  ↺
                </button>
              )}
            </div>
            <svg
              ref={svgRef}
              viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
              className="network-svg"
              preserveAspectRatio="xMidYMid meet"
              onMouseDown={onSvgMouseDown}
              style={{ cursor: isPanning ? 'grabbing' : zoomLevel > 1.01 ? 'grab' : 'default' }}
            >
              <defs>
                <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="var(--accent-cyan)" stopOpacity="0.35" />
                  <stop offset="100%" stopColor="var(--accent-cyan)" stopOpacity="0" />
                </radialGradient>
                <radialGradient id="outputGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="var(--accent-gold)" stopOpacity="0.45" />
                  <stop offset="100%" stopColor="var(--accent-gold)" stopOpacity="0" />
                </radialGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2.2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                {/* Arrow marker for edges */}
                <marker id="arrowCyan" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="4" markerHeight="4" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent-cyan)" opacity="0.6" />
                </marker>
                <marker id="arrowGold" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="4" markerHeight="4" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent-gold)" opacity="0.6" />
                </marker>
                <marker id="arrowGray" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="4" markerHeight="4" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" fill="rgba(255,255,255,0.12)" />
                </marker>
              </defs>

              {/* Edges — disabled first (behind), then enabled */}
              {reachableEdges.filter((e) => !e.enabled).map((e, i) => {
                const sp = layout.positions.get(e.src);
                const dp = layout.positions.get(e.dst);
                if (!sp || !dp) return null;
                return (
                  <line
                    key={`dis-${i}`}
                    x1={sp.x} y1={sp.y} x2={dp.x} y2={dp.y}
                    stroke="rgba(255,255,255,0.16)"
                    strokeWidth={0.7}
                    strokeDasharray="2 4"
                    markerEnd="url(#arrowGray)"
                  />
                );
              })}
              {reachableEdges.filter((e) => e.enabled).map((e, i) => {
                const sp = layout.positions.get(e.src);
                const dp = layout.positions.get(e.dst);
                if (!sp || !dp) return null;
                const srcAct = activations[e.src] ?? 0;
                const hot = srcAct > 0.35;
                const isNeg = e.sign === -1;
                return (
                  <line
                    key={`en-${i}`}
                    x1={sp.x} y1={sp.y} x2={dp.x} y2={dp.y}
                    stroke={isNeg ? 'var(--accent-gold)' : 'var(--accent-cyan)'}
                    strokeOpacity={edgeAlpha(srcAct)}
                    strokeWidth={hot ? 2.0 : 1.1}
                    strokeDasharray={isNeg ? '5 3' : 'none'}
                    markerEnd={isNeg ? 'url(#arrowGold)' : 'url(#arrowCyan)'}
                    className={hot ? 'edge-active' : 'edge-idle'}
                  />
                );
              })}

              {/* Nodes — only reachable */}
              {Array.from(layout.positions.entries())
                .filter(([nid]) => layout.reachable.has(nid))
                .map(([nid, pos]) => {
                  const meta = layout.nodeMeta.get(nid);
                  if (!meta) return null;

                  const isIn = meta.type === 0;
                  const isBias = meta.type === 1;
                  const isOut = meta.type === 3;
                  const isHid = meta.type === 2;

                  const act = isIn ? (beliefs[nid] ?? 0) : (activations[nid] ?? 0);
                  const r = isOut ? OUTPUT_R : isIn ? INPUT_R : NODE_R;
                  const fill = actColor(act, isOut);
                  const stroke = isOut
                    ? 'var(--accent-gold)'
                    : isBias
                      ? 'var(--accent-purple)'
                      : 'var(--accent-cyan)';
                  const alive = act > 0.45;

                  // Label
                  let label = '';
                  let labelRight = false;
                  if (isIn && nid < INPUT_COUNT) {
                    label = INPUT_SHORT[nid] ?? `I${nid}`;
                    labelRight = false;
                  } else if (isBias) {
                    label = 'BIAS';
                    labelRight = false;
                  } else if (isOut) {
                    label = OUTPUT_LABELS[nid - OUTPUT_START] ?? `O${nid}`;
                    labelRight = true;
                  }
                  // Hidden nodes: show act|agg inside, no external label

                  const lx = labelRight ? pos.x + r + 6 : pos.x - r - 6;
                  const la = labelRight ? 'start' : 'end';

                  // Full tooltip text with descriptions
                  const tooltipParts: string[] = [];
                  if (isIn && nid < INPUT_COUNT) {
                    tooltipParts.push(`Input #${nid}: ${FEATURE_NAMES[nid]}`);
                    tooltipParts.push(FEATURE_DESCRIPTIONS[nid]);
                  } else if (isBias) {
                    tooltipParts.push('Bias node — always outputs 1.0');
                    tooltipParts.push('Provides a constant reference signal to the network.');
                  } else if (isOut) {
                    const oi = nid - OUTPUT_START;
                    tooltipParts.push(`Output #${oi}: ${OUTPUT_LABELS[oi]}`);
                    tooltipParts.push(OUTPUT_DESCRIPTIONS[oi]);
                  } else if (isHid) {
                    const actName = ACT_NAMES[meta.act];
                    const aggName = AGG_NAMES[meta.agg];
                    const actDetail = actName === 'ID' ? 'pass-through' : actName === 'NOT' ? 'invert (1−x)' : 'step at 0.5·W';
                    const aggDetail = aggName === 'SUM' ? 'weighted sum' : aggName === 'MIN' ? 'logical AND (minimum)' : 'logical OR (maximum)';
                    tooltipParts.push(`Hidden node H${nid}`);
                    tooltipParts.push(`Activation: ${actName} (${actDetail})`);
                    tooltipParts.push(`Aggregation: ${aggName} (${aggDetail})`);
                  }
                  tooltipParts.push(`Current value: ${act.toFixed(4)}`);
                  if (isHid || isOut) tooltipParts.push(`Topological depth: ${meta.depth}`);

                  return (
                    <g key={nid} className="network-node-group">
                      {/* Glow halo */}
                      {alive && (
                        <circle
                          cx={pos.x} cy={pos.y} r={r * 2.8}
                          fill={isOut ? 'url(#outputGlow)' : 'url(#nodeGlow)'}
                          className="node-glow-pulse"
                        />
                      )}

                      {/* Shape — boxes for hidden/outputs (matching Graphviz DOT style) */}
                      {isBias ? (
                        <rect
                          x={pos.x - r} y={pos.y - r}
                          width={r * 2} height={r * 2} rx={2.5}
                          fill={fill} stroke={stroke} strokeWidth={1}
                          transform={`rotate(45 ${pos.x} ${pos.y})`}
                        />
                      ) : isHid ? (
                        <rect
                          x={pos.x - r} y={pos.y - r}
                          width={r * 2} height={r * 2} rx={2}
                          fill={fill} stroke={stroke} strokeWidth={1}
                        />
                      ) : isOut ? (
                        <rect
                          x={pos.x - r} y={pos.y - r}
                          width={r * 2} height={r * 2} rx={3}
                          fill={fill} stroke={stroke} strokeWidth={2}
                          filter={alive ? 'url(#glow)' : undefined}
                        />
                      ) : (
                        <circle
                          cx={pos.x} cy={pos.y} r={r}
                          fill={fill} stroke={stroke} strokeWidth={1}
                        />
                      )}

                      {/* Value inside node */}
                      {(isOut || isHid) && (
                        <text x={pos.x} y={pos.y + 0.5} textAnchor="middle"
                          dominantBaseline="central" fill="#fff"
                          fontSize={isOut ? 6.5 : 5} fontWeight={700}
                          style={{ pointerEvents: 'none' }}>
                          {act.toFixed(1)}
                        </text>
                      )}
                      {isIn && (
                        <text x={pos.x} y={pos.y + 0.5} textAnchor="middle"
                          dominantBaseline="central" fill="#fff"
                          fontSize={4.2} fontWeight={600}
                          style={{ pointerEvents: 'none' }}>
                          {act > 0.01 ? act.toFixed(1) : ''}
                        </text>
                      )}
                      {isBias && (
                        <text x={pos.x} y={pos.y + 0.5} textAnchor="middle"
                          dominantBaseline="central" fill="#fff"
                          fontSize={4.2} fontWeight={700}
                          style={{ pointerEvents: 'none' }}>
                          1.0
                        </text>
                      )}

                      {/* External label */}
                      {label && (
                        <text x={lx} y={pos.y + 0.5} textAnchor={la}
                          dominantBaseline="central" fill="rgba(255,255,255,0.5)"
                          fontSize={isOut ? 7.5 : 5.2}
                          fontWeight={isOut ? 700 : 400}
                          fontFamily="'Outfit', sans-serif"
                          style={{ pointerEvents: 'none' }}>
                          {label}
                        </text>
                      )}

                      <title>{tooltipParts.join('\n')}</title>
                    </g>
                  );
                })}

              {/* Section labels */}
              <text x={MARGIN.left + 2} y={14} textAnchor="start" fill="rgba(255,255,255,0.22)" fontSize={7.5} fontFamily="'Outfit',sans-serif">
                INPUTS
              </text>
              <text x={SVG_WIDTH - MARGIN.right} y={14} textAnchor="end" fill="rgba(255,255,255,0.22)" fontSize={7.5} fontFamily="'Outfit',sans-serif">
                OUTPUTS
              </text>
              {layout.maxDepth > 1 && (
                <text x={SVG_WIDTH / 2} y={14} textAnchor="middle" fill="rgba(255,255,255,0.18)" fontSize={7} fontFamily="'Outfit',sans-serif">
                  HIDDEN LAYERS
                </text>
              )}
            </svg>
          </div>

          {/* Legend */}
          <div className="inspector-legend">
            <span className="legend-item"><span className="leg-line solid-cyan" />+1 sign (solid)</span>
            <span className="legend-item"><span className="leg-line dashed-gold" />−1 sign (dashed)</span>
            <span className="legend-item"><span className="leg-line dotted-gray" />disabled</span>
            <span className="legend-item"><span className="leg-dot active" />active &gt;0.45</span>
            <span className="legend-item"><span className="leg-dot idle" />idle</span>
          </div>

          {/* Output bar chart */}
          <div className="inspector-output-bars">
            {OUTPUT_LABELS.map((name, idx) => {
              const val = outputs[idx] ?? 0;
              const pct = Math.round(Math.max(0, Math.min(1, val)) * 100);
              const chosen = val > 0 && outputs.every((o, i) => i === idx || val >= o - 1e-9);
              return (
                <div key={name} className={`obar-item ${chosen ? 'chosen' : ''}`} title={OUTPUT_DESCRIPTIONS[idx]}>
                  <span className="obar-label">{name}</span>
                  <div className="obar-track">
                    <div className="obar-fill" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="obar-value">{val.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Resize handle */}
      {!collapsed && (
        <div className="inspector-resize-handle" onMouseDown={onResizeMouseDown}>
          <svg width="12" height="12" viewBox="0 0 12 12">
            <path d="M10,2 L2,10 M10,6 L6,10 M10,10" stroke="rgba(255,255,255,0.3)" strokeWidth="1.2" fill="none" />
          </svg>
        </div>
      )}
    </div>
  );
};

export default NetworkInspectorPanel;

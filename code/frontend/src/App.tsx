import React, { useEffect, useState, useRef } from 'react';
import init, { WannSuecaGameSession } from './wasm/sueca_wasm';
import wasmUrl from './wasm/sueca_wasm_bg.wasm?url';
import bestGenome from './best_genome_final.json';
import Card from './components/Card';
import NetworkInspectorPanel, { type NetworkEval } from './components/NetworkInspectorPanel';

interface WasmLastTrick {
  cards: number[];
  seats: number[];
  winner: number;
  points: number;
}

interface WasmGameState {
  trump: number;
  player_hand: number[];
  legal_moves: number[];
  other_hands_sizes: number[];
  other_hands: number[][];
  current_trick: number[];
  current_trick_seats: number[];
  led_suit: number;
  current_player: number;
  team_02_score: number;
  team_13_score: number;
  trick_number: number;
  voids: number[];
  is_over: boolean;
  winner_team?: number;
  game_points_02: number;
  game_points_13: number;
  last_trick?: WasmLastTrick;
}

const SUIT_SYMBOLS = ['♥', '♦', '♣', '♠'];
const SUIT_NAMES = ['Hearts', 'Diamonds', 'Clubs', 'Spades'];
const SUIT_COLORS = ['#ff2a5f', '#ffb800', '#00f5d4', '#00bbf9'];
const RANK_LABELS = ['2', '3', '4', '5', '6', 'Q', 'J', 'K', '7', 'A'];

function getCardName(card: number): string {
  const suit = SUIT_NAMES[Math.floor(card / 10)];
  const rank = RANK_LABELS[card % 10];
  const sym = SUIT_SYMBOLS[Math.floor(card / 10)];
  return `${rank}${sym} (${suit})`;
}

export const App: React.FC = () => {
  const [wasmReady, setWasmReady] = useState(false);
  const [session, setSession] = useState<WannSuecaGameSession | null>(null);
  const [gameState, setGameState] = useState<WasmGameState | null>(null);
  
  // Decoupled visual state for smooth trick completes
  const [visualTrick, setVisualTrick] = useState<{ cards: number[]; seats: number[] }>({ cards: [], seats: [] });
  
  const [logs, setLogs] = useState<string[]>([]);
  const [trickWinnerMsg, setTrickWinnerMsg] = useState<string | null>(null);
  const [isResolvingTrick, setIsResolvingTrick] = useState(false);
  const [isBotThinking, setIsBotThinking] = useState(false);
  const [trickDots, setTrickDots] = useState<( 'us' | 'them' | null)[]>(() => Array(10).fill(null));
  
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 100000));
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Refs for synchronous guard checks — prevents stale closures and concurrent bot turns
  const isBotThinkingRef = useRef(false);
  const sessionRef = useRef<WannSuecaGameSession | null>(null);
  const gameIdRef = useRef(0); // incremented on each new game; async ops abort if mismatch

  // Match & settings states
  const [gpLimit, setGpLimit] = useState(10);
  const [gpScoreUs, setGpScoreUs] = useState(0);
  const [gpScoreThem, setGpScoreThem] = useState(0);
  const [dealHistory, setDealHistory] = useState<{ id: string, usPts: number, themPts: number, usGP: number, themGP: number }[]>([]);
  const [showGameOverModal, setShowGameOverModal] = useState(false);
  
  const [showSettings, setShowSettings] = useState(false);
  const [showVoidTracker, setShowVoidTracker] = useState(false);
  const [showMatchLogs, setShowMatchLogs] = useState(false);
  const [voidTrackerExpanded, setVoidTrackerExpanded] = useState(false);
  const [matchLogsExpanded, setMatchLogsExpanded] = useState(false);
  // Drag state for sidebar panels — null means use CSS positioning
  const [voidTrackerOffset, setVoidTrackerOffset] = useState<{ x: number; y: number } | null>(null);
  const [matchLogsOffset, setMatchLogsOffset] = useState<{ x: number; y: number } | null>(null);
  const [draggingSidebar, setDraggingSidebar] = useState<'void' | 'logs' | null>(null);
  const sidebarDragRef = useRef({ startX: 0, startY: 0, origX: 0, origY: 0 });
  const sidebarDragMoved = useRef(false);
  const [animSpeed, setAnimSpeed] = useState<number>(1);
  const [botTypes, setBotTypes] = useState<number[]>([0, 0, 0]); // Seat 1, 2, 3
  const [trickOffsets, setTrickOffsets] = useState<Record<number, { dx: number; dy: number; rot: number }>>({});
  const [autoContinue, setAutoContinue] = useState(true);
  const [inspectingSeat, setInspectingSeat] = useState<number | null>(null);
  const [networkEval, setNetworkEval] = useState<NetworkEval | null>(null);
  const [pendingContinue, setPendingContinue] = useState(false);

  const logsEndRef = useRef<HTMLDivElement>(null);
  const pendingResolveRef = useRef<{ lastTrick: WasmLastTrick; isOver: boolean; gp02: number; gp13: number; trickNumber: number; team02Score: number; team13Score: number } | null>(null);

  const getPlayerName = (seat: number, currentBotTypes = botTypes) => {
    if (seat === 0) return 'You';
    const typeNames = ['WANN', 'Initial Bot', 'Hard Bot'];
    const botType = currentBotTypes[seat - 1];
    const botTypeName = typeNames[botType] || 'WANN';
    if (seat === 2) return `${botTypeName} Partner`;
    return `${botTypeName} Opponent ${seat === 1 ? 'L' : 'R'}`;
  };

  const generateTrickOffsets = () => {
    const offsets: Record<number, { dx: number; dy: number; rot: number }> = {};
    for (let i = 0; i < 4; i++) {
      offsets[i] = {
        dx: (Math.random() - 0.5) * 15,
        dy: (Math.random() - 0.5) * 15,
        rot: (Math.random() - 0.5) * 45,
      };
    }
    setTrickOffsets(offsets);
  };

  const handleBotTypeChange = (index: number, value: number) => {
    const newTypes = [...botTypes];
    newTypes[index] = value;
    setBotTypes(newTypes);
    if (session) {
      session.set_bot_types(newTypes[0], newTypes[1], newTypes[2]);
    }
    // Close inspector if the inspected seat is no longer WANN
    const seat = index + 1; // index 0→seat 1, index 1→seat 2, index 2→seat 3
    if (inspectingSeat === seat && value !== 0) {
      setInspectingSeat(null);
      setNetworkEval(null);
    }
    const targetName = index === 1 ? 'Partner' : index === 0 ? 'Opponent L' : 'Opponent R';
    const botLabel = value === 0 ? 'WANN Brain' : value === 1 ? 'Initial Bot' : 'Hard Bot';
    setLogs((prev) => [...prev, `Configured ${targetName} to play as ${botLabel}.`]);
  };

  // Shared trick resolution — called when a trick completes (player or bot)
  const resolveTrick = (
    nextState: WasmGameState,
    lastT: WasmLastTrick,
  ): boolean => {
    const shouldPause = !autoContinue || inspectingSeat !== null;

    setGameState(nextState);
    setVisualTrick({
      cards: [...lastT.cards],
      seats: [...lastT.seats],
    });
    setIsResolvingTrick(true);
    setTrickWinnerMsg(`${getPlayerName(lastT.winner)} wins the trick (+${lastT.points} points)`);

    pendingResolveRef.current = {
      lastTrick: lastT,
      isOver: nextState.is_over,
      gp02: nextState.game_points_02,
      gp13: nextState.game_points_13,
      trickNumber: nextState.trick_number,
      team02Score: nextState.team_02_score,
      team13Score: nextState.team_13_score,
    };

    if (shouldPause) {
      setPendingContinue(true);
      return true;
    }

    // Auto-continue after delay
    setTimeout(() => {
      finalizeTrickResolution();
    }, 1500 / animSpeed);
    return false;
  };

  // Finalize trick resolution: clear visuals, update dots/logs/scores
  const finalizeTrickResolution = () => {
    const pending = pendingResolveRef.current;
    if (!pending) return;

    setVisualTrick({ cards: [], seats: [] });
    setTrickDots((prev) => {
      const updated = [...prev];
      updated[pending.trickNumber - 1] = (pending.lastTrick.winner % 2 === 0) ? 'us' : 'them';
      return updated;
    });
    setLogs((prev) => [...prev, `--- Trick won by ${getPlayerName(pending.lastTrick.winner)} (${pending.lastTrick.points} pts) ---`]);
    setTrickWinnerMsg(null);
    setIsResolvingTrick(false);
    setPendingContinue(false);
    isBotThinkingRef.current = false;
    setIsBotThinking(false);

    generateTrickOffsets();

    if (pending.isOver) {
      setGpScoreUs((prev) => prev + pending.gp02);
      setGpScoreThem((prev) => prev + pending.gp13);
      setDealHistory((prev) => [...prev, {
        id: crypto.randomUUID(),
        usPts: pending.team02Score,
        themPts: pending.team13Score,
        usGP: pending.gp02,
        themGP: pending.gp13,
      }]);
    }

    pendingResolveRef.current = null;
  };

  // Unified Continue handler: advances past trick resolution or between-card pauses
  const handleContinue = () => {
    if (isResolvingTrick) {
      finalizeTrickResolution();
    } else {
      setPendingContinue(false);
    }
  };



  // Initialize WASM
  useEffect(() => {
    const initWasm = async () => {
      try {
        await init(wasmUrl);
        setWasmReady(true);
      } catch (err) {
        console.error('WASM init failed', err);
        setErrorMsg('Failed to initialize WebAssembly game engine.');
      }
    };
    initWasm();
  }, []);

  // Scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Start new game
  const startNewGame = (customSeed?: number, resetGP = false) => {
    if (!wasmReady) return;
    try {
      // Abort any in-flight bot turn from a previous game
      gameIdRef.current += 1;
      isBotThinkingRef.current = false;

      const activeSeed = customSeed !== undefined ? customSeed : Math.floor(Math.random() * 100000);
      setSeed(activeSeed);

      const newSession = new WannSuecaGameSession(JSON.stringify(bestGenome), BigInt(activeSeed));
      newSession.set_bot_types(botTypes[0], botTypes[1], botTypes[2]);
      
      const stateStr = newSession.get_state_json();
      const stateObj = JSON.parse(stateStr) as WasmGameState;
      
      setSession(newSession);
      setGameState(stateObj);
      
      setVisualTrick({ cards: stateObj.current_trick, seats: stateObj.current_trick_seats });
      
      generateTrickOffsets();
      
      if (resetGP) {
        setGpScoreUs(0);
        setGpScoreThem(0);
        setDealHistory([]);
        setShowGameOverModal(false);
      }
      
      const pName = getPlayerName(stateObj.current_player);
      setLogs([
        `Game started (Seed: ${activeSeed}). Trump is ${SUIT_NAMES[stateObj.trump]} ${SUIT_SYMBOLS[stateObj.trump]}.`, 
        `${pName} leads the first trick.`
      ]);
      setTrickWinnerMsg(null);
      setIsResolvingTrick(false);
      setIsBotThinking(false);
      setPendingContinue(false);
      setInspectingSeat(null);
      setNetworkEval(null);
      pendingResolveRef.current = null;
      setTrickDots(Array(10).fill(null));
      setErrorMsg(null);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(`Failed to start game: ${err.message || err}`);
    }
  };

  // Keep sessionRef in sync with session state
  useEffect(() => {
    sessionRef.current = session;
  }, [session]);

  // Autostart first game once WASM is ready
  useEffect(() => {
    if (wasmReady) {
      startNewGame();
    }
  }, [wasmReady]);

  // Sidebar panel drag effect
  useEffect(() => {
    if (!draggingSidebar) return;
    sidebarDragMoved.current = false;
    const onMove = (e: MouseEvent) => {
      const dx = e.clientX - sidebarDragRef.current.startX;
      const dy = e.clientY - sidebarDragRef.current.startY;
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) sidebarDragMoved.current = true;
      const offset = {
        x: sidebarDragRef.current.origX + dx,
        y: Math.max(0, sidebarDragRef.current.origY + dy),
      };
      if (draggingSidebar === 'void') setVoidTrackerOffset(offset);
      else setMatchLogsOffset(offset);
    };
    const onUp = () => setDraggingSidebar(null);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); };
  }, [draggingSidebar]);

  // Poll WANN eval data when inspector is active.
  // Depends on gameState (card plays, trick completion) AND isResolvingTrick
  // (Continue click, trick cleanup) to refresh at every visible state transition.
  useEffect(() => {
    if (inspectingSeat === null || !session || !gameState || gameState.is_over) {
      if (inspectingSeat === null) setNetworkEval(null);
      return;
    }
    try {
      const evalJson = session.get_realtime_bot_eval(inspectingSeat);
      setNetworkEval(JSON.parse(evalJson) as NetworkEval);
    } catch {
      // Silently ignore — eval may fail during state transitions
    }
  }, [gameState, inspectingSeat, session, isResolvingTrick]);

  // Auto-transition to next deal or game over
  useEffect(() => {
    if (gameState && gameState.is_over && !isResolvingTrick && !pendingContinue && !showGameOverModal) {
      if (gpScoreUs >= gpLimit || gpScoreThem >= gpLimit) {
        setShowGameOverModal(true);
      } else {
        // Automatically start the next deal
        startNewGame(undefined, false);
      }
    }
  }, [gameState, isResolvingTrick, pendingContinue, gpScoreUs, gpScoreThem, gpLimit, showGameOverModal]);

  // Bot Turn Trigger Loop
  useEffect(() => {
    if (!session || !gameState || gameState.is_over || isResolvingTrick || pendingContinue || isBotThinkingRef.current) return;

    const currentPlayer = gameState.current_player;
    if (currentPlayer === 0) return; // Player's turn, wait for input

    // Synchronous guard: grab the session ref to avoid stale closure issues
    const currentSession = sessionRef.current;
    if (!currentSession) return;

    // Triggers bot turn
    const myGameId = gameIdRef.current;
    const triggerBotPlay = async () => {
      isBotThinkingRef.current = true;
      setIsBotThinking(true);

      if (gameIdRef.current !== myGameId || sessionRef.current !== currentSession) {
        isBotThinkingRef.current = false;
        setIsBotThinking(false);
        return;
      }

      // Artificial thinking delay scaled by animSpeed
      await new Promise((resolve) => setTimeout(resolve, 800 / animSpeed));

      try {
        const isCompleting = gameState.current_trick.length === 3;
        const playedCard = currentSession.play_bot_turn();

        const stateStr = currentSession.get_state_json();
        const nextState = JSON.parse(stateStr) as WasmGameState;

        // Add to logs
        const cardName = getCardName(playedCard);
        setLogs((prev) => [...prev, `${getPlayerName(currentPlayer)} played ${cardName}.`]);

        if (isCompleting) {
          // This bot completed the trick
          const lastT = nextState.last_trick!;
          resolveTrick(nextState, lastT);
          // If resolveTrick returned true (paused), leave isBotThinkingRef true
          // so the guard stays up until the user clicks Continue
        } else {
          // Standard play, not completing the trick
          setGameState(nextState);
          setVisualTrick({ cards: nextState.current_trick, seats: nextState.current_trick_seats });
          isBotThinkingRef.current = false;
          setIsBotThinking(false);
          if (!autoContinue) setPendingContinue(true);
        }
      } catch (err: any) {
        console.error(err);
        setErrorMsg(`Engine error during bot play: ${err.message || err}`);
        isBotThinkingRef.current = false;
        setIsBotThinking(false);
      }
    };

    triggerBotPlay();
  }, [gameState?.current_player, gameState?.trick_number, isResolvingTrick, pendingContinue, session, animSpeed, botTypes]);

  // Handle Player Card Play
  const handlePlayerCardPlay = async (card: number) => {
    if (!session || !gameState || isResolvingTrick || isBotThinking || gameState.current_player !== 0) return;

    try {
      const isCompleting = gameState.current_trick.length === 3;
      session.play_player_card(card);

      const stateStr = session.get_state_json();
      const nextState = JSON.parse(stateStr) as WasmGameState;

      // Add to logs
      const cardName = getCardName(card);
      setLogs((prev) => [...prev, `You played ${cardName}.`]);

      if (isCompleting) {
        // Player completed the trick
        const lastT = nextState.last_trick!;
        resolveTrick(nextState, lastT);
      } else {
        // Standard play
        setGameState(nextState);
        setVisualTrick({ cards: nextState.current_trick, seats: nextState.current_trick_seats });
        if (!autoContinue) setPendingContinue(true);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg(`Play rejected: ${err.message || err}`);
    }
  };

  if (errorMsg) {
    return (
      <div className="overlay-modal">
        <div className="modal-content" style={{ borderColor: 'var(--accent-red)' }}>
          <h2 className="winner-banner lost">Engine Error</h2>
          <p style={{ margin: '20px 0', fontSize: '0.95rem' }}>{errorMsg}</p>
          <button type="button" className="btn-primary" onClick={() => window.location.reload()}>Reload Page</button>
        </div>
      </div>
    );
  }

  if (!wasmReady || !gameState) {
    return (
      <div className="overlay-modal">
        <div className="modal-content">
          <div className="trump-indicator" style={{ marginBottom: '20px' }}>
            <span className="card-center-icon neon-glow-cyan" style={{ fontSize: '4rem' }}>♠</span>
          </div>
          <h2 style={{ fontFamily: 'Outfit', fontWeight: 800 }}>Loading Sueca WANN Engine...</h2>
          <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.9rem', marginTop: '10px' }}>Compiling WebAssembly and loading evolved logical weights...</p>
        </div>
      </div>
    );
  }

  // Find played cards for current trick seat positions
  const getPlayedCardForSeat = (seat: number) => {
    const idx = visualTrick.seats.indexOf(seat);
    if (idx !== -1) {
      return { card: visualTrick.cards[idx], seatIndex: seat };
    }
    return null;
  };

  return (
    <div className="game-container" style={{ '--anim-speed': animSpeed } as React.CSSProperties}>
      {/* HEADER */}
      <header className="game-header">
        <div className="game-title">SUECA WANN</div>
        
        {/* Central game state info & Trick dots */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px', position: 'absolute', left: '50%', transform: 'translateX(-50%)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '30px', background: 'rgba(255,255,255,0.02)', padding: '6px 20px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Trump Suit</span>
              <span style={{ color: SUIT_COLORS[gameState.trump], fontWeight: 700, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ fontSize: '1.2rem', lineHeight: 1 }}>{SUIT_SYMBOLS[gameState.trump]}</span> {SUIT_NAMES[gameState.trump]}
              </span>
            </div>
            <div style={{ width: '1px', height: '24px', background: 'rgba(255,255,255,0.1)' }} />
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Seed</span>
              <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem' }}>{seed}</span>
            </div>
            {pendingContinue && (isResolvingTrick || gameState.current_player !== 0) && (
              <>
                <div style={{ width: '1px', height: '24px', background: 'rgba(255,255,255,0.1)' }} />
                <button type="button" className="btn-continue-header" onClick={handleContinue}>
                  Continue ▶
                </button>
              </>
            )}
          </div>
          {/* Trick outcome dots */}
          <div className="trick-indicator-list" style={{ margin: '0' }}>
            {trickDots.map((dot, idx) => (
              <div 
                key={idx} 
                className={`trick-dot ${dot === 'us' ? 'won-us' : dot === 'them' ? 'won-them' : ''}`}
                title={`Trick ${idx + 1}`}
              />
            ))}
          </div>
        </div>

        {/* Right Section: Match Tracker & Settings */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <div className="match-tracker">
            <div className="tracker-row">
              <span className="tracker-label us">Us</span>
              <div className="tracker-bars">
                {Array(gpLimit).fill(0).map((_, i) => (
                  <span key={i} className={`tracker-bar ${i < gpScoreUs ? 'active-us' : ''}`}>|</span>
                ))}
              </div>
            </div>
            <div className="tracker-row">
              <span className="tracker-label them">Them</span>
              <div className="tracker-bars">
                {Array(gpLimit).fill(0).map((_, i) => (
                  <span key={i} className={`tracker-bar ${i < gpScoreThem ? 'active-them' : ''}`}>|</span>
                ))}
              </div>
            </div>
          </div>
          
          <button type="button" className="btn-settings-gear" onClick={() => setShowSettings(true)} title="Settings">
            ⚙
          </button>
        </div>
      </header>

      {/* GAME BOARD LAYOUT */}
      <div className="game-main">
        {/* Absolutely Positioned Moveable Panels */}
        
        {/* Void Tracker */}
        {showVoidTracker && (
          <div
            className={`sidebar-panel void-tracker-panel side-left ${voidTrackerExpanded ? 'expanded' : ''} ${draggingSidebar === 'void' ? 'dragging' : ''}`}
            style={voidTrackerOffset ? { left: voidTrackerOffset.x, top: voidTrackerOffset.y, right: 'auto' } : undefined}
          >
            <div
              className="panel-header"
              onMouseDown={(e) => {
                if ((e.target as HTMLElement).closest('button')) return;
                const panel = e.currentTarget.parentElement!;
                const panelRect = panel.getBoundingClientRect();
                const parentRect = panel.offsetParent!.getBoundingClientRect();
                sidebarDragRef.current = {
                  startX: e.clientX, startY: e.clientY,
                  origX: panelRect.left - parentRect.left,
                  origY: panelRect.top - parentRect.top,
                };
                setDraggingSidebar('void');
                e.preventDefault();
              }}
            >
              <span
                className="panel-title-text"
                onClick={() => {
                  if (sidebarDragMoved.current) { sidebarDragMoved.current = false; return; }
                  setVoidTrackerExpanded(!voidTrackerExpanded);
                }}
              >
                Void Tracker
              </span>
              <button
                type="button"
                className="panel-minimize-btn"
                onClick={() => setVoidTrackerExpanded(!voidTrackerExpanded)}
                title={voidTrackerExpanded ? 'Collapse' : 'Expand'}
              >
                {voidTrackerExpanded ? '−' : '□'}
              </button>
            </div>
            {voidTrackerExpanded && (
              <div className="voids-grid">
                {[0, 1, 2, 3].map((playerIdx) => (
                  <div key={playerIdx} style={{ display: 'contents' }}>
                    <div style={{ gridColumn: 'span 4', color: 'rgba(255,255,255,0.4)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.5px', marginTop: '6px', fontWeight: 600 }}>
                      {getPlayerName(playerIdx)}
                    </div>
                    {SUIT_SYMBOLS.map((sym, suitIdx) => {
                      const isVoid = (gameState.voids[playerIdx] & (1 << suitIdx)) !== 0;
                      return (
                        <div
                          key={suitIdx}
                          className={`void-item ${isVoid ? 'void-active' : ''}`}
                          title={`${getPlayerName(playerIdx)} void in ${SUIT_NAMES[suitIdx]}`}
                        >
                          {sym}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Match Logs */}
        {showMatchLogs && (
          <div
            className={`sidebar-panel match-logs-panel side-right ${matchLogsExpanded ? 'expanded' : ''} ${draggingSidebar === 'logs' ? 'dragging' : ''}`}
            style={matchLogsOffset ? { left: matchLogsOffset.x, top: matchLogsOffset.y, right: 'auto' } : undefined}
          >
            <div
              className="panel-header"
              onMouseDown={(e) => {
                if ((e.target as HTMLElement).closest('button')) return;
                const panel = e.currentTarget.parentElement!;
                const panelRect = panel.getBoundingClientRect();
                const parentRect = panel.offsetParent!.getBoundingClientRect();
                sidebarDragRef.current = {
                  startX: e.clientX, startY: e.clientY,
                  origX: panelRect.left - parentRect.left,
                  origY: panelRect.top - parentRect.top,
                };
                setDraggingSidebar('logs');
                e.preventDefault();
              }}
            >
              <span
                className="panel-title-text"
                onClick={() => {
                  if (sidebarDragMoved.current) { sidebarDragMoved.current = false; return; }
                  setMatchLogsExpanded(!matchLogsExpanded);
                }}
              >
                Match Logs
              </span>
              <button
                type="button"
                className="panel-minimize-btn"
                onClick={() => setMatchLogsExpanded(!matchLogsExpanded)}
                title={matchLogsExpanded ? 'Collapse' : 'Expand'}
              >
                {matchLogsExpanded ? '−' : '□'}
              </button>
            </div>
            {matchLogsExpanded && (
              <div className="log-list">
                {logs.map((log, idx) => (
                  /* eslint-disable-next-line react-doctor/no-array-index-as-key */
                  <div key={idx} className="log-item">{log}</div>
                ))}
                <div ref={logsEndRef} />
              </div>
            )}
          </div>
        )}

        {/* SEAT: TOP (Partner) */}
        <div className="seat-top">
            <div className={`player-info-card ${gameState.current_player === 2 && !gameState.is_over ? 'active-turn' : ''} ${inspectingSeat === 2 ? 'inspected' : ''}`}>
              <span className="player-role">Partner</span>
              <span className="player-name">{getPlayerName(2)}</span>
              {botTypes[1] === 0 && (
                <button
                  type="button"
                  className={`btn-inspect-brain ${inspectingSeat === 2 ? 'active' : ''}`}
                  onClick={() => setInspectingSeat(inspectingSeat === 2 ? null : 2)}
                  title={inspectingSeat === 2 ? 'Close inspector' : 'Inspect WANN brain'}
                >
                  🧠
                </button>
              )}
            </div>
          </div>

          <div className="center-row">
            {/* SEAT: LEFT (Opponent L) */}
            <div className="seat-left">
              <div className={`player-info-card ${gameState.current_player === 1 && !gameState.is_over ? 'active-turn' : ''} ${inspectingSeat === 1 ? 'inspected' : ''}`}>
                <span className="player-role">Opponent L</span>
                <span className="player-name">{getPlayerName(1)}</span>
                {botTypes[0] === 0 && (
                  <button
                    type="button"
                    className={`btn-inspect-brain ${inspectingSeat === 1 ? 'active' : ''}`}
                    onClick={() => setInspectingSeat(inspectingSeat === 1 ? null : 1)}
                    title={inspectingSeat === 1 ? 'Close inspector' : 'Inspect WANN brain'}
                  >
                    🧠
                  </button>
                )}
              </div>
            </div>

            {/* CENTRAL PLAY ARENA */}
            <div className="play-arena">
              <div className="table-surface" />
              
              <div className="trick-cards-overlay">
                {/* Show played cards of the current trick overlayed by played order */}
                {[0, 1, 2, 3].map((seatIdx) => {
                  const played = getPlayedCardForSeat(seatIdx);
                  if (!played) return null;
                  
                  let baseTransform = '';
                  if (seatIdx === 0) baseTransform = 'translateY(10px) rotate(-5deg)';
                  else if (seatIdx === 3) baseTransform = 'translateX(45px) rotate(12deg)';
                  else if (seatIdx === 2) baseTransform = 'translateY(-10px) rotate(4deg)';
                  else if (seatIdx === 1) baseTransform = 'translateX(-45px) rotate(-8deg)';
                  
                  const offsets = trickOffsets[seatIdx] || { dx: 0, dy: 0, rot: 0 };
                  const playOrderIndex = visualTrick.seats.indexOf(seatIdx);
                  const zIndex = 10 + playOrderIndex;
                  const transformStyle = `${baseTransform} translate(${offsets.dx}px, ${offsets.dy}px) rotate(${offsets.rot}deg)`;
                  
                  return (
                    <div 
                      key={seatIdx} 
                      className={`played-card-wrapper`}
                      style={{
                        transform: transformStyle,
                        zIndex: zIndex,
                      }}
                    >
                      <Card 
                        card={played.card} 
                        throwAnimSeat={played.seatIndex}
                      />
                    </div>
                  );
                })}

                {/* Trick winner alert banner */}
                {trickWinnerMsg && (
                  <div className="trick-winner-alert">
                    {trickWinnerMsg}
                  </div>
                )}
              </div>
            </div>

            {/* SEAT: RIGHT (Opponent R) */}
            <div className="seat-right">
              <div className={`player-info-card ${gameState.current_player === 3 && !gameState.is_over ? 'active-turn' : ''} ${inspectingSeat === 3 ? 'inspected' : ''}`}>
                <span className="player-role">Opponent R</span>
                <span className="player-name">{getPlayerName(3)}</span>
                {botTypes[2] === 0 && (
                  <button
                    type="button"
                    className={`btn-inspect-brain ${inspectingSeat === 3 ? 'active' : ''}`}
                    onClick={() => setInspectingSeat(inspectingSeat === 3 ? null : 3)}
                    title={inspectingSeat === 3 ? 'Close inspector' : 'Inspect WANN brain'}
                  >
                    🧠
                  </button>
                )}
              </div>
            </div>
          </div>

        {/* SEAT: BOTTOM (Player Hand) */}
        <div className="seat-bottom">
          <div 
            className="turn-indicator"
            style={{ 
              color: gameState.current_player === 0 && !gameState.is_over && !isResolvingTrick ? 'var(--accent-cyan)' : 'rgba(255,255,255,0.3)',
            }}
          >
            {gameState.current_player === 0 && !gameState.is_over && !isResolvingTrick ? (
              <>
                <span className="card-dot active" style={{ display: 'inline-block' }} /> Your Turn
              </>
            ) : isBotThinking ? (
              'Opponents are thinking...'
            ) : isResolvingTrick ? (
              'Resolving trick...'
            ) : (
              'Waiting for turn...'
            )}
          </div>

          <div className="hand-area">
            {gameState.player_hand.map((card) => {
              const isPlayable = gameState.legal_moves.includes(card);
              const isMyTurn = gameState.current_player === 0;
              return (
                <Card
                  key={card}
                  card={card}
                  isPlayable={isPlayable}
                  disabled={!isMyTurn || isResolvingTrick}
                  onClick={() => handlePlayerCardPlay(card)}
                />
              );
            })}
          </div>
        </div>
      </div>

      {/* NETWORK INSPECTOR PANEL */}
      {inspectingSeat !== null && (
        <NetworkInspectorPanel
          genome={(networkEval?.brain_type === 'follow') ? bestGenome.follow : bestGenome.lead}
          evalData={networkEval}
          playerName={getPlayerName(inspectingSeat)}
          onClose={() => {
            setInspectingSeat(null);
            setNetworkEval(null);
            // Pause persists — user must click Continue explicitly
          }}
        />
      )}

      {/* SETTINGS OVERLAY MODAL */}
      {showSettings && (
        // eslint-disable-next-line react-doctor/prefer-tag-over-role
        <div className="settings-overlay" role="button" tabIndex={0} onClick={() => setShowSettings(false)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setShowSettings(false); }}>
          <div className="settings-content" role="presentation" onClick={(e) => e.stopPropagation()}>
            <h3 className="settings-title">Match Settings</h3>

            <div className="settings-grid">
              {/* ── Left Column: General Settings ── */}
              <div className="settings-col">
                <h4 className="settings-section-title">General</h4>

                <div className="setting-row">
                  <span className="setting-label">Match GP Limit</span>
                  <div className="setting-control">
                    <select value={gpLimit} onChange={(e) => setGpLimit(parseInt(e.target.value))}>
                      <option value={2}>2 GP</option>
                      <option value={4}>4 GP</option>
                      <option value={10}>10 GP</option>
                      <option value={20}>20 GP</option>
                    </select>
                  </div>
                </div>

                <div className="setting-row">
                  <span className="setting-label">Animation Speed</span>
                  <div className="setting-control">
                    <select value={animSpeed} onChange={(e) => setAnimSpeed(parseInt(e.target.value))}>
                      <option value={1}>1x (Normal)</option>
                      <option value={2}>2x (Fast)</option>
                      <option value={4}>4x (Insane)</option>
                    </select>
                  </div>
                </div>

                <div className="setting-row">
                  <span className="setting-label">Auto-continue Tricks</span>
                  <div className="setting-control">
                    <label className="toggle-switch" aria-label="Auto-continue Tricks">
                      <input type="checkbox" checked={autoContinue} onChange={(e) => setAutoContinue(e.target.checked)} />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>

                <div className="setting-row">
                  <span className="setting-label">Show Void Tracker</span>
                  <div className="setting-control">
                    <label className="toggle-switch" aria-label="Show Void Tracker">
                      <input type="checkbox" checked={showVoidTracker} onChange={(e) => setShowVoidTracker(e.target.checked)} />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>

                <div className="setting-row">
                  <span className="setting-label">Show Match Logs</span>
                  <div className="setting-control">
                    <label className="toggle-switch" aria-label="Show Match Logs">
                      <input type="checkbox" checked={showMatchLogs} onChange={(e) => setShowMatchLogs(e.target.checked)} />
                      <span className="slider"></span>
                    </label>
                  </div>
                </div>
              </div>

              {/* ── Right Column: Bot Configurations ── */}
              <div className="settings-col">
                <h4 className="settings-section-title">Bot Brains</h4>

                <div className="setting-row">
                  <span className="setting-label">Opponent L (Seat 1)</span>
                  <div className="setting-control">
                    <select value={botTypes[0]} onChange={(e) => handleBotTypeChange(0, parseInt(e.target.value))}>
                      <option value={0}>WANN Brain</option>
                      <option value={1}>Initial Bot</option>
                      <option value={2}>Hard Bot</option>
                    </select>
                  </div>
                </div>

                <div className="setting-row">
                  <span className="setting-label">Partner (Seat 2)</span>
                  <div className="setting-control">
                    <select value={botTypes[1]} onChange={(e) => handleBotTypeChange(1, parseInt(e.target.value))}>
                      <option value={0}>WANN Brain</option>
                      <option value={1}>Initial Bot</option>
                      <option value={2}>Hard Bot</option>
                    </select>
                  </div>
                </div>

                <div className="setting-row">
                  <span className="setting-label">Opponent R (Seat 3)</span>
                  <div className="setting-control">
                    <select value={botTypes[2]} onChange={(e) => handleBotTypeChange(2, parseInt(e.target.value))}>
                      <option value={0}>WANN Brain</option>
                      <option value={1}>Initial Bot</option>
                      <option value={2}>Hard Bot</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            <div className="settings-actions">
              <button type="button" className="btn-primary" onClick={() => setShowSettings(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MATCH OVER DIALOG OVERLAY */}
      {(gpScoreUs >= gpLimit || gpScoreThem >= gpLimit) && (
        <div className="overlay-modal">
          <div className="modal-content" style={{ borderColor: 'var(--accent-purple)' }}>
            <div className="trump-indicator" style={{ marginBottom: '15px' }}>
              <span className="card-center-icon neon-glow-cyan" style={{ fontSize: '4.5rem' }}>🏆</span>
            </div>
            
            {gpScoreUs >= gpLimit ? (
              <h1 className="winner-banner">Match Victory!</h1>
            ) : (
              <h1 className="winner-banner lost">Match Defeat</h1>
            )}

            <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.95rem', margin: '15px 0' }}>
              {gpScoreUs >= gpLimit 
                ? 'Congratulations! Your team won the match.' 
                : 'The opponents beat your team in the match.'}
            </p>

            <div className="game-over-stats" style={{ background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '12px' }}>
              <div className="stat-row">
                <span className="stat-label">Final Score Us (0+2)</span>
                <span className="stat-value" style={{ color: 'var(--accent-green)', fontSize: '1.2rem' }}>{gpScoreUs} GP</span>
              </div>
              <div className="stat-row" style={{ border: 'none' }}>
                <span className="stat-label">Final Score Them (1+3)</span>
                <span className="stat-value" style={{ color: 'var(--accent-red)', fontSize: '1.2rem' }}>{gpScoreThem} GP</span>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '20px' }}>
              <button 
                type="button"
                className="btn-primary" 
                onClick={() => startNewGame(undefined, true)}
              >
                Start New Match
              </button>
            </div>
          </div>
        </div>
      )}

      {/* GAME OVER DIALOG OVERLAY */}
      {showGameOverModal && (
        <div className="overlay-modal">
          <div className="modal-content" style={{ minWidth: '400px' }}>
            <div className="trump-indicator" style={{ marginBottom: '15px' }}>
              <span className="card-center-icon neon-glow-gold" style={{ fontSize: '3.5rem' }}>★</span>
            </div>
            
            {gpScoreUs >= gpLimit ? (
              <h1 className="winner-banner">Game Won!</h1>
            ) : (
              <h1 className="winner-banner lost">Game Lost</h1>
            )}

            <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.9rem', marginBottom: '20px' }}>
              Final Score: Us <span style={{color:'var(--accent-green)'}}>{gpScoreUs}</span> - <span style={{color:'var(--accent-red)'}}>{gpScoreThem}</span> Them
            </p>

            <div className="game-over-stats" style={{ maxHeight: '300px', overflowY: 'auto', marginBottom: '20px' }}>
              <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                    <th style={{ padding: '8px' }}>Deal</th>
                    <th style={{ padding: '8px', color: 'var(--accent-green)' }}>Us Pts</th>
                    <th style={{ padding: '8px', color: 'var(--accent-red)' }}>Them Pts</th>
                    <th style={{ padding: '8px' }}>GP</th>
                  </tr>
                </thead>
                <tbody>
                  {dealHistory.map((deal, idx) => (
                    <tr key={deal.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '8px' }}>#{idx + 1}</td>
                      <td style={{ padding: '8px' }}>{deal.usPts}</td>
                      <td style={{ padding: '8px' }}>{deal.themPts}</td>
                      <td style={{ padding: '8px' }}>
                        {deal.usGP > 0 ? <span style={{color:'var(--accent-green)'}}>+{deal.usGP}</span> : deal.themGP > 0 ? <span style={{color:'var(--accent-red)'}}>+{deal.themGP}</span> : '0'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <button 
              type="button"
              className="btn-primary" 
              onClick={() => startNewGame(undefined, true)}
              style={{ width: '100%' }}
            >
              Play Again
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
export default App;

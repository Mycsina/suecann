/* tslint:disable */
/* eslint-disable */

export class WannSuecaGameSession {
    free(): void;
    [Symbol.dispose](): void;
    current_player(): number;
    /**
     * Returns real-time WANN evaluation data for the inspector panel.
     * Encodes belief state for the given seat, runs the appropriate brain
     * (lead or follow), and returns beliefs, averaged outputs, and
     * per-node activations as a JSON string.
     */
    get_realtime_bot_eval(seat_idx: number): string;
    get_state_json(): string;
    is_game_over(): boolean;
    constructor(genome_json: string, seed: bigint);
    play_bot_turn(): number;
    play_player_card(card: number): void;
    set_bot_types(bot1: number, bot2: number, bot3: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wannsuecagamesession_free: (a: number, b: number) => void;
    readonly wannsuecagamesession_current_player: (a: number) => number;
    readonly wannsuecagamesession_get_realtime_bot_eval: (a: number, b: number) => [number, number, number, number];
    readonly wannsuecagamesession_get_state_json: (a: number) => [number, number];
    readonly wannsuecagamesession_is_game_over: (a: number) => number;
    readonly wannsuecagamesession_new: (a: number, b: number, c: bigint) => [number, number, number];
    readonly wannsuecagamesession_play_bot_turn: (a: number) => [number, number, number];
    readonly wannsuecagamesession_play_player_card: (a: number, b: number) => [number, number];
    readonly wannsuecagamesession_set_bot_types: (a: number, b: number, c: number, d: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

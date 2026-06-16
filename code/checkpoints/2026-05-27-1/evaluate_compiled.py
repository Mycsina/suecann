import numpy as np

def evaluate_compiled(belief: np.ndarray, W: float) -> np.ndarray:
    # Helpers
    def NOT(x): return 1.0 - np.clip(x, 0.0, 1.0)
    def THRESHOLD(x, w):
        if w >= 0:
            return 1.0 if x > 0.5 * w else 0.0
        else:
            return 1.0 if x < 0.5 * w else 0.0
    def SUM(*args): return sum(args)
    def AND(*args): return min(args) if args else 0.0
    def OR(*args): return max(args) if args else 0.0
    def CLAMP(x): return np.clip(x, 0.0, 1.0)

    # Inputs mapping
    Has_Led_Suit = float(belief[0])
    Has_Trump = float(belief[1])
    Led_Suit_Power = float(belief[2])
    Trump_Power = float(belief[3])
    Hand_Point_Density = float(belief[4])
    Am_I_Leading = float(belief[5])
    Am_I_Last_To_Play = float(belief[6])
    Is_Partner_Winning = float(belief[7])
    Trick_Point_Value = float(belief[8])
    Has_Trick_Been_Cut = float(belief[9])
    Partner_Void_Led = float(belief[10])
    Partner_Void_Trump = float(belief[11])
    Any_Opp_Void_Led = float(belief[12])
    Any_Opp_Void_Trump = float(belief[13])
    Led_Suit_Ace_Played = float(belief[14])
    Led_Suit_7_Played = float(belief[15])
    Trump_Ace_Played = float(belief[16])
    Game_Pts_Remaining = float(belief[17])
    Trick_Number = float(belief[18])
    Trumps_Remaining = float(belief[19])
    Score_Delta = float(belief[20])
    BIAS = 1.0

    # Hidden node logic
    hidden_26 = THRESHOLD(SUM((Trumps_Remaining * W)), W)
    hidden_38 = CLAMP(SUM((NOT(Is_Partner_Winning) * W), (NOT(Trick_Point_Value) * W)))
    hidden_39 = THRESHOLD(OR((Trick_Number * W)), W)
    hidden_42 = NOT(OR((Has_Trick_Been_Cut * W)))
    hidden_44 = CLAMP(AND((NOT(Am_I_Last_To_Play) * W)))
    hidden_45 = THRESHOLD(AND((hidden_39 * W)), W)
    hidden_48 = CLAMP(AND((Am_I_Last_To_Play * W)))
    hidden_49 = THRESHOLD(AND((Trick_Point_Value * W)), W)
    hidden_50 = CLAMP(OR((Trump_Power * W)))
    hidden_51 = CLAMP(SUM((Game_Pts_Remaining * W)))
    hidden_54 = CLAMP(OR((Is_Partner_Winning * W)))
    hidden_55 = NOT(AND((Hand_Point_Density * W)))
    hidden_47 = THRESHOLD(AND((NOT(hidden_55) * W)), W)
    hidden_30 = CLAMP(AND((hidden_47 * W)))
    hidden_40 = CLAMP(AND((hidden_30 * W)))
    hidden_31 = NOT(OR((hidden_26 * W), (NOT(hidden_38) * W)))
    hidden_28 = THRESHOLD(OR((NOT(Trump_Power) * W), (NOT(hidden_31) * W), (hidden_42 * W)), W)
    hidden_37 = NOT(SUM((hidden_28 * W), (hidden_45 * W)))
    hidden_35 = NOT(AND((hidden_26 * W), (hidden_37 * W)))
    hidden_27 = NOT(AND((hidden_35 * W)))
    hidden_41 = THRESHOLD(SUM((hidden_27 * W)), W)
    hidden_32 = CLAMP(SUM((Led_Suit_Power * W)))
    hidden_53 = THRESHOLD(SUM((hidden_32 * W)), W)
    hidden_29 = CLAMP(OR((NOT(hidden_53) * W)))
    hidden_46 = NOT(AND((hidden_29 * W)))
    hidden_43 = THRESHOLD(AND((hidden_46 * W), (hidden_50 * W), (hidden_51 * W)), W)
    hidden_52 = THRESHOLD(OR((NOT(hidden_43) * W)), W)
    hidden_36 = CLAMP(OR((NOT(hidden_40) * W), (NOT(hidden_44) * W), (NOT(hidden_52) * W)))
    hidden_33 = CLAMP(SUM((hidden_29 * W)))
    hidden_34 = NOT(SUM((NOT(hidden_36) * W), (NOT(hidden_54) * W)))

    # Output node logic
    MAX_FORCE = CLAMP(SUM((Trick_Point_Value * W), (Has_Led_Suit * W), (NOT(hidden_41) * W)))
    MIN_FORCE = CLAMP(SUM((NOT(Led_Suit_Power) * W), (hidden_31 * W)))
    EFFICIENT_WIN = CLAMP(SUM((NOT(Is_Partner_Winning) * W), (Trick_Point_Value * W), (NOT(hidden_33) * W)))
    EQUITY_BUILDER = CLAMP(SUM((Has_Trump * W), (NOT(hidden_28) * W), (hidden_34 * W), (NOT(hidden_48) * W), (hidden_49 * W)))

    return np.array([
        MAX_FORCE,
        MIN_FORCE,
        EFFICIENT_WIN,
        EQUITY_BUILDER,
    ], dtype=np.float64)
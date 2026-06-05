import React from 'react';

export interface CardProps {
  card: number;
  disabled?: boolean;
  isPlayable?: boolean;
  onClick?: () => void;
  throwAnimSeat?: number; // 0..3 to trigger throw animation from seat
  isFlipped?: boolean; // if false, shows back of card
  style?: React.CSSProperties;
}

const SUITS = [
  { name: 'Hearts', symbol: '♥', colorClass: 'neon-glow-red', styleColor: '#ff2a5f' },
  { name: 'Diamonds', symbol: '♦', colorClass: 'neon-glow-gold', styleColor: '#ffb800' },
  { name: 'Clubs', symbol: '♣', colorClass: 'neon-glow-green', styleColor: '#00f5d4' },
  { name: 'Spades', symbol: '♠', colorClass: 'neon-glow-cyan', styleColor: '#00bbf9' },
];

const RANKS = [
  { label: '2', points: 0 },
  { label: '3', points: 0 },
  { label: '4', points: 0 },
  { label: '5', points: 0 },
  { label: '6', points: 0 },
  { label: 'Q', points: 2 },
  { label: 'J', points: 3 },
  { label: 'K', points: 4 },
  { label: '7', points: 10 },
  { label: 'A', points: 11 },
];

export const Card: React.FC<CardProps> = ({
  card,
  disabled = false,
  isPlayable = true,
  onClick,
  throwAnimSeat,
  isFlipped = true,
  style,
}) => {
  const suitIndex = Math.floor(card / 10);
  const rankIndex = card % 10;
  
  const suit = SUITS[suitIndex];
  const rank = RANKS[rankIndex];

  const handleCardClick = () => {
    if (!disabled && isPlayable && onClick) {
      onClick();
    }
  };

  const throwClass = throwAnimSeat !== undefined ? `throw-anim-${throwAnimSeat}` : '';
  const disableClass = disabled || !isPlayable ? 'disabled' : '';

  if (!isFlipped) {
    return (
      <div className={`card-container ${throwClass}`} style={style}>
        <div className="card-inner" style={{ transform: 'rotateY(180deg)' }}>
          <div className="card-face card-back" />
        </div>
      </div>
    );
  }

  return (
    // eslint-disable-next-line react-doctor/no-static-element-interactions, react-doctor/click-events-have-key-events
    <div 
      className={`card-container ${disableClass} ${throwClass}`}
      onClick={handleCardClick}
      style={style}
    >
      <div className="card-inner">
        <div className="card-face card-front" style={{ borderColor: `${suit.styleColor}33` }}>
          <div className={`card-corner top-left ${suit.colorClass}`}>
            <span className="card-value">{rank.label}</span>
            <span className="card-suit-small">{suit.symbol}</span>
          </div>
          
          <div className={`card-center-icon ${suit.colorClass}`}>
            {suit.symbol}
          </div>
          
          <div className={`card-corner bottom-right ${suit.colorClass}`}>
            <span className="card-value">{rank.label}</span>
            <span className="card-suit-small">{suit.symbol}</span>
          </div>

          {rank.points > 0 && (
            <div className="card-points-badge">
              +{rank.points} PTS
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
export default Card;

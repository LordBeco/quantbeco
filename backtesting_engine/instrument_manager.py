"""
Instrument Manager

Manages instrument specifications and pip calculations for various trading instruments.
"""

from typing import Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class InstrumentSpec:
    """Instrument specifications"""
    name: str
    pip_value: float
    min_lot_size: float
    max_leverage: float
    commission_per_lot: float
    typical_spread: float


class InstrumentManager:
    """Manages instrument specifications and pip calculations"""
    
    def __init__(self):
        """Initialize with default instrument specifications"""
        self._instruments: Dict[str, InstrumentSpec] = {}
        self._load_default_instruments()
    
    def _load_default_instruments(self):
        """Load default instrument specifications"""
        # Major Forex Pairs
        self._instruments.update({
            'EURUSD': InstrumentSpec('EURUSD', 10.0, 0.01, 100, 7.0, 1.5),
            'GBPUSD': InstrumentSpec('GBPUSD', 10.0, 0.01, 100, 7.0, 2.0),
            'USDJPY': InstrumentSpec('USDJPY', 0.09, 0.01, 100, 7.0, 1.8),
            'USDCHF': InstrumentSpec('USDCHF', 10.0, 0.01, 100, 7.0, 2.2),
            'AUDUSD': InstrumentSpec('AUDUSD', 10.0, 0.01, 100, 7.0, 1.9),
            'USDCAD': InstrumentSpec('USDCAD', 7.5, 0.01, 100, 7.0, 2.1),
            'NZDUSD': InstrumentSpec('NZDUSD', 10.0, 0.01, 100, 7.0, 2.5),
        })
        
        # Minor Forex Pairs
        self._instruments.update({
            'EURGBP': InstrumentSpec('EURGBP', 12.5, 0.01, 100, 7.0, 2.8),
            'EURJPY': InstrumentSpec('EURJPY', 0.09, 0.01, 100, 7.0, 2.5),
            'GBPJPY': InstrumentSpec('GBPJPY', 0.09, 0.01, 100, 7.0, 3.2),
            'EURCHF': InstrumentSpec('EURCHF', 10.0, 0.01, 100, 7.0, 3.0),
            'GBPCHF': InstrumentSpec('GBPCHF', 10.0, 0.01, 100, 7.0, 3.5),
            'AUDCAD': InstrumentSpec('AUDCAD', 7.5, 0.01, 100, 7.0, 3.8),
        })
        
        # Major Indices
        self._instruments.update({
            'SPX500': InstrumentSpec('SPX500', 1.0, 0.1, 20, 0.5, 0.4),
            'NAS100': InstrumentSpec('NAS100', 1.0, 0.1, 20, 0.5, 1.0),
            'US30': InstrumentSpec('US30', 1.0, 0.1, 20, 0.5, 2.0),
            'GER40': InstrumentSpec('GER40', 1.0, 0.1, 20, 0.5, 1.5),
            'UK100': InstrumentSpec('UK100', 1.0, 0.1, 20, 0.5, 1.2),
            'JPN225': InstrumentSpec('JPN225', 1.0, 0.1, 10, 0.5, 8.0),
        })
        
        # Commodities
        self._instruments.update({
            'XAUUSD': InstrumentSpec('XAUUSD', 1.0, 0.01, 100, 0.5, 0.3),
            'XAGUSD': InstrumentSpec('XAGUSD', 50.0, 0.01, 100, 0.5, 0.03),
            'USOIL': InstrumentSpec('USOIL', 10.0, 0.01, 20, 0.05, 0.03),
            'UKOIL': InstrumentSpec('UKOIL', 10.0, 0.01, 20, 0.05, 0.04),
        })
        
        # Cryptocurrencies
        self._instruments.update({
            'BTCUSD': InstrumentSpec('BTCUSD', 1.0, 0.01, 2, 0.1, 10.0),
            'ETHUSD': InstrumentSpec('ETHUSD', 1.0, 0.01, 2, 0.1, 1.0),
            'LTCUSD': InstrumentSpec('LTCUSD', 1.0, 0.01, 2, 0.1, 0.5),
        })
    
    def get_pip_value(self, instrument: str) -> float:
        """Get pip value for specified instrument"""
        instrument = instrument.upper()
        if instrument in self._instruments:
            return self._instruments[instrument].pip_value
        else:
            # Default pip value for unknown instruments
            return 1.0
    
    def get_instrument_spec(self, instrument: str) -> Optional[InstrumentSpec]:
        """Get complete instrument specification"""
        instrument = instrument.upper()
        return self._instruments.get(instrument)
    
    def calculate_position_size(self, equity: float, risk_pct: float, 
                               stop_distance: float, instrument: str = None,
                               method: str = 'risk_based') -> float:
        """Calculate position size based on risk management rules
        
        Args:
            equity: Current account equity
            risk_pct: Risk percentage (e.g., 0.02 for 2%)
            stop_distance: Distance to stop loss in pips
            instrument: Trading instrument (for pip value calculation)
            method: Position sizing method ('fixed', 'risk_based', 'compounding')
        """
        if method == 'fixed':
            return 0.01  # Standard lot size
        
        elif method == 'risk_based':
            if stop_distance <= 0:
                return 0.01  # Default to minimum lot size
            
            # Calculate risk amount in account currency
            risk_amount = equity * risk_pct
            
            # Get pip value for the instrument
            pip_value = self.get_pip_value(instrument) if instrument else 10.0
            
            # Calculate position size
            position_size = risk_amount / (stop_distance * pip_value)
            
            # Apply minimum and maximum lot size constraints
            if instrument:
                spec = self.get_instrument_spec(instrument)
                if spec:
                    position_size = max(position_size, spec.min_lot_size)
                    # Apply reasonable maximum (e.g., 10 lots)
                    position_size = min(position_size, 10.0)
            
            return round(position_size, 2)
        
        elif method == 'compounding':
            # Kelly criterion-based position sizing
            # Simplified version: use risk-based with compounding factor
            base_size = self.calculate_position_size(equity, risk_pct, stop_distance, 
                                                   instrument, 'risk_based')
            
            # Apply compounding factor based on equity growth
            # Assume starting equity of 10,000 for scaling
            compounding_factor = max(1.0, equity / 10000.0)
            compounded_size = base_size * math.sqrt(compounding_factor)
            
            return round(compounded_size, 2)
        
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
    
    def add_custom_instrument(self, spec: InstrumentSpec):
        """Add custom instrument specification"""
        self._instruments[spec.name.upper()] = spec
    
    def get_available_instruments(self) -> Dict[str, InstrumentSpec]:
        """Get all available instruments"""
        return self._instruments.copy()
    
    def calculate_margin_required(self, instrument: str, position_size: float, 
                                 price: float) -> float:
        """Calculate margin required for a position"""
        spec = self.get_instrument_spec(instrument)
        if not spec:
            return 0.0
        
        # Calculate notional value
        notional_value = position_size * 100000 * price  # Assuming standard lot = 100,000 units
        
        # Calculate margin requirement
        margin_required = notional_value / spec.max_leverage
        
        return margin_required
    
    def calculate_commission(self, instrument: str, position_size: float) -> float:
        """Calculate commission for a trade"""
        spec = self.get_instrument_spec(instrument)
        if not spec:
            return 0.0
        
        return position_size * spec.commission_per_lot
    
    def get_typical_spread(self, instrument: str) -> float:
        """Get typical spread for an instrument"""
        spec = self.get_instrument_spec(instrument)
        if not spec:
            return 2.0  # Default spread
        
        return spec.typical_spread
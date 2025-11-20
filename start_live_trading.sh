#!/bin/bash
# Start the live trading system with enhanced GEX LSTM model

# Ensure we're in the project root directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Live Trading System with Enhanced GEX Model"
echo "======================================================"
echo ""
echo "This system will:"
echo "  • Consume live tick data from Redis (TastyTrade/Schwab)"
echo "  • Fetch real-time GEX updates every second"
echo "  • Make trading decisions using enhanced LSTM model"
echo "  • Send JSON trade updates to API endpoint (no real trades)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Default parameters
MODEL_PATH="ml/models/enhanced_gex_model.pt"
SYMBOL="MNQ"
THRESHOLD="0.3"
MAX_POSITION="1"
COMMISSION="0.42"
API_ENDPOINT="http://localhost:8877/ml-trade"

# Allow overriding with environment variables
MODEL_PATH="${LIVE_MODEL_PATH:-$MODEL_PATH}"
SYMBOL="${LIVE_SYMBOL:-$SYMBOL}"
THRESHOLD="${LIVE_THRESHOLD:-$THRESHOLD}"
MAX_POSITION="${LIVE_MAX_POSITION:-$MAX_POSITION}"
COMMISSION="${LIVE_COMMISSION:-$COMMISSION}"
API_ENDPOINT="${LIVE_API_ENDPOINT:-$API_ENDPOINT}"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Symbol: $SYMBOL"
echo "  Threshold: $THRESHOLD"
echo "  Max Position: $MAX_POSITION"
echo "  Commission: $COMMISSION"
echo "  API Endpoint: $API_ENDPOINT"
echo ""

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "⚠️  Warning: Redis server not detected. Make sure Redis is running:"
    echo "   redis-server redis/redis.conf &"
    echo ""
fi

# Check if data pipeline is running
if ! pgrep -f "data-pipeline.py" > /dev/null; then
    echo "⚠️  Warning: Data pipeline not detected. Make sure data sources are running:"
    echo "   python data-pipeline.py --host 0.0.0.0 --port 8877"
    echo ""
fi

echo "Starting live trader..."
python ml/live_trader.py \
    --model "$MODEL_PATH" \
    --symbol "$SYMBOL" \
    --threshold "$THRESHOLD" \
    --max-position "$MAX_POSITION" \
    --commission "$COMMISSION" \
    --api-endpoint "$API_ENDPOINT"
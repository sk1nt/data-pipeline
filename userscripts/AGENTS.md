# Userscripts Agent Instructions

## Domain Scope
This folder contains browser userscripts (Tampermonkey/Greasemonkey) that capture data from external websites and forward to the data pipeline.

## Current Scripts

### `uw-phoenix-capture.user.js`
Captures Unusual Whales Phoenix WebSocket data and forwards to pipeline.

**Flow:**
```
UW Website → Phoenix WebSocket → Userscript → POST /uw → Pipeline
```

**Subscribed Channels:**
- `market_agg_socket` - Market aggregation updates
- `option_trades_super_algo` - Options flow alerts
- `option_trades_super_algo:SPX` - SPX-specific options flow

**Target Endpoint:** `POST http://192.168.168.151:8877/uw`

## Message Format (Phoenix Protocol)
```javascript
// Incoming: [joinRef, ref, topic, eventType, payload]
["1", "1", "market_agg_socket", "update", { data: {...} }]

// Forwarded as-is to /uw endpoint
```

## Development Guidelines

### Userscript Header
```javascript
// ==UserScript==
// @name         Script Name
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  Description
// @match        https://target-site.com/*
// @grant        none
// @run-at       document-start
// ==/UserScript==
```

### Key Patterns

**WebSocket Interception:**
```javascript
const OriginalWebSocket = window.WebSocket;
window.WebSocket = function(url, protocols) {
    // Intercept and capture token/data
    return new OriginalWebSocket(url, protocols);
};
```

**Overlay for Debugging:**
```javascript
function updateOverlay(status, payload) {
    // Show connection status and last payload on page
}
```

**Forward to Pipeline:**
```javascript
fetch('http://YOUR_PIPELINE_HOST:8877/uw', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(rawMsg)
});
```

## Configuration

Update the target endpoint in the script:
```javascript
// Line ~65: Update IP/port for your environment
fetch('http://192.168.168.151:8877/uw', { ... })
```

## Testing

1. Install Tampermonkey browser extension
2. Create new script, paste content
3. Navigate to https://unusualwhales.com
4. Open DevTools Console to see logs
5. Check overlay in bottom-right corner

## Backend Integration

The `/uw` endpoint is defined in:
- `data-pipeline.py` → `universal_webhook()` function
- Calls `src/services/uw_message_service.py`

## Related Issues
- Issue #18: [Epic] Unusual Whales Message Micro Agent
- Issue #19: UW Alert Rule Engine

## Do NOT
- Commit tokens or auth credentials
- Forward raw trade_ids arrays (strip for privacy)
- Spam the endpoint (respect rate limits)

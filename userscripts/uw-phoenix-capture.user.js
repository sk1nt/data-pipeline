// ==UserScript==
// @name         UW Phoenix Capture v3.0 - Sweep Detection + Extended Channels
// @namespace    http://tampermonkey.net/
// @version      3.0
// @description  Intercept UW Phoenix websockets, pre-tag sweeps, forward option_trades + market_agg + dark_pool messages
// @match        https://unusualwhales.com/*
// @grant        none
// @run-at       document-start
// ==/UserScript==

(function () {
    'use strict';

    // ── Config ──────────────────────────────────────────────────────────────
    const CONFIG = {
        PIPELINE_HOST: localStorage.getItem('uw_pipeline_host') || '192.168.168.151',
        PIPELINE_PORT: localStorage.getItem('uw_pipeline_port') || '8877',
        MIN_SWEEP_PREMIUM: 50000,   // pre-filter below $50k before forwarding
        DEBUG: false,
    };

    // ── Channels to subscribe ────────────────────────────────────────────────
    // option_trades_super_algo:SYMBOL channels give per-symbol filtered streams
    // dark_pool_socket may or may not be available on the server — failure is silent
    const CHANNELS = [
        'option_trades_super_algo',         // all stocks
        'option_trades_super_algo:SPX',     // SPX index
        'option_trades_super_algo:NDX',     // NDX index
        'option_trades_super_algo:QQQ',     // QQQ ETF
        'market_agg_socket',                // market-wide aggregation
        'dark_pool_socket',                 // dark pool prints (may 404)
    ];

    const ENDPOINT = `http://${CONFIG.PIPELINE_HOST}:${CONFIG.PIPELINE_PORT}/uw`;

    // ── State ────────────────────────────────────────────────────────────────
    const OriginalWebSocket = window.WebSocket;
    const trackedSockets = new Set();
    const counters = {};
    CHANNELS.forEach(c => { counters[c] = 0; });
    counters['sockets_intercepted'] = 0;
    counters['sweeps_detected'] = 0;
    counters['filtered_low_premium'] = 0;

    // ── Sweep detection ──────────────────────────────────────────────────────
    function detectSweep(data) {
        if (!data || typeof data !== 'object') return false;
        const askVol = data.ask_vol || 0;
        const bidVol = data.bid_vol || 0;
        const total = askVol + bidVol;
        const askRatio = total > 0 ? askVol / total : 0;
        const tags = data.tags || [];
        const premium = parseFloat(data.premium || 0);

        // Sweep conditions (mirrors the API's has_sweep logic):
        //   1. UW already tagged it bullish/bearish on ask-side
        //   2. Ask-side dominant (>= 75% of volume)
        //   3. Premium above threshold
        const taggedAskSide = tags.includes('ask_side') || tags.includes('bullish');
        return taggedAskSide && askRatio >= 0.75 && premium >= CONFIG.MIN_SWEEP_PREMIUM;
    }

    function sweepConviction(data) {
        const premium = parseFloat(data.premium || 0);
        const askVol = data.ask_vol || 0;
        const bidVol = data.bid_vol || 0;
        const total = askVol + bidVol;
        const askRatio = total > 0 ? askVol / total : 0;
        if (premium >= 500000 && askRatio >= 0.85) return 'very_high';
        if (premium >= 250000 && askRatio >= 0.75) return 'high';
        if (premium >= 100000 && askRatio >= 0.65) return 'medium';
        return 'low';
    }

    // ── Message handler ──────────────────────────────────────────────────────
    function handlePhoenixMessage(event) {
        try {
            const msg = JSON.parse(event.data);
            if (!Array.isArray(msg) || msg.length !== 5) return;

            const [joinRef, ref, topic, eventType, payload] = msg;

            // Skip Phoenix protocol messages
            if (eventType.startsWith('phx_')) return;
            if (!payload || typeof payload !== 'object' || Object.keys(payload).length === 0) return;

            // ── dark_pool_socket ─────────────────────────────────────────────
            if (topic === 'dark_pool_socket') {
                if (CONFIG.DEBUG) console.log('[UW-CAPTURE] dark_pool_socket', eventType);
                forwardRawMsg('dark_pool_socket', topic, eventType, msg);
                return;
            }

            // ── market_agg_socket ────────────────────────────────────────────
            if (topic === 'market_agg_socket') {
                if (CONFIG.DEBUG) console.log('[UW-CAPTURE] market_agg_socket', eventType);
                forwardRawMsg('market_agg_socket', topic, eventType, msg);
                return;
            }

            // ── option_trades_super_algo (all variants) ──────────────────────
            if (topic.startsWith('option_trades_super_algo')) {
                const data = payload.data || {};

                // Remove trade_ids (privacy/size)
                if (data.is_agg === true && data.trade_ids) {
                    delete data.trade_ids;
                }

                const premium = parseFloat(data.premium || 0);

                // Pre-filter very small trades before network hop
                if (premium > 0 && premium < CONFIG.MIN_SWEEP_PREMIUM) {
                    counters['filtered_low_premium']++;
                    return;
                }

                // Enrich with computed sweep fields
                const isSweep = detectSweep(data);
                const askVol = data.ask_vol || 0;
                const bidVol = data.bid_vol || 0;
                const total = askVol + bidVol;
                payload._computed = {
                    is_sweep: isSweep,
                    sweep_conviction: isSweep ? sweepConviction(data) : 'none',
                    ask_vol_ratio: total > 0 ? Math.round((askVol / total) * 1000) / 1000 : null,
                    captured_at: new Date().toISOString(),
                };

                if (isSweep) counters['sweeps_detected']++;

                const endpoint = topic.includes(':') ? topic : 'option_trades_super_algo';
                if (CONFIG.DEBUG) console.log('[UW-CAPTURE] option_trade', topic, 'sweep=', isSweep);
                forwardRawMsg(endpoint, topic, eventType, msg);
                return;
            }

        } catch (e) {
            // Ignore parse errors — not all messages are JSON
        }
    }

    // ── Socket attachment ────────────────────────────────────────────────────
    function attachToSocket(socket, url) {
        if (trackedSockets.has(socket)) return;
        trackedSockets.add(socket);
        counters.sockets_intercepted++;
        if (CONFIG.DEBUG) console.log(`[UW-CAPTURE] Intercepted socket #${counters.sockets_intercepted}: ${url.substring(0, 80)}`);
        socket.addEventListener('message', handlePhoenixMessage);

        // Subscribe to all configured channels once connected
        socket.addEventListener('open', () => {
            let refCounter = 0;

            // Heartbeat
            setInterval(() => {
                refCounter++;
                socket.send(JSON.stringify([null, String(refCounter), 'phoenix', 'heartbeat', {}]));
            }, 30000);

            // Join all channels
            CHANNELS.forEach(channel => {
                refCounter++;
                const ref = String(refCounter);
                socket.send(JSON.stringify([ref, ref, channel, 'phx_join', {}]));
                if (CONFIG.DEBUG) console.log(`[UW-CAPTURE] Joined channel: ${channel}`);
            });

            updateOverlay('Connected — subscribed to ' + CHANNELS.length + ' channels', null);
        });

        socket.addEventListener('close', () => {
            updateOverlay('Socket closed — waiting for reconnect', null);
        });
    }

    // ── WebSocket wrapper ────────────────────────────────────────────────────
    window.WebSocket = function (url, protocols) {
        const socket = protocols ? new OriginalWebSocket(url, protocols) : new OriginalWebSocket(url);
        if (typeof url === 'string' && url.includes('unusualwhales.com')) {
            attachToSocket(socket, url);
        }
        return socket;
    };
    window.WebSocket.prototype = OriginalWebSocket.prototype;
    window.WebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
    window.WebSocket.OPEN = OriginalWebSocket.OPEN;
    window.WebSocket.CLOSING = OriginalWebSocket.CLOSING;
    window.WebSocket.CLOSED = OriginalWebSocket.CLOSED;

    // ── Forward to pipeline ──────────────────────────────────────────────────
    function forwardRawMsg(endpoint, topic, eventType, rawMsg) {
        fetch(ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rawMsg),
        })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.text();
            })
            .then(() => {
                counters[endpoint] = (counters[endpoint] || 0) + 1;
                updateOverlay(`✅ ${endpoint} → ${eventType}`, null);
            })
            .catch(err => {
                updateOverlay(`❌ ${endpoint}: ${err.message}`, null);
            });
    }

    // ── Overlay UI ───────────────────────────────────────────────────────────
    const OVERLAY_ID = 'uw-capture-overlay';
    const OVERLAY_STYLE = `
        position: fixed; bottom: 10px; right: 10px;
        background: rgba(0,0,0,0.88); color: #fff;
        font: 11px monospace; padding: 8px 12px;
        border-radius: 6px; z-index: 99999;
        box-shadow: 0 0 8px rgba(0,0,0,0.5);
        max-width: 360px; max-height: 260px;
        overflow-y: auto; white-space: pre-wrap;
    `;

    function ensureOverlay() {
        let el = document.getElementById(OVERLAY_ID);
        if (!el) {
            el = document.createElement('div');
            el.id = OVERLAY_ID;
            el.setAttribute('style', OVERLAY_STYLE);
            const append = () => document.body && document.body.appendChild(el);
            document.body ? append() : document.addEventListener('DOMContentLoaded', append, { once: true });
        }
        return el;
    }

    function updateOverlay(status, _payload) {
        const el = ensureOverlay();
        let display = `UW Capture v3.0\n${status}\n\n`;
        for (const [k, v] of Object.entries(counters)) {
            display += `• ${k}: ${v}\n`;
        }
        el.textContent = display;
    }

    updateOverlay('Waiting for UW WebSocket...', null);
    console.log('[UW-CAPTURE] Script loaded v3.0 — extended channels + sweep detection');
})();


(function () {
    console.log('Script loaded');
    let capturedToken = null;
    const OVERLAY_ID = 'uw-metric-overlay';
    const OVERLAY_STYLE = `
    position: fixed;
    bottom: 10px;
    right: 10px;
    background: rgba(0,0,0,0.85);
    color: #fff;
    font: 12px monospace;
    padding: 8px 12px;
    border-radius: 6px;
    z-index: 99999;
    box-shadow: 0 0 8px rgba(0,0,0,0.5);
    max-width: 400px;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
  `;

    // Per-topic counters
    const counters = {
        "market_agg_socket": 0,
        "option_trades_super_algo": 0,
        "option_trades_super_algo:SPX": 0
    };

    function ensureOverlay() {
        let el = document.getElementById(OVERLAY_ID);
        if (!el) {
            el = document.createElement('div');
            el.id = OVERLAY_ID;
            el.setAttribute('style', OVERLAY_STYLE);
            if (document.body) {
                document.body.appendChild(el);
                console.log('Overlay appended to body');
            } else {
                document.addEventListener('DOMContentLoaded', () => {
                    document.body.appendChild(el);
                    console.log('Overlay appended after DOMContentLoaded');
                }, { once: true });
            }
        }
        return el;
    }

    function updateOverlay(status, payload) {
        const el = ensureOverlay();
        let display = `${status}\n\nCounters:\n`;
        for (const [endpoint, count] of Object.entries(counters)) {
            display += `• ${endpoint}: ${count} frames\n`;
        }
        if (payload) {
            display += `\nLast Payload:\n${JSON.stringify(payload, null, 2)}`;
        }
        el.textContent = display;
    }

    function forwardRawMsg(endpoint, topic, eventType, rawMsg) {
        console.log("Sending to 8877:", rawMsg); // Debug log

        fetch('http://192.168.168.151:8877/uw', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rawMsg)
        })
            .then(res => {
                if (!res.ok) throw new Error('Network response not ok');
                return res.text(); // safer than res.json() if server returns empty
            })
            .then(() => {
                counters[endpoint] = (counters[endpoint] || 0) + 1;
                updateOverlay(`✅ Forwarded ${endpoint} → ${eventType}`, rawMsg);
            })
            .catch(err => {
                updateOverlay(`❌ Error forwarding ${endpoint} → ${eventType}: ${err.message}`, rawMsg);
            });
    }

    const OriginalWebSocket = window.WebSocket;
    window.WebSocket = function (url, protocols) {
        console.log('Opening WebSocket to:', url);
        if (url.includes('unusualwhales.com') && url.includes('token=')) {
            const match = url.match(/token=([^&]+)/);
            if (match) {
                capturedToken = match[1];
                console.log('Captured token:', capturedToken);
                setTimeout(() => startPhoenixSocket(capturedToken), 1000);
            }
        }
        return new OriginalWebSocket(url, protocols);
    };

    function startPhoenixSocket(token) {
        if (!token) return;
        console.log('Starting Phoenix socket');
        updateOverlay('Connecting to Phoenix...', null);
        const WS_URL = `wss://ws.unusualwhales.com/v2/foo/websocket?token=${token}&vsn=2.0.0`;
        const socket = new OriginalWebSocket(WS_URL);
        let refCounter = 0;
        let heartbeatInterval;

        socket.addEventListener('open', () => {
            console.log('Phoenix socket opened');
            updateOverlay('✅ Phoenix socket connected', null);
            heartbeatInterval = setInterval(() => {
                refCounter++;
                const heartbeatMsg = [null, refCounter.toString(), "phoenix", "heartbeat", {}];
                socket.send(JSON.stringify(heartbeatMsg));
                console.log('Sent heartbeat');
            }, 30000);

            const channels = ['option_trades_super_algo', 'option_trades_super_algo:SPX', 'market_agg_socket'];
            channels.forEach(channel => {
                refCounter++;
                const joinRef = refCounter.toString();
                const subscribeMsg = [joinRef, joinRef, channel, "phx_join", {}];
                socket.send(JSON.stringify(subscribeMsg));
                console.log('Sent subscribe to channel:', channel, 'with ref:', joinRef);
            });
        });

        socket.addEventListener('message', (event) => {
            console.log('Message received:', event.data);
            try {
                const msg = JSON.parse(event.data);
                if (Array.isArray(msg) && msg.length === 5) {
                    const [joinRef, ref, topic, eventType, payload] = msg;

                    console.log(`Debug: Topic: ${topic}, EventType: ${eventType}, Payload type: ${typeof payload}, Payload keys length: ${payload ? Object.keys(payload).length : 'n/a'}, Starts with option_trades_super_algo: ${topic.startsWith('option_trades_super_algo')}, Event starts with phx_: ${eventType.startsWith('phx_')}, Is tick: ${eventType === 'tick'}`);

                    if (eventType === 'phx_reply') {
                        const status = payload.status || 'unknown';
                        const response = payload.response || {};
                        console.log(`phx_reply for topic ${topic}, ref ${ref}: status=${status}`, response);
                        updateOverlay(`Join ${topic}: ${status}`, {response});
                    } else if ((topic === 'market_agg_socket' || topic.startsWith('option_trades_super_algo')) &&
                        !eventType.startsWith('phx_') &&
                        eventType !== 'tick' &&
                        payload && typeof payload === 'object' &&
                        Object.keys(payload).length > 0) {

                        console.log('Condition met - forwarding message');

                        // Check if it's an aggregated trade and remove trade_ids if present
                        if (payload.data && payload.data.is_agg === true && payload.data.trade_ids) {
                            delete payload.data.trade_ids;
                        }

                        let endpoint = topic;
                        if (topic.startsWith('option_trades_super_algo') && !topic.includes(':')) {
                            endpoint = 'option_trades_super_algo';
                        } else if (topic === 'option_trades_super_algo:SPX') {
                            endpoint = 'option_trades_super_algo:SPX';
                        }

                        forwardRawMsg(endpoint, topic, eventType, msg);
                    } else {
                        console.log(`Message not forwarded: topic=${topic}, eventType=${eventType}`);
                    }
                }
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        });

        socket.addEventListener('close', () => {
            console.log('Phoenix socket closed');
            if (heartbeatInterval) clearInterval(heartbeatInterval);
            updateOverlay('❌ Phoenix socket closed', null);
        });

        socket.addEventListener('error', (err) => {
            console.error('Phoenix socket error:', err);
            if (heartbeatInterval) clearInterval(heartbeatInterval);
            updateOverlay('❌ Phoenix socket error', null);
        });
    }
})();
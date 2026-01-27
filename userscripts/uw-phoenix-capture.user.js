// ==UserScript==
// @name         UW Phoenix Auto-Dump + Overlay (Auto-Token) - no dedup (patched)
// @namespace    http://tampermonkey.net/
// @version      2.5
// @description  Capture Phoenix token, subscribe, forward all payloads (no dedup) with overlay showing last payload + counters
// @match        https://unusualwhales.com/*
// @grant        none
// @run-at       document-start
// ==/UserScript==

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
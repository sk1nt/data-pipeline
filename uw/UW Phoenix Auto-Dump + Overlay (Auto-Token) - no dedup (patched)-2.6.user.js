// ==UserScript==
// @name         UW Phoenix Auto-Dump + Overlay (Auto-Token) - no dedup (patched)
// @namespace    http://tampermonkey.net/
// @version      2.8
// @description  Intercept UW Phoenix websockets, forward option_trades_super_algo and market_agg_socket messages
// @match        https://unusualwhales.com/*
// @grant        none
// @run-at       document-start
// ==/UserScript==

(function () {
    'use strict';
    
    // Wrap WebSocket IMMEDIATELY before any other code runs
    const OriginalWebSocket = window.WebSocket;
    const trackedSockets = new Set();
    
    const counters = {
        "option_trades_super_algo": 0,
        "option_trades_super_algo:SPX": 0,
        "market_agg_socket": 0,
        "sockets_intercepted": 0
    };

    function handlePhoenixMessage(event) {
        try {
            const msg = JSON.parse(event.data);
            
            if (!Array.isArray(msg) || msg.length !== 5) {
                return;
            }

            const [joinRef, ref, topic, eventType, payload] = msg;

            // Skip Phoenix protocol messages
            if (eventType.startsWith('phx_')) {
                return;
            }

            // Skip empty payloads
            if (!payload || typeof payload !== 'object' || Object.keys(payload).length === 0) {
                return;
            }

            // Forward market_agg_socket messages
            if (topic === 'market_agg_socket') {
                console.log("[UW-CAPTURE] MATCH: market_agg_socket", eventType);
                forwardRawMsg('market_agg_socket', topic, eventType, msg);
                return;
            }

            // Forward option_trades_super_algo messages (including :SPX variant)
            if (topic.startsWith('option_trades_super_algo')) {
                console.log("[UW-CAPTURE] MATCH: option_trades_super_algo", eventType);

                // Remove trade_ids from aggregated trades (privacy/size)
                if (payload.data?.is_agg === true && payload.data?.trade_ids) {
                    delete payload.data.trade_ids;
                }

                let endpoint = topic.includes(':') ? topic : 'option_trades_super_algo';
                forwardRawMsg(endpoint, topic, eventType, msg);
                return;
            }

        } catch (e) {
            // Ignore parse errors - not all messages are JSON
        }
    }

    function attachToSocket(socket, url) {
        if (trackedSockets.has(socket)) return;
        trackedSockets.add(socket);
        
        counters.sockets_intercepted++;
        console.log(`[UW-CAPTURE] Intercepted UW WebSocket #${counters.sockets_intercepted}: ${url.substring(0, 80)}...`);
        
        socket.addEventListener('message', handlePhoenixMessage);
        updateOverlay(`Intercepted socket #${counters.sockets_intercepted}`, null);
    }

    // Replace WebSocket constructor
    window.WebSocket = function(url, protocols) {
        const socket = protocols 
            ? new OriginalWebSocket(url, protocols) 
            : new OriginalWebSocket(url);
        
        // Check if this is a UW Phoenix websocket
        if (typeof url === 'string' && url.includes('unusualwhales.com')) {
            attachToSocket(socket, url);
        }
        
        return socket;
    };
    
    // Copy static properties and prototype
    window.WebSocket.prototype = OriginalWebSocket.prototype;
    window.WebSocket.CONNECTING = OriginalWebSocket.CONNECTING;
    window.WebSocket.OPEN = OriginalWebSocket.OPEN;
    window.WebSocket.CLOSING = OriginalWebSocket.CLOSING;
    window.WebSocket.CLOSED = OriginalWebSocket.CLOSED;

    console.log('[UW-CAPTURE] Script loaded v2.8 - WebSocket wrapped');

    // === UI and forwarding code below ===
    
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

    function ensureOverlay() {
        let el = document.getElementById(OVERLAY_ID);
        if (!el) {
            el = document.createElement('div');
            el.id = OVERLAY_ID;
            el.setAttribute('style', OVERLAY_STYLE);
            if (document.body) {
                document.body.appendChild(el);
            } else {
                document.addEventListener('DOMContentLoaded', () => {
                    document.body.appendChild(el);
                }, { once: true });
            }
        }
        return el;
    }

    function updateOverlay(status, payload) {
        const el = ensureOverlay();
        let display = `${status}\n\nCounters:\n`;
        for (const [endpoint, count] of Object.entries(counters)) {
            display += `• ${endpoint}: ${count}\n`;
        }
        if (payload) {
            display += `\nLast Payload:\n${JSON.stringify(payload, null, 2)}`;
        }
        el.textContent = display;
    }

    function forwardRawMsg(endpoint, topic, eventType, rawMsg) {
        console.log("[UW-CAPTURE] Forwarding:", endpoint, eventType);

        fetch('http://192.168.168.151:8877/uw', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(rawMsg)
        })
            .then(res => {
                if (!res.ok) throw new Error('Network response not ok');
                return res.text();
            })
            .then(() => {
                counters[endpoint] = (counters[endpoint] || 0) + 1;
                updateOverlay(`✅ Forwarded ${endpoint} → ${eventType}`, rawMsg);
            })
            .catch(err => {
                updateOverlay(`❌ Error forwarding ${endpoint} → ${eventType}: ${err.message}`, rawMsg);
            });
    }

    updateOverlay('Waiting for UW WebSocket...', null);
})();

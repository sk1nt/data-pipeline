// Financial Tick Data Pipeline Monitor JavaScript

class PipelineMonitor {
    constructor() {
        this.apiBase = window.location.origin + '/api/v1';
        this.updateInterval = 5000; // 5 seconds
        this.init();
    }

    init() {
        this.loadStatus();
        this.loadDataSample();
        setInterval(() => {
            this.loadStatus();
            this.loadDataSample();
        }, this.updateInterval);
    }

    async loadStatus() {
        try {
            const response = await fetch(`${this.apiBase}/status`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.updateStatusDisplay(data);
            this.updateTimestamp(data.timestamp);
        } catch (error) {
            console.error('Failed to load status:', error);
            this.showError('services-grid', 'Failed to load service status');
        }
    }

    async loadDataSample() {
        try {
            // Try to get a small sample of real-time data
            const response = await fetch(`${this.apiBase}/ticks/realtime?symbols=AAPL&limit=5`);
            if (response.status === 401) {
                this.updateDataDisplay('Authentication required to view data samples');
                return;
            }

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.updateDataDisplay(data);
        } catch (error) {
            console.error('Failed to load data sample:', error);
            this.updateDataDisplay('Unable to load data sample (authentication may be required)');
        }
    }

    updateStatusDisplay(data) {
        const grid = document.getElementById('services-grid');
        grid.innerHTML = '';

        data.services.forEach(service => {
            const card = document.createElement('div');
            card.className = 'service-card';

            const statusClass = `status-${service.current_status}`;

            card.innerHTML = `
                <div class="service-name">${service.service_name}</div>
                <div class="service-status ${statusClass}">${service.current_status}</div>
                <div class="uptime">Uptime: ${service.uptime_percentage}%</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                    Last update: ${new Date(service.last_update_time).toLocaleString()}
                </div>
            `;

            grid.appendChild(card);
        });
    }

    updateDataDisplay(data) {
        const display = document.getElementById('data-display');

        if (typeof data === 'string') {
            display.textContent = data;
            return;
        }

        if (data.data && data.data.length > 0) {
            let content = `Latest ${data.data.length} tick(s):\n\n`;
            data.data.forEach((tick, index) => {
                content += `${index + 1}. ${tick.symbol} @ ${tick.price} (${tick.tick_type})\n`;
                content += `   Time: ${new Date(tick.timestamp).toLocaleString()}\n`;
                if (tick.volume) {
                    content += `   Volume: ${tick.volume}\n`;
                }
                content += '\n';
            });

            if (data.metadata) {
                content += `Response time: ${data.metadata.response_time_ms}ms\n`;
                content += `Cache hit: ${data.metadata.cache_hit}`;
            }

            display.textContent = content;
        } else {
            display.textContent = 'No recent tick data available';
        }
    }

    updateTimestamp(timestamp) {
        const timeElement = document.getElementById('update-time');
        timeElement.textContent = new Date(timestamp).toLocaleString();
    }

    showError(elementId, message) {
        const element = document.getElementById(elementId);
        element.innerHTML = `<div class="error">${message}</div>`;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PipelineMonitor();
});
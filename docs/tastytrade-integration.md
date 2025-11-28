# Tastytrade Integration Documentation

## Overview
This document describes the integration between the data pipeline and Tastytrade for automated order placement.

## Features
- **Futures Orders**: Manual orders via Discord !tt commands
- **Options Orders**: Automated orders triggered by Discord alerts
- **Sandbox Testing**: All orders tested in sandbox before production
- **Error Handling**: Comprehensive validation and error responses

## Architecture
- **Discord Bot**: Handles chat commands and alerts
- **FastAPI Server**: REST API for order management
- **Tastytrade SDK**: API client for order placement
- **Redis**: Caching and session management
- **DuckDB**: Order history and metadata storage

## Configuration
Set the following environment variables:
- `DISCORD_TOKEN`: Discord bot token
- `TASTYTRADE_CLIENT_SECRET`: Tastytrade OAuth client secret
- `TASTYTRADE_REFRESH_TOKEN`: Tastytrade refresh token
- `TASTYTRADE_USE_SANDBOX`: true/false for environment
- `REDIS_URL`: Redis connection URL
- `DATABASE_URL`: DuckDB database URL

## API Endpoints
- `POST /api/futures/orders`: Place futures order
- `POST /api/options/orders`: Process options alert
- `DELETE /api/orders/{id}`: Cancel order
- `GET /health`: Health check

## Discord Commands
- `!tt buy <symbol> <tp_ticks> [quantity] [mode]`: Buy futures
- `!tt sell <symbol> <tp_ticks> [quantity] [mode]`: Sell futures
- `!tt flat <symbol> <tp_ticks> [quantity] [mode]`: Close futures position

## Alert Format
Options alerts must follow: `Alert: BTO/STC SYMBOL STRIKEc/p MM/DD @ PRICE`

## Security
- User ID verification for all operations
- Authorized users only for futures and alerts
- Environment separation (sandbox/production)

## Deployment
Use `docker-compose.yml` for containerized deployment with Redis.
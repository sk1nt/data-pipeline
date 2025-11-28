# Tastytrade Ordering Automation with Chat Interface

## Feature Description
Enable automated order placement through Tastytrade API, with a chat-based interface for initiating orders. All orders must be thoroughly tested in the Tastytrade sandbox environment before being switched to production.

## Clarifications

### Session 2025-11-27
- Q: What triggers the automated options ordering? → A: When specific text is seen in control channel or XXXXXXXXXXXXXXXXXX, allocate configured % of BP for options. If buy message in 1255265167113978008, allocate 1/2 of that %. Attempt fills at mid, then up 2-3 increments, else leave as day order. On fill, create limit exit for 1/2 at 100% profit.
- Q: How are traders authenticated before accepting chat-based orders? → A: By verifying Discord user ID; only 704125082750156840 can open futures trades with !tt commands, and alerts issued by 700068629626224700 or 704125082750156840.

## User Scenarios & Testing

### Primary User Scenario: Chat-Initiated Futures Order Placement
1. Authorized trader (user ID 704125082750156840) sends !tt buy/sell/flat command via chat
2. System validates user permissions and parses command parameters
3. System places the futures order in Tastytrade sandbox environment
4. System executes thorough testing of the order placement and execution
5. Upon successful testing, system switches the order configuration to production
6. Trader receives confirmation of order placement in production

### Secondary User Scenario: Automated Options Order Execution
1. System detects alert message in designated channels with format "Alert: BTO/CALLS/PUTS [SYMBOL] [STRIKE] [EXPIRY] @ [PRICE]" issued by authorized users (700068629626224700 or 704125082750156840)
2. System calculates order quantity based on configured allocation percentage of buying power (BP), using 1/2 allocation for buy messages in 1255265167113978008
3. System attempts to place options order at mid price
4. If no fill, system increments price up by 2-3 price levels to attempt fill
5. If still no fill, system returns to original price and leaves order open as day order
6. Upon order fill, system creates a limit exit order for 1/2 quantity at 100% profit target
7. System logs the automated order execution

### Edge Case: Invalid Order Parameters
1. Trader submits order with invalid parameters via chat
2. System responds with validation errors and suggestions
3. No order is placed until parameters are corrected

### Testing Scenarios
- Sandbox order placement with various order types (market, limit, options, futures)
- Error handling for API failures in sandbox
- Performance testing of order execution times
- Validation of order status updates
- Switch from sandbox to production configuration
- User authentication and permission validation

## Functional Requirements

### Order Placement
- System must support placing orders through Tastytrade API
- Orders must include standard parameters: symbol, quantity, order type, price
- System must validate order parameters before submission
- Orders must be placed in sandbox environment initially
- For automated orders, system must attempt fills at mid price, then increment up 2-3 price levels if needed
- If no fill after increments, return to original price and leave as day order
- Upon fill, create limit exit order for 1/2 quantity at 100% profit

### Chat Interface
- System must provide chat-based interface for order initiation
- Futures orders initiated via !tt buy/sell/flat commands by authorized user (704125082750156840)
- Options orders triggered by alert messages in format "Alert: [ACTION] [SYMBOL] [STRIKE] [EXPIRY] @ [PRICE]" from authorized users
- Chat commands must support natural language order specifications
- System must parse chat messages to extract order details
- System must provide feedback on order status via chat

### Testing and Validation
- All orders must be tested in sandbox environment first
- System must perform thorough testing of order placement and execution
- Testing must include error scenarios and edge cases
- System must validate successful order execution before production switch

### Production Deployment
- System must have mechanism to switch from sandbox to production
- Production orders must use live Tastytrade accounts
- System must maintain separate configurations for sandbox and production
- Switch to production only after complete testing validation

### Security and Permissions
- System must authenticate users by verifying Discord user ID
- Only user 704125082750156840 can initiate futures orders with !tt commands
- Alert messages for options must be issued by authorized users (700068629626224700 or 704125082750156840)
- Orders must be validated against user permissions and risk limits
- System must log all order activities for audit purposes

## Success Criteria

### Functional Success
- 100% of initiated orders are successfully placed in sandbox
- 95% of tested orders pass validation criteria
- All production orders execute without API errors
- Chat interface processes 99% of valid commands correctly

### Performance Success
- Order placement completes within 5 seconds of chat command
- System handles 100 concurrent order requests
- Testing phase completes within 30 minutes per order type

### User Experience Success
- Traders can place orders using simple chat commands
- System provides clear feedback on order status
- Error messages are helpful and actionable
- No orders are placed in production without sandbox testing

### Reliability Success
- System maintains 99.9% uptime for order services
- All order executions are logged and auditable
- Failed orders are automatically retried or escalated

## Key Entities

- **Order**: Contains symbol, quantity, type, price, status, environment (sandbox/production)
- **Trader**: User with permissions, chat handle, account details
- **ChatMessage**: Message containing order command, timestamp, sender
- **TestResult**: Outcome of sandbox testing, validation metrics
- **Account**: Tastytrade account configuration for sandbox and production, including buying power (BP) and allocation percentage

## Assumptions

- Tastytrade API provides sandbox and production environments
- Chat interface is integrated with existing Discord bot infrastructure
- Automated ordering is options-based, triggered by specific messages in designated Discord channels
- Manual ordering and futures orders are initiated via chat commands
- Testing includes both functional and performance validation

## Dependencies

- Tastytrade API integration (sandbox and production access)
- Chat platform integration (Discord bot)
- Market data feeds for automated signals
- User authentication and authorization system
- Logging and monitoring infrastructure

## Out of Scope

- Integration with other trading platforms
- Advanced order types beyond standard market/limit/options/futures
- Real-time market data visualization
- Portfolio management features
- Multi-broker order routing
// position_bridge.cpp
// Writes the current Sierra Chart position to position_snapshot.json so
// sweep_runner.py (PositionMonitorService) can read live position state
// without needing a separate broker API call.
//
// Produces one file written atomically:
//   <DataFilesFolder>\position_snapshot.json
//
// JSON schema:
//   {
//     "ts_ms":        1777525977138,   // UTC milliseconds
//     "symbol":       "MNQ",           // short symbol (from study input)
//     "quantity":     2,               // positive=long, negative=short, 0=flat
//     "entry_price":  21234.50,        // average entry price (0 when flat)
//     "open_pnl":     87.50,           // unrealised P&L in dollars
//     "source":       "sc_position"
//   }
//
// Install:
//   1. Copy this file to C:\SierraChart\ACS_Source\
//   2. In SC: Analysis > Build Custom Studies DLL
//   3. Add "Position Bridge" study to the same chart as sweep_dom_exporter.
//   4. Set the Symbol input to match SC_DOM_SYMBOL in your .env (default: MNQ)
//
// The study auto-updates on every tick (AutoLoop = 0, UpdateAlways = 1).
// Update interval is capped by In_UpdateMs to avoid excessive I/O.

#include "sierrachart.h"
#include <stdio.h>
#include <time.h>

SCDLLName("Position Bridge")

static const int UPDATE_MS_DEFAULT = 250;

// ---------------------------------------------------------------------------
// Helpers (shared with sweep_dom_exporter.cpp)
// ---------------------------------------------------------------------------

static int WriteAtomic(const SCString& Path, const SCString& Content)
{
    SCString TmpPath = Path + ".tmp";
    FILE* f = fopen(TmpPath.GetChars(), "wb");
    if (!f) return 0;
    fwrite(Content.GetChars(), 1, Content.GetLength(), f);
    fclose(f);
    if (rename(TmpPath.GetChars(), Path.GetChars()) != 0) {
        remove(Path.GetChars());
        rename(TmpPath.GetChars(), Path.GetChars());
    }
    return 1;
}

static int64_t NowMs()
{
    SYSTEMTIME st;
    GetSystemTime(&st);
    FILETIME ft;
    SystemTimeToFileTime(&st, &ft);
    int64_t t = ((int64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    return (t - 116444736000000000LL) / 10000LL;
}

// ---------------------------------------------------------------------------
// Study
// ---------------------------------------------------------------------------

SCSFExport scsf_PositionBridge(SCStudyInterfaceRef sc)
{
    SCInputRef In_Symbol   = sc.Input[0];
    SCInputRef In_UpdateMs = sc.Input[1];

    if (sc.SetDefaults)
    {
        sc.GraphName        = "Position Bridge";
        sc.StudyDescription = "Writes position_snapshot.json for sweep_runner.py.";
        sc.AutoLoop         = 0;   // manual: runs on every tick
        sc.UpdateAlways     = 1;

        In_Symbol.Name = "Symbol Override (blank = chart symbol)";
        In_Symbol.SetString("MNQ");

        In_UpdateMs.Name = "Min Update Interval (ms)";
        In_UpdateMs.SetInt(UPDATE_MS_DEFAULT);
        In_UpdateMs.SetIntLimits(50, 5000);

        sc.Subgraph[0].Name      = "Last Write OK";
        sc.Subgraph[0].DrawStyle = DRAWSTYLE_IGNORE;

        return;
    }

    // Rate-limit
    int64_t& LastWriteMs = sc.GetPersistentInt64(1);
    int64_t  NowTime     = NowMs();
    if ((NowTime - LastWriteMs) < In_UpdateMs.GetInt())
        return;
    LastWriteMs = NowTime;

    // Symbol
    SCString Symbol = In_Symbol.GetString();
    if (Symbol.IsEmpty())
        Symbol = sc.Symbol;

    // ------------------------------------------------------------------
    // Read position from SC's built-in trade position tracking
    // ------------------------------------------------------------------
    // sc.GetTradePosition() returns a s_SCTradePosition struct with:
    //   PositionQuantity  (int,   positive=long, negative=short)
    //   AveragePrice      (float, average fill price; 0.0 when flat)
    //   OpenProfitLoss    (float, unrealised P&L in account currency)
    // ------------------------------------------------------------------
    s_SCTradePosition pos;
    sc.GetTradePosition(pos);

    int   Qty        = pos.PositionQuantity;
    float AvgPrice   = pos.AveragePrice;
    float OpenPnl    = pos.OpenProfitLoss;

    // When flat, SC may report a stale average price — zero it out explicitly
    if (Qty == 0) {
        AvgPrice = 0.0f;
        OpenPnl  = 0.0f;
    }

    // ------------------------------------------------------------------
    // Build JSON
    // ------------------------------------------------------------------
    SCString Json;
    Json.Format(
        "{"
        "\"ts_ms\":%lld,"
        "\"symbol\":\"%s\","
        "\"quantity\":%d,"
        "\"entry_price\":%.2f,"
        "\"open_pnl\":%.2f,"
        "\"source\":\"sc_position\""
        "}",
        (long long)NowTime,
        Symbol.GetChars(),
        Qty,
        (double)AvgPrice,
        (double)OpenPnl
    );

    // ------------------------------------------------------------------
    // Write atomically
    // ------------------------------------------------------------------
    SCString DataDir = sc.DataFilesFolder();
    int Ok = WriteAtomic(DataDir + "position_snapshot.json", Json);
    sc.Subgraph[0][sc.Index] = (float)Ok;
}

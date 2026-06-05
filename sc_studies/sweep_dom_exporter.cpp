// sweep_dom_exporter.cpp
// Exports DOM order book + CVD snapshot to JSON files for sweep_runner.py.
//
// Produces two files written atomically (write temp, rename):
//   <DataFilesFolder>\dom_snapshot.json   — DOM order book
//   <DataFilesFolder>\cvd_snapshot.json   — CVD rolling windows
//
// Install: copy this file to C:\SierraChart\ACS_Source\
//          In SC: Analysis > Build Custom Studies DLL
//          Add study to the MNQM26 (or your active NQ futures) chart.
//          Chart must have "Save Bid and Ask Volume" enabled in Chart Settings.

#include "sierrachart.h"
#include "ACSILDepthBars.h"
#include <math.h>
#include <stdio.h>
#include <time.h>

SCDLLName("Sweep DOM Exporter")

static const int DOM_LEVELS     = 20;
static const int UPDATE_MS_DEFAULT = 100;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Write a file atomically: write to .tmp then rename so Python never reads
// a partial file. Returns 1 on success.
static int WriteAtomic(const SCString& Path, const SCString& Content)
{
    SCString TmpPath = Path + ".tmp";

    int Handle = 0;
    // Overwrite mode
    FILE* f = fopen(TmpPath.GetChars(), "wb");
    if (!f) return 0;
    fwrite(Content.GetChars(), 1, Content.GetLength(), f);
    fclose(f);

    // Atomic rename (Windows MoveFileEx with REPLACE flag via CRT rename)
    if (rename(TmpPath.GetChars(), Path.GetChars()) != 0) {
        // Fallback: delete dest then rename
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
    // FILETIME is 100-nanosecond intervals since 1601-01-01
    int64_t t = ((int64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    // Convert to Unix epoch ms: subtract 116444736000000000 (100ns intervals)
    // then divide by 10000 to get ms
    return (t - 116444736000000000LL) / 10000LL;
}

// ---------------------------------------------------------------------------
// Study
// ---------------------------------------------------------------------------

SCSFExport scsf_SweepDOMExporter(SCStudyInterfaceRef sc)
{
    SCInputRef In_DomLevels    = sc.Input[0];
    SCInputRef In_UpdateMs     = sc.Input[1];
    SCInputRef In_Symbol       = sc.Input[2];

    if (sc.SetDefaults)
    {
        sc.GraphName        = "Sweep DOM Exporter";
        sc.StudyDescription = "Writes dom_snapshot.json and cvd_snapshot.json "
                              "for sweep_runner.py (Python sweep classifier).";
        sc.AutoLoop         = 0;   // manual: runs on every tick
        sc.UpdateAlways     = 1;

        In_DomLevels.Name = "DOM Levels to Export";
        In_DomLevels.SetInt(DOM_LEVELS);
        In_DomLevels.SetIntLimits(1, 40);

        In_UpdateMs.Name = "Min Update Interval (ms)";
        In_UpdateMs.SetInt(UPDATE_MS_DEFAULT);
        In_UpdateMs.SetIntLimits(50, 5000);

        In_Symbol.Name = "Symbol Override (blank = chart symbol)";
        In_Symbol.SetString("MNQ");

        sc.Subgraph[0].Name      = "Last Write OK";
        sc.Subgraph[0].DrawStyle = DRAWSTYLE_IGNORE;

        return;
    }

    // Rate-limit: only write every In_UpdateMs milliseconds
    int64_t& LastWriteMs = sc.GetPersistentInt64(1);
    int64_t  NowTime     = NowMs();
    if ((NowTime - LastWriteMs) < In_UpdateMs.GetInt())
        return;
    LastWriteMs = NowTime;

    const int Levels = In_DomLevels.GetInt();

    // Symbol string used as the Redis channel key (e.g. "MNQ")
    SCString Symbol = In_Symbol.GetString();
    if (Symbol.IsEmpty())
        Symbol = sc.Symbol;

    // ------------------------------------------------------------------
    // DOM order book via c_ACSILDepthBars
    // ------------------------------------------------------------------
    c_ACSILDepthBars* DepthBars = sc.GetMarketDepthBars();

    SCString DomJson;
    float BestBid = sc.Bid;
    float BestAsk = sc.Ask;
    float LastPrice = sc.Close[sc.ArraySize - 1];
    float TickSize  = sc.TickSize;

    if (DepthBars != nullptr && BestBid > 0.0f && BestAsk > 0.0f)
    {
        int CurrentBar = sc.ArraySize - 1;

        // Collect bid levels (best bid down)
        SCString BidsArr = "[";
        float BidDepth1  = 0.0f, BidDepth5  = 0.0f;
        float BidDepth10 = 0.0f, BidDepth20 = 0.0f;
        for (int lvl = 0; lvl < Levels; lvl++)
        {
            float Price = BestBid - (lvl * TickSize);
            int   TickIdx = DepthBars->PriceToTickIndex(Price);
            int   Qty = DepthBars->GetLastBidQuantity(CurrentBar, TickIdx);

            if (lvl > 0) BidsArr += ",";
            SCString Entry;
            Entry.Format("{\"price\":%.2f,\"size\":%d}", Price, Qty);
            BidsArr += Entry;

            // Depth sums
            float Dist = (float)lvl;  // ticks from best
            if (Dist < 1.0f)  BidDepth1  += Qty;
            if (Dist < 5.0f)  BidDepth5  += Qty;
            if (Dist < 10.0f) BidDepth10 += Qty;
            if (Dist < 20.0f) BidDepth20 += Qty;
        }
        BidsArr += "]";

        // Collect ask levels (best ask up)
        SCString AsksArr = "[";
        float AskDepth1  = 0.0f, AskDepth5  = 0.0f;
        float AskDepth10 = 0.0f, AskDepth20 = 0.0f;
        for (int lvl = 0; lvl < Levels; lvl++)
        {
            float Price = BestAsk + (lvl * TickSize);
            int   TickIdx = DepthBars->PriceToTickIndex(Price);
            int   Qty = DepthBars->GetLastAskQuantity(CurrentBar, TickIdx);

            if (lvl > 0) AsksArr += ",";
            SCString Entry;
            Entry.Format("{\"price\":%.2f,\"size\":%d}", Price, Qty);
            AsksArr += Entry;

            float Dist = (float)lvl;
            if (Dist < 1.0f)  AskDepth1  += Qty;
            if (Dist < 5.0f)  AskDepth5  += Qty;
            if (Dist < 10.0f) AskDepth10 += Qty;
            if (Dist < 20.0f) AskDepth20 += Qty;
        }
        AsksArr += "]";

        // OFI: compare best-bid/best-ask size to previous tick
        float& PrevBidBest = sc.GetPersistentFloat(10);
        float& PrevAskBest = sc.GetPersistentFloat(11);
        float  CurBidBest  = (float)DepthBars->GetLastBidQuantity(
            CurrentBar, DepthBars->PriceToTickIndex(BestBid));
        float  CurAskBest  = (float)DepthBars->GetLastAskQuantity(
            CurrentBar, DepthBars->PriceToTickIndex(BestAsk));
        float  Ofi = (CurBidBest - PrevBidBest) - (CurAskBest - PrevAskBest);
        PrevBidBest = CurBidBest;
        PrevAskBest = CurAskBest;

        float TotalDepth5   = BidDepth5 + AskDepth5;
        float ImbalanceRatio = (TotalDepth5 > 0.0f)
            ? (BidDepth5 / TotalDepth5) : 0.5f;

        DomJson.Format(
            "{"
            "\"ts_ms\":%lld,"
            "\"symbol\":\"%s\","
            "\"price\":%.2f,"
            "\"bids\":%s,"
            "\"asks\":%s,"
            "\"ofi_1s\":%.2f,"
            "\"bid_depth_1\":%.1f,"
            "\"bid_depth_5\":%.1f,"
            "\"bid_depth_10\":%.1f,"
            "\"bid_depth_20\":%.1f,"
            "\"ask_depth_1\":%.1f,"
            "\"ask_depth_5\":%.1f,"
            "\"ask_depth_10\":%.1f,"
            "\"ask_depth_20\":%.1f,"
            "\"imbalance_ratio\":%.4f"
            "}",
            (long long)NowTime,
            Symbol.GetChars(),
            LastPrice,
            BidsArr.GetChars(),
            AsksArr.GetChars(),
            Ofi,
            BidDepth1, BidDepth5, BidDepth10, BidDepth20,
            AskDepth1, AskDepth5, AskDepth10, AskDepth20,
            ImbalanceRatio
        );
    }
    else
    {
        // No depth data: write minimal L1 payload
        DomJson.Format(
            "{\"ts_ms\":%lld,\"symbol\":\"%s\",\"price\":%.2f,"
            "\"bids\":[{\"price\":%.2f,\"size\":%d}],"
            "\"asks\":[{\"price\":%.2f,\"size\":%d}],"
            "\"ofi_1s\":0.0,\"bid_depth_5\":%.1f,\"ask_depth_5\":%.1f,"
            "\"imbalance_ratio\":0.5}",
            (long long)NowTime, Symbol.GetChars(), LastPrice,
            BestBid, sc.BidSize, BestAsk, sc.AskSize,
            (float)sc.BidSize, (float)sc.AskSize
        );
    }

    // ------------------------------------------------------------------
    // CVD: sum bar bid/ask volumes over rolling windows
    // ------------------------------------------------------------------
    // Requires "Save Bid and Ask Volume" in Chart Settings > Bar Period.
    // SC_BIDVOL / SC_ASKVOL arrays hold per-bar traded-at-bid / traded-at-ask.
    SCFloatArrayRef BidVol = sc.BaseDataIn[SC_BIDVOL];
    SCFloatArrayRef AskVol = sc.BaseDataIn[SC_ASKVOL];

    // Walk back through bars to accumulate rolling CVD windows.
    // Works for any bar period; we use bar DateTimes to determine windows.
    SCDateTime NowDT = sc.GetCurrentDateTime();
    double CvdRunDay  = 0.0, CvdRunDay_BuyVol1 = 0.0, CvdRunDay_SellVol1 = 0.0;
    double Cvd1  = 0.0, Cvd5  = 0.0, Cvd15 = 0.0;
    double Buy1  = 0.0, Sell1 = 0.0;
    bool   In1   = false, In5 = false, In15 = false;

    SCDateTime DayStart = sc.GetTradingDayStartDateTimeOfBar(sc.ArraySize - 1);

    for (int i = sc.ArraySize - 1; i >= 0; i--)
    {
        SCDateTime BarDT = sc.BaseDateTimeIn[i];
        double ElapsedSec = (NowDT - BarDT).GetAsDouble() * 86400.0;

        double BV = BidVol[i];  // traded at bid (sell-aggressor)
        double AV = AskVol[i];  // traded at ask (buy-aggressor)
        double Delta = AV - BV;

        // Running day CVD from trading day start
        if (BarDT >= DayStart)
            CvdRunDay += Delta;

        if (ElapsedSec <= 60.0)  { Cvd1  += Delta; Buy1  += AV; Sell1 += BV; In1  = true; }
        if (ElapsedSec <= 300.0) { Cvd5  += Delta; In5  = true; }
        if (ElapsedSec <= 900.0) { Cvd15 += Delta; In15 = true; }

        if (ElapsedSec > 900.0) break;
    }

    SCString CvdJson;
    CvdJson.Format(
        "{"
        "\"ts_ms\":%lld,"
        "\"symbol\":\"%s\","
        "\"cvd_1min\":%.1f,"
        "\"cvd_5min\":%.1f,"
        "\"cvd_15min\":%.1f,"
        "\"cvd_running_day\":%.1f,"
        "\"buy_vol_1min\":%.1f,"
        "\"sell_vol_1min\":%.1f"
        "}",
        (long long)NowTime,
        Symbol.GetChars(),
        Cvd1, Cvd5, Cvd15, CvdRunDay,
        Buy1, Sell1
    );

    // ------------------------------------------------------------------
    // Write files
    // ------------------------------------------------------------------
    SCString DataDir = sc.DataFilesFolder();
    int DomOk = WriteAtomic(DataDir + "dom_snapshot.json", DomJson);
    int CvdOk = WriteAtomic(DataDir + "cvd_snapshot.json", CvdJson);

    sc.Subgraph[0][sc.ArraySize - 1] = (DomOk && CvdOk) ? 1.0f : 0.0f;
}

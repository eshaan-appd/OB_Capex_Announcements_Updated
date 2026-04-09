import os
import re
import time
import tempfile
import json
from datetime import datetime, date, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from openai import OpenAI
import streamlit as st

# =========================================
# OpenAI setup (Responses + web_search)
# =========================================
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets).")
    st.stop()

client = OpenAI(api_key=api_key)
OPENAI_MODEL = "gpt-4.1-mini"

# =========================================
# BSE fetching (multi-day, strict)
# =========================================
HOME = "https://www.bseindia.com/"
CORP = "https://www.bseindia.com/corporates/ann.html"

ENDPOINTS = [
    "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w",
    "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w",
]

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": HOME,
    "Origin": "https://www.bseindia.com",
    "X-Requested-With": "XMLHttpRequest",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


CONNECT_TIMEOUT = 12
READ_TIMEOUT = 60
DAY_RETRIES = 3
REQUEST_SLEEP_SECONDS = 0.35


def _build_bse_session() -> requests.Session:
    """Create a hardened requests session with retry/backoff for flaky BSE endpoints."""
    s = requests.Session()

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    s.headers.update(BASE_HEADERS)
    return s


def _safe_warmup(s: requests.Session, log=None):
    """Warm up cookies so BSE is less likely to return HTML/non-JSON responses."""
    for url in [HOME, CORP]:
        try:
            s.get(url, timeout=(CONNECT_TIMEOUT, 20))
            time.sleep(0.15)
        except Exception as e:
            if log is not None:
                log.append(f"Warmup failed for {url}: {type(e).__name__}: {e}")


def _call_once(s: requests.Session, url: str, params: dict, log=None):
    """Single guarded call to BSE announcements API with timeout handling."""
    try:
        r = s.get(url, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        status = r.status_code
        ct = (r.headers.get("content-type") or "").lower()

        if status != 200:
            return [], None, {
                "retryable": status in {429, 500, 502, 503, 504},
                "status": status,
                "ct": ct,
                "error": f"HTTP {status}",
            }

        if "application/json" not in ct:
            snippet = (r.text or "")[:200].replace("\n", " ").strip()
            return [], None, {
                "blocked": True,
                "retryable": True,
                "ct": ct,
                "status": status,
                "snippet": snippet,
            }

        data = r.json()
        rows = data.get("Table") or []
        total = None
        try:
            total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
        except Exception:
            total = None
        return rows, total, {}

    except requests.exceptions.ReadTimeout as e:
        if log is not None:
            log.append(f"ReadTimeout for {url} params={params}: {e}")
        return [], None, {"retryable": True, "error": "read_timeout"}

    except requests.exceptions.ConnectTimeout as e:
        if log is not None:
            log.append(f"ConnectTimeout for {url} params={params}: {e}")
        return [], None, {"retryable": True, "error": "connect_timeout"}

    except requests.exceptions.RequestException as e:
        if log is not None:
            log.append(f"RequestException for {url} params={params}: {type(e).__name__}: {e}")
        return [], None, {"retryable": True, "error": f"request_exception: {type(e).__name__}"}

    except ValueError as e:
        if log is not None:
            log.append(f"JSON parse error for {url} params={params}: {e}")
        return [], None, {"retryable": True, "error": "json_parse_error"}


def _call_once_with_retries(s: requests.Session, url: str, params: dict, log=None, attempts: int = 3):
    """Manual retry wrapper because BSE frequently times out or returns HTML instead of JSON."""
    last_rows, last_total, last_meta = [], None, {}

    for attempt in range(1, attempts + 1):
        if log is not None:
            log.append(
                f"API call attempt {attempt}/{attempts} | "
                f"url={url} | page={params.get('pageno', params.get('Pageno'))}"
            )

        rows, total, meta = _call_once(s, url, params, log=log)
        last_rows, last_total, last_meta = rows, total, meta

        if rows:
            return rows, total, meta

        if meta.get("blocked"):
            if log is not None:
                log.append("Non-JSON/blocked response detected. Re-warming session.")
            _safe_warmup(s, log=log)

        if not meta.get("retryable"):
            break

        if attempt < attempts:
            sleep_for = min(2.5 * attempt, 8)
            if log is not None:
                log.append(f"Retrying after {sleep_for:.1f}s")
            time.sleep(sleep_for)

    return last_rows, last_total, last_meta


def _fetch_single_range(s, d1: str, d2: str, log):
    """
    Fetch announcements for [d1, d2] with a single API range call.
    In the multi-day wrapper this is called with d1==d2 (per-day).
    """
    search_opts = ["", "P"]
    seg_opts = ["C", "E"]
    subcat_opts = ["", "-1"]
    pageno_keys = ["pageno", "Pageno"]
    scrip_keys = ["strScrip", "strscrip"]

    for ep in ENDPOINTS:
        for strType in seg_opts:
            for strSearch in search_opts:
                for subcategory in subcat_opts:
                    for pageno_key in pageno_keys:
                        for scrip_key in scrip_keys:

                            params = {
                                pageno_key: 1,
                                "strCat": "-1",
                                "strPrevDate": d1,
                                "strToDate": d2,
                                scrip_key: "",
                                "strSearch": strSearch,
                                "strType": strType,
                                "subcategory": subcategory,
                            }

                            log.append(
                                f"Trying {ep} | {pageno_key} | {scrip_key} | "
                                f"Type={strType} | Search={strSearch!r} | "
                                f"subcategory={subcategory!r} | {d1}..{d2}"
                            )

                            rows_acc = []
                            page = 1
                            hard_fail_combo = False

                            while True:
                                rows, total, meta = _call_once_with_retries(
                                    s, ep, params, log=log, attempts=3
                                )

                                if meta.get("blocked") or (
                                    meta.get("retryable") and not rows and page == 1
                                ):
                                    log.append(
                                        f"Skipping combo after repeated failures | "
                                        f"endpoint={ep} | page={page} | meta={meta}"
                                    )
                                    hard_fail_combo = True
                                    break

                                if page == 1 and total == 0 and not rows:
                                    log.append("No rows for this combination.")
                                    break

                                if not rows:
                                    log.append("No more rows/pages for this combination.")
                                    break

                                rows_acc.extend(rows)

                                if total and len(rows_acc) >= total:
                                    log.append(
                                        f"Collected all rows for this combo: {len(rows_acc)} / {total}"
                                    )
                                    break

                                params[pageno_key] += 1
                                page += 1
                                time.sleep(REQUEST_SLEEP_SECONDS)

                            if rows_acc:
                                return rows_acc

                            if hard_fail_combo:
                                time.sleep(0.5)

    return []


def fetch_bse_announcements_strict(start_yyyymmdd: str, end_yyyymmdd: str, log=None):
    """
    Multi-day version of the original fetcher (works for end_date > start_date).

    Key hardening:
    - retry/backoff on slow BSE responses
    - re-warm session when BSE serves HTML instead of JSON
    - skip failed days instead of crashing the whole app
    """
    if log is None:
        log = []

    s = _build_bse_session()
    _safe_warmup(s, log=log)

    start_dt = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    end_dt = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    if end_dt < start_dt:
        raise ValueError("end_date cannot be earlier than start_date")

    all_rows = []
    failed_days = []

    cur = start_dt
    while cur <= end_dt:
        d_str = cur.strftime("%Y%m%d")
        log.append(f"Day fetch start: {d_str}")

        day_rows = []
        day_ok = False

        for attempt in range(1, DAY_RETRIES + 1):
            try:
                log.append(f"Day-level attempt {attempt}/{DAY_RETRIES} for {d_str}")
                day_rows = _fetch_single_range(s, d_str, d_str, log)
                day_ok = True
                break
            except Exception as e:
                log.append(f"Unexpected error on {d_str}, attempt {attempt}: {type(e).__name__}: {e}")
                _safe_warmup(s, log=log)
                time.sleep(min(2 * attempt, 6))

        if not day_ok:
            failed_days.append(d_str)
            log.append(f"Day skipped after repeated failures: {d_str}")
        else:
            log.append(f"Day fetch done: {d_str} | rows={len(day_rows)}")
            all_rows.extend(day_rows)

        cur += timedelta(days=1)
        time.sleep(REQUEST_SLEEP_SECONDS)

    if failed_days:
        log.append("Failed days: " + ", ".join(failed_days))

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "SCRIP_CD",
                "SLONGNAME",
                "HEADLINE",
                "NEWSSUB",
                "NEWS_DT",
                "ATTACHMENTNAME",
                "NSURL",
            ]
        )

    base_cols = [
        "SCRIP_CD",
        "SLONGNAME",
        "HEADLINE",
        "NEWSSUB",
        "NEWS_DT",
        "ATTACHMENTNAME",
        "NSURL",
        "NEWSID",
    ]

    seen = set(base_cols)
    extra_cols = []

    for r in all_rows:
        for k in r.keys():
            if k not in seen:
                extra_cols.append(k)
                seen.add(k)

    df = pd.DataFrame(all_rows, columns=base_cols + extra_cols)

    keys = ["NSURL", "NEWSID", "ATTACHMENTNAME", "HEADLINE"]
    keys = [k for k in keys if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys)

    if "NEWS_DT" in df.columns:
        df["_NEWS_DT_PARSED"] = pd.to_datetime(
            df["NEWS_DT"], errors="coerce", dayfirst=True
        )
        df = (
            df.sort_values("_NEWS_DT_PARSED", ascending=False)
            .drop(columns=["_NEWS_DT_PARSED"])
            .reset_index(drop=True)
        )

    return df


# =========================================
# Filters: Orders + Capex (same logic as original)
# =========================================
ORDER_KEYWORDS = ["order", "contract", "bagged", "supply", "purchase order"]
ORDER_REGEX = re.compile(
    r"\b(?:" + "|".join(map(re.escape, ORDER_KEYWORDS)) + r")\b", re.IGNORECASE
)

CAPEX_KEYWORDS = [
    "capex",
    "capital expenditure",
    "capacity expansion",
    "new plant",
    "manufacturing facility",
    "brownfield",
    "greenfield",
    "setting up a plant",
    "increase in capacity",
    "expansion",
]
CAPEX_REGEX = re.compile("|".join(CAPEX_KEYWORDS), re.IGNORECASE)


def enrich_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw DF down to order-related announcements."""
    if df.empty:
        return df
    mask = df["HEADLINE"].fillna("").str.contains(ORDER_REGEX)
    out = df.loc[
        mask,
        ["SLONGNAME", "HEADLINE", "NEWSSUB", "NEWS_DT", "ATTACHMENTNAME", "NSURL"],
    ].copy()
    out.columns = ["Company", "Announcement", "Details", "Date", "Attachment", "Link"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


def enrich_capex(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw DF down to capex-related announcements (headline + NEWSSUB)."""
    if df.empty:
        return df
    combined = df["HEADLINE"].fillna("") + " " + df["NEWSSUB"].fillna("")
    mask = combined.str.contains(CAPEX_REGEX, na=False)
    out = df.loc[
        mask,
        ["SLONGNAME", "HEADLINE", "NEWSSUB", "NEWS_DT", "ATTACHMENTNAME", "NSURL"],
    ].copy()
    out.columns = ["Company", "Announcement", "Details", "Date", "Attachment", "Link"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    return out.sort_values("Date", ascending=False).reset_index(drop=True)


# =========================================
# PDF utilities (from reference file)
# =========================================
def _candidate_pdf_urls(row: dict) -> list[str]:
    """Generate possible PDF URLs from ATTACHMENTNAME + NSURL."""
    cands = []
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        cands += [
            f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/Attach/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
        ]
    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        cands.append(ns if ns.lower().startswith("http") else HOME + ns.lstrip("/"))
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _download_pdf(url: str, timeout: int = 25) -> bytes:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/pdf,application/octet-stream,*/*",
            "Referer": CORP,
        }
    )
    r = s.get(url, timeout=timeout, allow_redirects=True, stream=False)
    if r.status_code != 200:
        return b""
    data = r.content
    if not data or len(data) < 500:
        return b""
    return data


def _upload_pdf_to_openai(pdf_bytes: bytes, fname: str = "document.pdf"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        f = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")
    return f


def _call_openai_json(
    prompt: str, file_id: str | None = None, max_tokens: int = 600, temperature: float = 0.2
) -> dict | None:
    """Call OpenAI Responses API expecting a JSON object back."""
    content = [{"type": "input_text", "text": prompt}]
    if file_id:
        content.append({"type": "input_file", "file_id": file_id})

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=[{"type": "web_search"}],
            input=[{"role": "user", "content": content}],
        )
    except Exception as e:
        st.warning(f"OpenAI call failed: {e}")
        return None

    txt = (getattr(resp, "output_text", None) or "").strip()
    if not txt:
        return None

    cleaned = (
        txt.strip()
        .replace("```json", "")
        .replace("```", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )

    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

    try:
        return json.loads(cleaned)
    except Exception as e:
        st.warning(f"JSON parse error: {e}")
        return None


# =========================================
# Orders enrichment via OpenAI + web_search
# =========================================
def enrich_orders_with_openai(orders_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every order announcement, use OpenAI (with web_search + PDF)
    to compute:
        - Revenue (₹ Cr)
        - Market Cap (₹ Cr)
        - Order Book from Filing (₹ Cr)
        - Current Order Book (₹ Cr)
        - Order Book from Filing / Revenue (x)
    """
    if orders_df.empty:
        return orders_df

    df = orders_df.copy()

    df["Revenue (₹ Cr)"] = np.nan
    df["Market Cap (₹ Cr)"] = np.nan
    df["Order Book from Filing (₹ Cr)"] = np.nan
    df["Current Order Book (₹ Cr)"] = np.nan
    df["Order Book from Filing / Revenue (x)"] = np.nan

    raw_index = raw_df.set_index(["SLONGNAME", "HEADLINE"])

    for idx, row in df.iterrows():
        company = str(row["Company"])
        headline = str(row["Announcement"])
        details = str(row.get("Details") or "")
        ann_date = row["Date"]
        date_str = ann_date.strftime("%Y-%m-%d") if pd.notnull(ann_date) else ""

        # Underlying raw row to get ATTACHMENTNAME / NSURL
        try:
            raw_row = raw_index.loc[(company, headline)]
            if isinstance(raw_row, pd.DataFrame):
                raw_row = raw_row.iloc[0]
        except KeyError:
            raw_row = None

        file_id = None
        if raw_row is not None:
            urls = _candidate_pdf_urls(raw_row)
            for u in urls:
                pdf_bytes = _download_pdf(u)
                if pdf_bytes:
                    try:
                        fobj = _upload_pdf_to_openai(pdf_bytes, fname="order.pdf")
                        file_id = fobj.id
                        break
                    except Exception:
                        pass
                time.sleep(0.2)

        prompt = f"""
You are a fundamental equity analyst specialising in Indian listed companies.

For the company and order-related stock-exchange announcement below, extract these metrics:

1) Use web_search to obtain the company's latest **revenue** (most recent full-year
   OR trailing-twelve-months) and **current market capitalisation**.
   - Prefer reliable Indian equity sources (Screener.in, stock exchanges, Moneycontrol, company filings).
   - Return both in **INR Crore (₹ Cr)**.
   - If a figure is in INR but not in crore, convert it (1 crore = 10,000,000 rupees).

2) From the attached PDF filing and the announcement text, identify the most recent
   disclosed **Order Book / Order Backlog** figure for the company (if any).
   - Typically phrased as "order book", "order backlog", "unexecuted order book", etc.
   - Treat this as **Order Book from Filing (₹ Cr)**.
   - Convert to INR Crore.

3) Using web_search (presentations, concall transcripts, annual reports, etc.),
   estimate the company's **Current Total Order Book (₹ Cr)** as of the latest
   disclosed period.
   - Convert to INR Crore.
   - If you only find a range, pick a reasonable point estimate.

4) Compute the ratio **Order Book from Filing / Revenue**:
   - If both revenue and "order book from filing" are available and positive,
     compute (order_book_from_filing / revenue).
   - If you cannot compute, leave it null.

Return ONLY valid JSON in this structure:

{{
  "revenue_cr": <number or null>,
  "market_cap_cr": <number or null>,
  "order_book_from_filing_cr": <number or null>,
  "current_order_book_cr": <number or null>,
  "order_book_filing_by_revenue_x": <number or null>
}}

Company: {company}
Headline: {headline}
Details: {details}
Announcement_date: {date_str}
"""

        data = _call_openai_json(prompt, file_id=file_id, max_tokens=900, temperature=0.1)
        if not data:
            continue

        def _to_num(x):
            if x is None:
                return np.nan
            try:
                return float(str(x).replace(",", ""))
            except Exception:
                return np.nan

        rev = _to_num(data.get("revenue_cr"))
        mcap = _to_num(data.get("market_cap_cr"))
        ob_filing = _to_num(data.get("order_book_from_filing_cr"))
        ob_cur = _to_num(data.get("current_order_book_cr"))
        ratio = _to_num(data.get("order_book_filing_by_revenue_x"))

        if pd.isna(ratio) and pd.notna(rev) and rev > 0 and pd.notna(ob_filing):
            ratio = ob_filing / rev

        df.at[idx, "Revenue (₹ Cr)"] = rev
        df.at[idx, "Market Cap (₹ Cr)"] = mcap
        df.at[idx, "Order Book from Filing (₹ Cr)"] = ob_filing
        df.at[idx, "Current Order Book (₹ Cr)"] = ob_cur
        df.at[idx, "Order Book from Filing / Revenue (x)"] = ratio

    return df


# =========================================
# Capex Impact enrichment via OpenAI + web_search
# =========================================
def enrich_capex_with_openai(capex_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every capex announcement, use OpenAI (with web_search + PDF)
    to generate an 'Impact' paragraph similar to the sample screenshot.
    """
    if capex_df.empty:
        return capex_df

    df = capex_df.copy()
    df["Impact"] = ""

    raw_index = raw_df.set_index(["SLONGNAME", "HEADLINE"])

    for idx, row in df.iterrows():
        company = str(row["Company"])
        headline = str(row["Announcement"])
        details = str(row.get("Details") or "")

        # Find raw row and any PDF
        try:
            raw_row = raw_index.loc[(company, headline)]
            if isinstance(raw_row, pd.DataFrame):
                raw_row = raw_row.iloc[0]
        except KeyError:
            raw_row = None

        file_id = None
        if raw_row is not None:
            urls = _candidate_pdf_urls(raw_row)
            for u in urls:
                pdf_bytes = _download_pdf(u)
                if pdf_bytes:
                    try:
                        fobj = _upload_pdf_to_openai(pdf_bytes, fname="capex.pdf")
                        file_id = fobj.id
                        break
                    except Exception:
                        pass
                time.sleep(0.2)

        keyword_list = ", ".join(CAPEX_KEYWORDS)

        prompt = f"""
You are a sell-side equity research analyst.

You are given:
- a BSE announcement about a possible capex / capacity expansion, and
- where available, the full PDF filing.

1) First, read the PDF (if attached) and the announcement text.
   Focus on phrases related to: {keyword_list}.

2) Determine:
   - what is being set up or expanded (plant / line / project),
   - key product or segment,
   - location and incremental capacity (if disclosed),
   - approximate capex outlay (₹ Cr) where explicitly given or reasonably inferable.

3) Using web_search only for context (industry structure, company scale),
   write a concise investor-facing **Impact** paragraph (3–6 sentences, ~120–160 words) that covers:
   - how this capex changes the company’s growth and margin trajectory vs current business,
   - rough revenue and EBITDA potential range once stabilised,
   - key execution / demand / regulatory risks and dependencies.

If the filing clearly does **not** relate to capex / plant / capacity expansion, return a short
sentence like: "This filing does not relate to capex or capacity expansion."

Tone: analytical, neutral, no hype. Output plain text only.

Company: {company}
Headline: {headline}
Details: {details}
"""

        content = [{"type": "input_text", "text": prompt}]
        if file_id:
            content.append({"type": "input_file", "file_id": file_id})

        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                temperature=0.25,
                max_output_tokens=260,
                tools=[{"type": "web_search"}],
                input=[{"role": "user", "content": content}],
            )
            impact = (getattr(resp, "output_text", None) or "").strip()
        except Exception as e:
            impact = f"(OpenAI error: {e})"

        df.at[idx, "Impact"] = impact

    return df


# =========================================
# Streamlit UI
# =========================================
st.set_page_config(
    page_title="BSE Orders & Capex (OpenAI-enriched)", layout="wide"
)
st.title("📣 BSE Order & Capex Announcements — OpenAI Web-search Enriched")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=date(2025, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=date.today())

run = st.button("🔎 Fetch & Enrich", use_container_width=True)

if run:
    if start_date > end_date:
        st.error("Start Date cannot be after End Date.")
        st.stop()

    ds = start_date.strftime("%Y%m%d")
    de = end_date.strftime("%Y%m%d")
    logs: list[str] = []

    with st.spinner("Fetching BSE announcements..."):
        df_raw = fetch_bse_announcements_strict(ds, de, log=logs)

    orders_df = enrich_orders(df_raw)
    capex_df = enrich_capex(df_raw)

    st.metric("Total Announcements", len(df_raw))
    st.metric("Order Announcements", len(orders_df))
    st.metric("Capex Announcements", len(capex_df))

    if df_raw.empty:
        st.warning("No announcements found for this date range.")
        st.stop()

    with st.spinner("Enriching order announcements via OpenAI (web_search + PDFs)..."):
        orders_enriched = enrich_orders_with_openai(orders_df, df_raw)

    with st.spinner("Generating capex Impact commentary via OpenAI (web_search + PDFs)..."):
        capex_enriched = enrich_capex_with_openai(capex_df, df_raw)

    tab_orders, tab_capex, tab_all, tab_logs = st.tabs(
        ["📦 Orders (Enriched)", "🏭 Capex (Impact)", "📄 All Raw", "🪵 Fetch Logs"]
    )

    with tab_orders:
        st.caption(
            "Revenue & Market Cap are fetched via OpenAI web_search. "
            "Order Book from Filing is extracted from the filing itself, "
            "Current Order Book uses latest web_search disclosures, and "
            "Order Book from Filing / Revenue is computed as a ratio."
        )
        st.dataframe(orders_enriched, use_container_width=True)

    with tab_capex:
        st.caption(
            "Capex announcements are filtered using the same keyword logic as your original app, "
            "then OpenAI reads the PDFs and performs web_search as needed to "
            "generate an investor-style Impact paragraph similar to the example screenshot."
        )
        st.dataframe(capex_enriched, use_container_width=True)

    with tab_all:
        st.dataframe(df_raw, use_container_width=True)

    with tab_logs:
        st.write("\n".join(logs))

"""Sentimen berita: judul berita dari Google News RSS (gratis, tanpa API key), dinilai dengan kamus kata.

Penilaian memakai kamus kata keuangan Bahasa Indonesia dan Inggris (naik/menguat/laba vs
turun/melemah/rugi) dengan penanganan kata negasi. Ringan dan tanpa unduhan model besar; bisa diganti
model bahasa (mis. IndoBERT) lewat `score_title` tanpa mengubah bagian lain.

RSS hanya memuat berita beberapa minggu terakhir, jadi sentimen belum punya riwayat panjang untuk diuji
pada data lama. Karena itu sentimen dipakai di skor Arah Tren, belum sebagai fitur model.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Callable, Protocol

import pandas as pd

from ..db import Store

log = logging.getLogger(__name__)

# (kata kunci pencarian, bahasa)
QUERIES: dict[str, list[tuple[str, str]]] = {
    "WTI": [("harga minyak dunia", "id"), ("crude oil prices", "en")],
    "IHSG": [("IHSG", "id")],
    "USDIDR": [("rupiah dolar", "id")],
    "GOLD_IDR_G": [("harga emas", "id"), ("gold price", "en")],
    "BTC": [("bitcoin", "id"), ("bitcoin price", "en")],
    "TLKM": [("saham TLKM Telkom", "id")],
    "ADRO": [("saham ADRO Alamtri", "id")],
    "BRPT": [("saham BRPT Barito Pacific", "id")],
}
LANG = {"id": "hl=id&gl=ID&ceid=ID:id", "en": "hl=en-US&gl=US&ceid=US:en"}
KEEP_DAYS = 30
WINDOW_DAYS = 7
MIN_HEADLINES = 5

POSITIVE = set("""
naik menguat penguatan melonjak lonjakan melesat meroket rekor tertinggi laba untung keuntungan tumbuh
pertumbuhan positif rebound pulih pemulihan membaik optimis optimisme bullish surplus dividen ekspansi
kenaikan menanjak menghijau borong diborong akumulasi melambung terbang cuan prospek upgrade
rise rises rising rose gain gains surge surges surged jump jumps jumped rally rallies rallied record
beat beats profit profits growth soar soars soared bullish upgrade upgraded rebound rebounds recover
recovery optimism strong stronger higher boost boosts climbs climb climbed
""".split())
NEGATIVE = set("""
turun melemah pelemahan anjlok merosot ambruk ambles longsor rugi kerugian defisit negatif pesimis
bearish koreksi tekanan tertekan terendah krisis resesi sanksi gagal gugatan phk penurunan memerah
terjun jatuh terpuruk lesu ancaman khawatir kekhawatiran waspada downgrade boncos merugi
fall falls falling fell drop drops dropped slump slumps plunge plunges plunged tumble tumbles tumbled
loss losses miss misses bearish downgrade downgraded fear fears crisis recession sanctions weak weaker
lower slide slides slid sink sinks sank crash crashes worries concern concerns decline declines declined
""".split())
NEGATORS = {"tidak", "tak", "belum", "bukan", "not", "no", "never", "without"}


def score_title(title: str, asset: str = "") -> float:
    """Skor -1..+1 dari satu judul. Untuk USD/IDR, kabar Rupiah dibalik (Rupiah menguat = USD/IDR turun)."""
    words = re.findall(r"[a-zA-Z]+", title.lower())
    pos = neg = 0
    for i, w in enumerate(words):
        flip = i > 0 and words[i - 1] in NEGATORS
        if w in POSITIVE:
            neg, pos = (neg + 1, pos) if flip else (neg, pos + 1)
        elif w in NEGATIVE:
            pos, neg = (pos + 1, neg) if flip else (pos, neg + 1)
    if pos + neg == 0:
        return 0.0
    s = (pos - neg) / (pos + neg)
    if asset == "USDIDR" and "rupiah" in words:
        s = -s
    return float(s)


class NewsSource(Protocol):
    def fetch_news(self, query: str, lang: str) -> list[dict]:
        """Daftar {title, link, source, published (Timestamp UTC)}."""


class GoogleNewsSource:
    def fetch_news(self, query: str, lang: str) -> list[dict]:
        import calendar

        import feedparser

        from .fundamentals import _http_get
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&{LANG[lang]}"
        feed = feedparser.parse(_http_get(url))
        out = []
        for e in feed.entries:
            t = e.get("published_parsed") or e.get("updated_parsed")
            if not t:
                continue
            title = e.get("title", "")
            src = (e.get("source") or {}).get("title") if isinstance(e.get("source"), dict) else None
            if src and title.endswith(f" - {src}"):
                title = title[: -len(src) - 3]
            out.append({"title": title.strip(), "link": e.get("link"), "source": src,
                        "published": pd.Timestamp(calendar.timegm(t), unit="s")})
        return out


def update_news(store: Store, source: NewsSource | None = None, assets: list[str] | None = None,
                progress: Callable[[str], None] = lambda m: None) -> dict[str, int]:
    source = source or GoogleNewsSource()
    progress("Mengambil berita terbaru…")
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=KEEP_DAYS)
    written: dict[str, int] = {}
    for asset in assets or list(QUERIES):
        items = []
        for q, lang in QUERIES.get(asset, []):
            try:
                items += source.fetch_news(q, lang)
            except Exception as exc:
                log.warning("Gagal mengambil berita %s (%s): %s", asset, q, exc)
        rows = [{"published": pd.Timestamp(it["published"]).strftime("%Y-%m-%d %H:%M"), "title": it["title"],
                 "link": it.get("link"), "source": it.get("source"), "score": score_title(it["title"], asset)}
                for it in items if it.get("title") and pd.Timestamp(it["published"]) >= cutoff]
        written[asset] = store.upsert_news(asset, rows)
    return written


@dataclass
class Sentiment:
    n: int = 0
    avg: float = 0.0
    headlines: list[dict] = field(default_factory=list)  # judul paling berpengaruh

    @property
    def direction(self) -> int:
        if self.n < MIN_HEADLINES:
            return 0
        return 1 if self.avg >= 0.15 else -1 if self.avg <= -0.15 else 0

    @property
    def mood(self) -> str:
        if self.n < MIN_HEADLINES:
            return "belum cukup berita"
        return {1: "positif", -1: "negatif", 0: "netral"}[self.direction]


def assess_sentiment(store: Store, asset: str, now: pd.Timestamp | None = None) -> Sentiment:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    since = (now - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d %H:%M")
    df = store.news(asset, since)
    if df.empty:
        return Sentiment()
    top = df.reindex(df.score.abs().sort_values(ascending=False, kind="stable").index)
    top = top[top.score != 0].head(3)
    return Sentiment(n=len(df), avg=float(df.score.mean()), headlines=top.to_dict("records"))

#!/usr/bin/env python3
"""
Robot Booster V.1
Système de sélection de contrats d'options :
- Calls uniquement si SPY > MM200 (régime haussier)
- Puts uniquement si SPY < MM200 (régime baissier)

Filtres :
- A : Sélection du titre (type EQUITY, volume, résultats éloignés, beta)
- B : Signal directionnel (confluence MACD / Stochastique / Volume / Figure + régime SPY)
- C : Contrat d'options conforme (spread, budget, delta min, échéance max, IV)

Avertissement : Analyse uniquement. Aucun ordre n'est transmis. Vérifier les données
avant toute décision. Les grecs sont des approximations Black-Scholes.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from datetime import datetime as _dt

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "beta_min": 1.5,
    "resultats_min_jours": 15,
    "volume_moyenne_jours": 63,
    "cross_window": 3,
    "score_seuil": 80,
    "atr_amplitude_elevee": 6.0,
    "atr_amplitude_moyenne": 3.0,
    "delta_min": 0.30,
    "echeance_max_jours": 90,
    "taux_sans_risque": 0.04,
    "frais_par_contrat": 0.65,
    "budget_min": 300.0,
    "budget_max": 1100.0,
    "spread_max": {"court": 0.08, "moyen": 0.10, "long": 0.15},
}

VERSION = "1.0"
TICKERS_DEFAUT = ["AAPL", "TSLA", "NVDA"]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def horodatage():
    return {
        "date_analyse": datetime.now(timezone.utc).isoformat(),
        "fuseau": "UTC",
        "delai_actions": "yfinance (journalier clôturé)",
    }


def en_tete_global():
    return {
        "version": VERSION,
        "date_lancement": datetime.now(timezone.utc).isoformat(),
        "fuseau": "UTC",
        "sources": {
            "actions": "yfinance (journalier clôturé)",
            "options": "yfinance chaîne d'options (temps réel approximatif)",
            "grecs": "Black-Scholes approximatif — à revérifier courtier",
        },
        "delais": {
            "actions": "journalier clôturé (pas de délai intraday)",
            "options": "yfinance, délai non garanti — revérifier avant ordre",
        },
        "marche_vise": "US uniquement — S&P 500 et Russell 2000",
        "regle_regime": "Calls uniquement si SPY > MM200 ; Puts uniquement si SPY < MM200",
        "regle_atr": "ATR% affiché comme amplitude attendue (elevee >=6%, moyenne 3-6%, faible <3%) ; valable haussier et baissier ; pas de rejet",
        "avertissement": "Analyse, pas une garantie de gain. Aucun ordre transmis.",
    }


def rejet(motif, detail=""):
    return {"decision": "REJET", "motif": motif, "detail": detail}


def ok(donnees):
    return {"decision": "OK", **donnees}


# ---------------------------------------------------------------------------
# Univers US (S&P 500 + Russell 2000)
# ---------------------------------------------------------------------------
_UNIVERSE_CACHE = None

def charger_univers_us():
    """Charge les tickers du S&P 500 (Wikipedia) et du Russell 2000 (GitHub). Cache en mémoire."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is not None:
        return _UNIVERSE_CACHE

    sp500 = set()
    r2000 = set()

    # S&P 500 via Wikipedia
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500 = set(tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass

    # Russell 2000 via GitHub (liste publique)
    try:
        url = "https://raw.githubusercontent.com/quanthero/US_Indices_Constituents/main/Russell2000.csv"
        df = pd.read_csv(url)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        r2000 = set(df[col].astype(str).str.upper().tolist())
    except Exception:
        pass

    _UNIVERSE_CACHE = sp500 | r2000
    return _UNIVERSE_CACHE


def est_dans_univers_us(ticker):
    """Version assouplie : accepte tous les tickers US (sans suffixe).
    Le contrôle strict S&P 500 / Russell 2000 est temporairement désactivé
    car le chargement des listes échoue sur certaines machines.
    """
    # Accepte les tickers US simples (pas de .TO, .T, etc.)
    return "." not in ticker and "-" not in ticker


# ---------------------------------------------------------------------------
# A. Sélection du titre
# ---------------------------------------------------------------------------
def calcul_atr_pct(hist, periode=14):
    """ATR(periode) / Close * 100, en %."""
    if hist is None or len(hist) < periode + 1:
        return None
    try:
        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(periode).mean()
        if atr.isna().all() or close.iloc[-1] <= 0:
            return None
        return round(float(atr.iloc[-1] / close.iloc[-1] * 100), 2)
    except Exception:
        return None


def filtre_A(ticker):
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception as e:
        return rejet("A_INFO_INDISPONIBLE", f"impossible de lire info : {e}")

    quote_type = info.get("quoteType")
    if quote_type is None:
        return rejet("A_TYPE_NON_VERIFIABLE", "quoteType absent")
    if quote_type != "EQUITY":
        return rejet("A_TYPE_EXCLU", f"quoteType={quote_type}")

    # Restriction au marché US (S&P 500 + Russell 2000)
    if not est_dans_univers_us(ticker):
        return rejet(
            "A_HORS_UNIVERS_US",
            "Ticker non présent dans le S&P 500 ni le Russell 2000 (marché US uniquement)",
        )

    beta = info.get("beta")
    beta_valide = isinstance(beta, (int, float)) and not math.isnan(beta)
    beta_score = 10 if (beta_valide and beta > CONFIG["beta_min"]) else 0

    hist = t.history(period="6mo", interval="1d", auto_adjust=False)
    if hist is None or hist.empty or "Volume" not in hist:
        return rejet("A_HISTORIQUE_INDISPONIBLE", "historique volume indisponible")

    hist_cloture = hist.iloc[:-1] if len(hist) > 1 else hist
    if len(hist_cloture) < CONFIG["volume_moyenne_jours"]:
        return rejet(
            "A_HISTORIQUE_INSUFFISANT",
            f"{len(hist_cloture)} séances < {CONFIG['volume_moyenne_jours']}",
        )

    vol_moy = hist_cloture["Volume"].tail(CONFIG["volume_moyenne_jours"]).mean()
    vol_dernier = hist_cloture["Volume"].iloc[-1]
    if not vol_moy or vol_moy <= 0:
        return rejet("A_VOLUME_MOYEN_INVALIDE", f"moyenne={vol_moy}")
    if not (vol_dernier > vol_moy):
        return rejet(
            "A_VOLUME_INSUFFISANT",
            f"séance={int(vol_dernier)} <= moyenne={int(vol_moy)}",
        )

    date_resultats = None
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
            date_resultats = dates[0] if dates else None
        elif cal is not None and hasattr(cal, "empty") and not cal.empty:
            for col in cal.columns:
                if "arnings" in str(col):
                    val = cal[col].iloc[0]
                    date_resultats = val if val is not None else None
                    break
    except Exception as e:
        return rejet("A_RESULTATS_NON_VERIFIABLE", f"lecture calendrier échouée : {e}")

    if date_resultats is None:
        return rejet("A_RESULTATS_NON_VERIFIABLE", "date de résultats absente")

    try:
        d = date_resultats.date() if isinstance(date_resultats, datetime) else date_resultats
        jours_avant = (d - datetime.now(timezone.utc).date()).days
    except Exception as e:
        return rejet(
            "A_RESULTATS_NON_VERIFIABLE",
            f"date illisible : {date_resultats} ({e})",
        )

    if jours_avant < CONFIG["resultats_min_jours"]:
        return rejet(
            "A_RESULTATS_TROP_PROCHES",
            f"{jours_avant}j < {CONFIG['resultats_min_jours']} (résultats {d})",
        )

    return ok({
        "ticker": ticker,
        "traite_a": datetime.now(timezone.utc).isoformat(),
        "statut_bougie": "CLÔTURÉ",
        "secteur": info.get("sector"),
        "quoteType": quote_type,
        "beta": beta if beta_valide else None,
        "beta_score": beta_score,
        "volume_dernier": int(vol_dernier),
        "volume_moyen_ref": int(vol_moy),
        "ratio_volume": round(float(vol_dernier) / float(vol_moy), 3),
        "date_resultats": str(d),
        "jours_avant_resultats": jours_avant,
    })


# ---------------------------------------------------------------------------
# B. Signal directionnel
# ---------------------------------------------------------------------------
def _ema(serie, span):
    return serie.ewm(span=span, adjust=False).mean()


def calcul_indicateurs(hist):
    df = hist.copy()
    df["MACD"] = _ema(df["Close"], 12) - _ema(df["Close"], 26)
    df["MACD_signal"] = _ema(df["MACD"], 9)
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["%K_raw"] = 100 * (df["Close"] - low14) / (high14 - low14)
    df["%K"] = df["%K_raw"].rolling(3).mean()
    df["%D"] = df["%K"].rolling(3).mean()
    return df.dropna()


def _croisement_recent(a, b, fenetre, sens):
    if len(a) < 2:
        return False, None
    for i in range(1, min(fenetre + 1, len(a))):
        if sens == "haut" and a.iloc[-i - 1] <= b.iloc[-i - 1] and a.iloc[-i] > b.iloc[-i]:
            return True, i
        if sens == "bas" and a.iloc[-i - 1] >= b.iloc[-i - 1] and a.iloc[-i] < b.iloc[-i]:
            return True, i
    return False, None


def _figure_graphique(df, sens):
    if len(df) < 20:
        return 0, "historique insuffisant"
    dernier = df.iloc[-1]
    if sens == "haussier":
        res = df["High"].iloc[-20:-1].max()
        if dernier["Close"] > res and dernier["Volume"] > df["Volume"].iloc[-20:-1].mean():
            return 20, f"cassure résistance {res:.2f} avec volume"
        return 0, f"pas de cassure (close {dernier['Close']:.2f} vs résist {res:.2f})"
    sup = df["Low"].iloc[-20:-1].min()
    if dernier["Close"] < sup and dernier["Volume"] > df["Volume"].iloc[-20:-1].mean():
        return 20, f"cassure support {sup:.2f} avec volume"
    return 0, f"pas de cassure (close {dernier['Close']:.2f} vs support {sup:.2f})"


def score_confluence(df, beta_score, sens):
    d = {}
    vol_moy = df["Volume"].iloc[-21:-1].mean()
    vol_now = df["Volume"].iloc[-1]
    d["volume"] = 25 if vol_now > vol_moy else (15 if vol_now > 0.8 * vol_moy else 0)

    sens_macd = "haut" if sens == "haussier" else "bas"
    macd_cross, _ = _croisement_recent(
        df["MACD"], df["MACD_signal"], CONFIG["cross_window"], sens_macd
    )
    hist_ok = (
        (df["MACD_hist"].iloc[-1] > df["MACD_hist"].iloc[-2])
        if sens == "haussier"
        else (df["MACD_hist"].iloc[-1] < df["MACD_hist"].iloc[-2])
    )
    d["macd"] = 25 if (macd_cross and hist_ok) else (15 if (macd_cross or hist_ok) else 0)

    sens_sto = "haut" if sens == "haussier" else "bas"
    sto_cross, _ = _croisement_recent(df["%K"], df["%D"], CONFIG["cross_window"], sens_sto)
    zone_ok = (df["%K"].iloc[-1] < 50) if sens == "haussier" else (df["%K"].iloc[-1] > 50)
    d["stoch"] = 20 if (sto_cross and zone_ok) else (10 if sto_cross else 0)

    fig_score, fig_desc = _figure_graphique(df, sens)
    d["figure"] = fig_score
    d["figure_desc"] = fig_desc
    d["beta"] = beta_score

    score = sum(d[k] for k in ("volume", "macd", "stoch", "figure", "beta"))
    briques = sum(1 for k in ("volume", "macd", "stoch", "figure") if d[k] > 0)
    return score, d, briques


def regime_spy():
    """Retourne True si SPY > MM200 aujourd'hui, False sinon. None si indisponible."""
    try:
        spy = yf.download(
            "SPY",
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if spy is None or spy.empty:
            return None
        close = spy["Close"].squeeze() if hasattr(spy.columns, "levels") else spy["Close"]
        mm200 = close.rolling(200).mean()
        if len(mm200.dropna()) == 0:
            return None
        return bool(close.iloc[-1] > mm200.iloc[-1])
    except Exception:
        return None


def filtre_B(ticker, sens):
    # Régime SPY : calls seulement si SPY > MM200 ; puts seulement si SPY < MM200
    regime = regime_spy()
    if regime is None:
        return rejet(
            "B_REGIME_INDISPONIBLE",
            "SPY/MM200 indisponible : impossible de valider le régime",
        )

    if sens == "haussier" and regime is False:
        return rejet(
            "B_REGIME_BAISSIER",
            "SPY < MM200 : calls non proposés en régime baissier",
        )
    if sens == "baissier" and regime is True:
        return rejet(
            "B_REGIME_HAUSSIER",
            "SPY > MM200 : puts non proposés en régime haussier",
        )

    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", interval="1d", auto_adjust=False)
    if hist is None or hist.empty or len(hist) < 50:
        return rejet("B_HISTORIQUE_INSUFFISANT", "moins de 50 séances")

    df = calcul_indicateurs(hist)
    if len(df) < 30:
        return rejet("B_INDICATEURS_INSUFFISANTS", "pas assez de bougies")

    # ATR% et amplitude attendue (information, pas de rejet)
    _atr = calcul_atr_pct(hist.iloc[:-1] if len(hist) > 1 else hist, 14)
    if _atr is None:
        _amplitude = "inconnue"
    elif _atr >= CONFIG["atr_amplitude_elevee"]:
        _amplitude = "elevee"
    elif _atr >= CONFIG["atr_amplitude_moyenne"]:
        _amplitude = "moyenne"
    else:
        _amplitude = "faible"

    try:
        beta = t.info.get("beta") or 0
    except Exception:
        beta = 0
    beta_score = 10 if beta and beta > CONFIG["beta_min"] else 0

    score, details, briques = score_confluence(df, beta_score, sens)
    if briques < 4:
        return rejet(
            "B_CONFLUENCE_INSUFFISANTE",
            f"{briques}/4 briques non nulles",
        )
    if score < CONFIG["score_seuil"]:
        return rejet(
            "B_SCORE_INSUFFISANT",
            f"{score}/100 < {CONFIG['score_seuil']}",
        )

    etat = (
        "CONFIRMÉ"
        if (details["macd"] >= 15 and details["stoch"] >= 10)
        else "EN PRÉPARATION"
    )

    return ok({
        "sens": sens,
        "traite_a": datetime.now(timezone.utc).isoformat(),
        "statut_bougie": "CLÔTURÉ",
        "etat": etat,
        "score": score,
        "details_score": details,
        "derniere_close": float(df["Close"].iloc[-1]),
        "macd": float(df["MACD"].iloc[-1]),
        "macd_signal": float(df["MACD_signal"].iloc[-1]),
        "stoch_K": float(df["%K"].iloc[-1]),
        "stoch_D": float(df["%D"].iloc[-1]),
        "atr_pct": _atr,
        "amplitude_attendue": _amplitude,
    })


# ---------------------------------------------------------------------------
# C. Filtre options
# ---------------------------------------------------------------------------
def black_scholes_grecs(S, K, T, r, sigma, type_option):
    """Delta et theta approximatifs (Black-Scholes, sans dividende)."""
    if not all(isinstance(x, (int, float)) and x > 0 for x in (S, K, T, sigma)):
        return None, None
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if type_option == "call":
            delta = norm.cdf(d1)
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (
                -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365
        return round(delta, 4), round(theta, 4)
    except Exception:
        return None, None


def seuil_spread(jours_echeance):
    if jours_echeance < 30:
        return CONFIG["spread_max"]["court"]
    if jours_echeance <= 60:
        return CONFIG["spread_max"]["moyen"]
    return CONFIG["spread_max"]["long"]


def filtre_C(ticker, sens, prix_sous_jacent, date_du_jour=None):
    """Cherche un contrat conforme. Rejet explicite si aucun."""
    t = yf.Ticker(ticker)
    try:
        expirations = t.options
    except Exception as e:
        return rejet("C_CHAINE_INDISPONIBLE", f"lecture chaîne échouée : {e}")

    if not expirations:
        return rejet("C_AUCUNE_ECHEANCE", "aucune expiration disponible")

    type_option = "call" if sens == "haussier" else "put"
    aujourd_hui = date_du_jour or _dt.now().date()
    candidats = []

    for exp in expirations:
        try:
            chain = t.option_chain(exp)
            df = chain.calls if type_option == "call" else chain.puts
        except Exception:
            continue
        if df is None or df.empty:
            continue

        try:
            date_exp = _dt.strptime(exp, "%Y-%m-%d").date()
            jours = (date_exp - aujourd_hui).days
        except Exception:
            continue
        if jours <= 0:
            continue

        T = jours / 365.0
        seuil = seuil_spread(jours)

        for row in df.itertuples():
            bid = getattr(row, "bid", None)
            ask = getattr(row, "ask", None)
            if not bid or not ask or bid <= 0 or ask <= 0:
                continue
            mid = (bid + ask) / 2
            spread_rel = (ask - bid) / mid if mid else None
            if spread_rel is None or spread_rel > seuil:
                continue

            strike = float(row.strike)
            iv = getattr(row, "impliedVolatility", None)
            if iv is None or iv <= 0.01:
                continue

            prime = mid
            multiplicateur = 100
            nb_contrats = 1
            debit = prime * multiplicateur * nb_contrats + CONFIG["frais_par_contrat"]
            if debit < CONFIG["budget_min"] or debit > CONFIG["budget_max"]:
                continue

            delta, theta = black_scholes_grecs(
                prix_sous_jacent,
                strike,
                T,
                CONFIG["taux_sans_risque"],
                iv,
                type_option,
            )
            if delta is None:
                continue

            # Règles v0.4 : delta minimum + échéance maximale
            if abs(delta) < CONFIG["delta_min"]:
                continue
            if jours > CONFIG["echeance_max_jours"]:
                continue

            candidats.append({
                "echeance": exp,
                "jours_echeance": jours,
                "strike": strike,
                "bid": round(float(bid), 2),
                "ask": round(float(ask), 2),
                "prime_limite": round(float(prime), 2),
                "spread_relatif": round(float(spread_rel), 4),
                "seuil_spread": seuil,
                "multiplicateur": multiplicateur,
                "nb_contrats": nb_contrats,
                "debit_total": round(float(debit), 2),
                "volume_contrat": int(getattr(row, "volume", 0) or 0),
                "open_interest": int(getattr(row, "openInterest", 0) or 0),
                "iv": round(float(iv), 4),
                "delta": float(delta),
                "theta": float(theta),
                "note_grecs": "approximation Black-Scholes, à revérifier courtier",
            })

    if not candidats:
        return rejet(
            "C_AUCUN_CONTRAT_CONFORME",
            f"{len(expirations)} échéances examinées ; aucun contrat ne passe spread+budget+IV+delta+echeance",
        )

    candidats.sort(key=lambda x: (x["spread_relatif"], -x["open_interest"]))
    return ok({
        "sens": sens,
        "traite_a": datetime.now(timezone.utc).isoformat(),
        "statut_bougie": "PROVISOIRE",
        "type_option": type_option,
        "nb_candidats_conformes": len(candidats),
        "meilleur_contrat": candidats[0],
        "autres_candidats": candidats[1:4],
    })


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyser_ticker(ticker, sens="haussier"):
    """Exécute les filtres A → B → C pour un ticker et un sens."""
    resultat = {"ticker": ticker, "sens_demande": sens}

    a = filtre_A(ticker)
    resultat["A"] = a
    if a["decision"] == "REJET":
        return resultat

    b = filtre_B(ticker, sens)
    resultat["B"] = b
    if b["decision"] == "REJET":
        return resultat

    # Prix sous-jacent (close de la veille si possible pour éviter le jour en cours incomplet)
    hist = yf.Ticker(ticker).history(period="5d", interval="1d", auto_adjust=False)
    px = float(hist["Close"].iloc[-2]) if len(hist) > 1 else float(hist["Close"].iloc[-1])
    resultat["prix_sous_jacent_utilise"] = px

    c = filtre_C(ticker, sens, px)
    resultat["C"] = c
    return resultat


def main():
    parser = argparse.ArgumentParser(
        description=f"Robot Booster V.{VERSION} — Calls (SPY>MM200) et Puts (SPY<MM200)"
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        default=TICKERS_DEFAUT,
        help=f"Tickers à analyser (défaut : {' '.join(TICKERS_DEFAUT)})",
    )
    parser.add_argument(
        "--sens",
        choices=["haussier", "baissier", "les_deux"],
        default="les_deux",
        help="Sens du signal : haussier, baissier ou les_deux (défaut)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Sortie JSON brute uniquement",
    )
    args = parser.parse_args()

    if not args.json:
        print("=== EN-TÊTE GLOBAL ===")
        print(json.dumps(en_tete_global(), indent=2, ensure_ascii=False))
        print()

    resultats = []
    sens_a_tester = ["haussier", "baissier"] if args.sens == "les_deux" else [args.sens]

    for tk in args.tickers:
        for sens in sens_a_tester:
            if not args.json:
                print(f"\n=== Analyse {tk} / {sens} ===")
            res = analyser_ticker(tk, sens)
            resultats.append(res)
            if not args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False, default=str))

    if args.json:
        print(json.dumps(resultats, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

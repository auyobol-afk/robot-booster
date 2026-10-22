import streamlit as st
import json
from datetime import datetime

# Import des fonctions du robot
from robot_booster_v1 import (
    analyser_ticker,
    en_tete_global,
    VERSION,
    charger_univers_us,
)

st.set_page_config(
    page_title=f"Robot Booster V.{VERSION}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .ok-card { background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
    .rejet-card { background-color: #ffebee; border-left: 5px solid #f44336; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
    .header-box { background-color: #f5f5f5; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(f"Robot Booster V.{VERSION}")
    st.markdown("**Marché US** — S&P 500 & Russell 2000")
    st.markdown("---")
    st.markdown("### Paramètres")
    sens_choice = st.radio(
        "Sens à analyser",
        ["les_deux", "haussier", "baissier"],
        index=0,
        help="les_deux = analyse haussière et baissière",
    )
    st.markdown("---")
    st.caption("Les patterns de bougies confirmés assouplissent le filtre volume.")
    st.caption("Score externe Finviz (si disponible) peut assouplir le seuil de confluence.")

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.title("📈 Robot Booster")
st.markdown("Sélection de contrats d'options (Calls / Puts) selon le régime SPY.")

with st.expander("Informations système", expanded=False):
    st.json(en_tete_global())

# ---------------------------------------------------------------------------
# Saisie
# ---------------------------------------------------------------------------
st.markdown("### Mode d'analyse")
mode = st.radio(
    "Choisissez un mode",
    [
        "Tickers manuels",
        "Scanner le marché complet",
        "Scanner automatique continu (beta > 1.5)",
    ],
    horizontal=True,
)

if mode == "Tickers manuels":
    col1, col2 = st.columns([3, 1])
    with col1:
        tickers_input = st.text_input(
            "Tickers à analyser (séparés par des espaces ou des virgules)",
            value="AAPL MSFT NVDA TSLA AMD",
            placeholder="Ex: AAPL MSFT NVDA TWST",
        )
    with col2:
        st.write("")
        st.write("")
        analyser = st.button("🔍 Analyser", type="primary", use_container_width=True)
    tickers = [t.strip().upper() for t in tickers_input.replace(",", " ").split() if t.strip()] if analyser else []
    scanner_continu = False

elif mode == "Scanner le marché complet":
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        limite = st.number_input(
            "Nombre max de titres à scanner (0 = tous)",
            min_value=0,
            max_value=2500,
            value=50,
            step=10,
        )
    with col2:
        st.write("")
        st.write("")
        scanner = st.button("🚀 Scanner le marché", type="primary", use_container_width=True)
    with col3:
        st.write("")
        st.write("")
        afficher_tous = st.checkbox("Afficher tous les résultats", value=False)
    tickers = []
    scanner_continu = False
    if scanner:
        with st.spinner("Chargement de l'univers US..."):
            univers = charger_univers_us()
        if not univers:
            st.error("Impossible de charger la liste des titres.")
        else:
            tickers = sorted(list(univers))
            if limite > 0:
                tickers = tickers[:limite]
            st.info(f"{len(tickers)} titres à analyser.")

else:  # Scanner automatique continu
    col1, col2 = st.columns([2, 1])
    with col1:
        taille_lot = st.number_input(
            "Taille de chaque lot",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Nombre de titres analysés à chaque passage",
        )
    with col2:
        st.write("")
        st.write("")
        lancer_continu = st.button("▶️ Lancer le scanner continu", type="primary", use_container_width=True)
    tickers = []
    scanner_continu = True
    afficher_tous = False

# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
lancer = (
    (mode == "Tickers manuels" and analyser and tickers)
    or (mode == "Scanner le marché complet" and scanner and tickers)
    or (mode == "Scanner automatique continu (beta > 1.5)" and lancer_continu)
)

if lancer:
    sens_a_tester = ["haussier", "baissier"] if sens_choice == "les_deux" else [sens_choice]

    if scanner_continu:
        with st.spinner("Chargement et filtrage de l'univers (beta > 1.5)..."):
            univers = charger_univers_us()
            if not univers:
                st.error("Impossible de charger l'univers.")
                st.stop()
            # On récupère les betas via yfinance (échantillon raisonnable)
            tickers_candidats = sorted(list(univers))
            tickers_beta = []
            progress_beta = st.progress(0)
            for i, tk in enumerate(tickers_candidats[:300]):  # limite pour éviter trop d'appels
                try:
                    import yfinance as yf
                    info = yf.Ticker(tk).info or {}
                    beta = info.get("beta")
                    if isinstance(beta, (int, float)) and beta > 1.5:
                        tickers_beta.append(tk)
                except Exception:
                    pass
                if i % 10 == 0:
                    progress_beta.progress(min((i + 1) / 300, 1.0))
            progress_beta.empty()
            if not tickers_beta:
                st.warning("Aucun titre avec beta > 1.5 trouvé dans l'échantillon. Analyse de l'univers complet.")
                tickers_beta = tickers_candidats
            st.info(f"{len(tickers_beta)} titres avec beta > 1.5 prêts pour le scan continu.")

        # Scan continu par lots
        resultats = []
        offset = 0
        stop_placeholder = st.empty()
        status = st.empty()
        progress = st.progress(0)

        while offset < len(tickers_beta):
            lot = tickers_beta[offset:offset + taille_lot]
            status.info(f"Analyse du lot {offset//taille_lot + 1} ({len(lot)} titres)…")

            for ticker in lot:
                for sens in sens_a_tester:
                    try:
                        res = analyser_ticker(ticker, sens)
                        resultats.append(res)
                    except Exception as e:
                        resultats.append({"ticker": ticker, "sens_demande": sens, "erreur": str(e)})

            # Vérifier s'il y a des OK
            ok_trouves = [r for r in resultats if r.get("A", {}).get("decision") == "OK"]
            if ok_trouves:
                status.success(f"{len(ok_trouves)} titre(s) ont passé le filtre A. Arrêt du scan continu.")
                break

            offset += taille_lot
            progress.progress(min(offset / len(tickers_beta), 1.0))

            # Bouton d'arrêt
            if stop_placeholder.button("⏹️ Arrêter le scan", key=f"stop_{offset}"):
                status.warning("Scan interrompu par l'utilisateur.")
                break

        progress.empty()
        stop_placeholder.empty()
        resultats_affiches = [r for r in resultats if r.get("A", {}).get("decision") == "OK"] or resultats[-50:]

    else:
        # Mode normal (manuel ou scanner simple)
        progress = st.progress(0)
        status = st.empty()
        resultats = []
        total = len(tickers) * len(sens_a_tester)
        done = 0

        for ticker in tickers:
            for sens in sens_a_tester:
                status.info(f"Analyse de **{ticker}** ({sens})… ({done+1}/{total})")
                try:
                    res = analyser_ticker(ticker, sens)
                    resultats.append(res)
                except Exception as e:
                    resultats.append({"ticker": ticker, "sens_demande": sens, "erreur": str(e)})
                done += 1
                progress.progress(min(done / total, 1.0))

        status.success(f"Analyse terminée — {len(resultats)} résultats")
        progress.empty()

        if mode != "Tickers manuels" and not afficher_tous:
            resultats_affiches = [r for r in resultats if r.get("A", {}).get("decision") == "OK"]
            if not resultats_affiches:
                st.warning("Aucun titre n'a passé le filtre A.")
            else:
                st.success(f"{len(resultats_affiches)} titre(s) ont passé le filtre A.")
        else:
            resultats_affiches = resultats

    # Affichage commun
    st.markdown("---")
    st.subheader("Résultats")

    for res in resultats_affiches:
        ticker = res.get("ticker", "?")
        sens = res.get("sens_demande", "?")
        a = res.get("A", {})
        b = res.get("B", {})
        c = res.get("C", {})

        decision_a = a.get("decision", "—")
        is_ok = decision_a == "OK" and b.get("decision") == "OK"
        card_class = "ok-card" if is_ok else "rejet-card"

        with st.container():
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown(f"**{ticker}** — {sens.capitalize()}")

            cols = st.columns(3)
            cols[0].metric("Filtre A", decision_a)
            cols[1].metric("Filtre B", b.get("decision", "—"))
            cols[2].metric("Filtre C", c.get("decision", "—") if c else "—")

            if decision_a == "REJET":
                st.caption(f"Motif A : {a.get('motif', '')} — {a.get('detail', '')}")
            else:
                pattern = a.get("pattern_description")
                if pattern:
                    st.info(f"Pattern détecté : {pattern}")
                st.caption(
                    f"Volume ratio : {a.get('ratio_volume', '—')} "
                    f"(seuil {a.get('seuil_volume_applique', 1.0)}) | "
                    f"Résultats dans {a.get('jours_avant_resultats', '—')}j | "
                    f"Beta : {a.get('beta', '—')}"
                )

            if b.get("decision") == "OK":
                st.success(
                    f"Score : {b.get('score')}/100 — État : {b.get('etat')} "
                    f"| ATR% : {b.get('atr_pct')} ({b.get('amplitude_attendue')})"
                )
                ext = b.get("score_externe")
                if ext:
                    st.caption(f"Score externe : {ext.get('note')} ({ext.get('source')})")

            if c.get("decision") == "OK":
                st.markdown("**Meilleur contrat**")
                st.json(c.get("meilleur_contrat", {}))

            st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        label="Télécharger les résultats (JSON)",
        data=json.dumps(resultats, indent=2, ensure_ascii=False, default=str),
        file_name=f"robot_booster_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

elif mode == "Tickers manuels":
    st.info("Saisissez des tickers et cliquez sur **Analyser**.")
elif mode == "Scanner le marché complet":
    st.info("Cliquez sur **Scanner le marché** pour analyser un lot de titres.")
else:
    st.info("Cliquez sur **Lancer le scanner continu** pour analyser automatiquement les titres à beta > 1.5, lot après lot, jusqu’à trouver des résultats.")

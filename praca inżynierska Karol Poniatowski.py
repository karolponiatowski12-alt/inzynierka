import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Model Marczuka - Porównywarka",
    layout="wide",
    page_icon=""
)

# Lista wszystkich wykresów które chciał dodać użytkownik

if 'runs' not in st.session_state:
    st.session_state.runs = []

COLORS = ['#00f2fe', '#ff0844', '#00ff00', '#ffeb3b', '#aa00ff', '#ff9800']


def simulate_marczuk(beta, alpha, tau, V0, t_max, gamma, mu_c, c_star, rho, mu_f, eta, sigma, mu_m, m_star):
    """
    Parapetry potrzebne do użycia modelu:

    Zmienne modelu:
    V - zagęszczenie antygenu / poziom infekcji
    C - aktywność komórek odpornościowych
    F - stężenie przeciwciał
    m - poziom uszkodzenia organu

    Parametry główne:
    beta - szybkość namnażania antygenu
    alpha - siła odpowiedzi immunologicznej
    tau - opóźnienie reakcji immunologicznej
    V0 - początkowa ilość antygenu

    Parametry fizjologiczne:
    gamma - skuteczność przeciwciał w neutralizacji antygenu
    mu_c - tempo powrotu C do poziomu bazowego
    c_star - bazowy poziom aktywności odpornościowej
    rho - tempo produkcji przeciwciał przez komórki odpornościowe
    mu_f - naturalny zanik przeciwciał
    eta - zużycie przeciwciał w walce z antygenem
    sigma - tempo narastania uszkodzenia organu
    mu_m - tempo regeneracji organu
    m_star - próg uszkodzenia organu
    """
#ilość i długość kroku, do wyznaczania wykresów
    dt = 0.005
    steps = int(t_max / dt)
    delay_steps = max(1, int(tau / dt))

    V = V0
    C = c_star
    F = (rho * c_star) / mu_f
    m = 0.0

# Zmienne pokazujące wartości dla danych z opóźnieniem, czyli przesunięcie w czasie zjawisk
    V_hist = np.ones(delay_steps) * V
    F_hist = np.ones(delay_steps) * F

# zmniejszenie ilosci puntów bez utraty wyników
    save_steps = steps // 10

    t_arr = np.zeros(save_steps)
    V_arr = np.zeros(save_steps)
    C_arr = np.zeros(save_steps)
    F_arr = np.zeros(save_steps)
    m_arr = np.zeros(save_steps)

    idx = 0

    for i in range(steps):

        if i % 10 == 0:
            t_arr[idx] = i * dt

            V_arr[idx] = max(V, 1e-6) #by nie przyjeło wartosci log(0)

            C_arr[idx] = C
            F_arr[idx] = F
            m_arr[idx] = m

            idx += 1

        # Pobranie wartości opóźnionych i aktualizacja historii
        V_delay = V_hist[i % delay_steps]
        F_delay = F_hist[i % delay_steps]
        V_hist[i % delay_steps] = V
        F_hist[i % delay_steps] = F

        # Funkcja osłabienia odporności xi(m)
        # Jeśli uszkodzenie organu jest mniejsze od progu uszkodzenia, odporność działa normalnie.
        # Po przekroczeniu progu m_star odporność zaczyna słabnąć.
        if m < m_star:
            xi = 1.0
        else:
            xi = max(0.0, 1.0 - (m - m_star) / (1.0 - m_star))


        dV = (beta - gamma * F) * V
        dC = alpha * xi * V_delay * F_delay - mu_c * (C - c_star)
        dF = rho * C - (mu_f + eta * gamma * V) * F
        dm = sigma * V - mu_m * m

        V += dV * dt
        C += dC * dt
        F += dF * dt
        m += dm * dt

        # Ograniczenia 
        V = max(0, V)
        C = max(0, C)
        F = max(0, F)
        m = max(0, min(1, m))

    return t_arr, V_arr, C_arr, F_arr, m_arr


st.sidebar.title("Parametry symulacji")
st.sidebar.subheader("Parametry główne")

beta = st.sidebar.slider(
    "Szybkość namnażania wirusa (β)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1
)

alpha = st.sidebar.slider(
    "Siła odpowiedzi immunologicznej (α)",
    min_value=0.5,
    max_value=15.0,
    value=5.0,
    step=0.5
)

tau = st.sidebar.slider(
    "Opóźnienie reakcji (τ)",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.1
)

V0 = st.sidebar.number_input(
    "Początkowa ilość antygenu (V₀)",
    value=0.1,
    step=0.1
)


with st.sidebar.expander("Parametry fizjologiczne modelu", expanded=False):

    gamma = st.slider(
        "Skuteczność przeciwciał (γ)",
        min_value=0.1,
        max_value=3.0,
        value=0.8,
        step=0.1
    )

    mu_c = st.slider(
        "Tempo powrotu C do poziomu bazowego (μc)",
        min_value=0.1,
        max_value=3.0,
        value=0.5,
        step=0.1
    )

    c_star = st.slider(
        "Bazowy poziom odporności (c*)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    rho = st.slider(
        "Tempo produkcji przeciwciał (ρ)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    mu_f = st.slider(
        "Naturalny zanik przeciwciał (μf)",
        min_value=0.1,
        max_value=3.0,
        value=0.2,
        step=0.1
    )

    eta = st.slider(
        "Zużycie przeciwciał w walce z antygenem (η)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    sigma = st.slider(
        "Tempo uszkodzenia organu (σ)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    mu_m = st.slider(
        "Tempo regeneracji organu (μm)",
        min_value=0.1,
        max_value=3.0,
        value=0.5,
        step=0.1
    )

    m_star = st.slider(
        "Próg uszkodzenia organu (m*)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )


col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Dodaj przebieg", use_container_width=True):

        # Wywołanie funkcji
        t, V, C, F, m = simulate_marczuk(
            beta=beta,
            alpha=alpha,
            tau=tau,
            V0=V0,
            t_max=40,
            gamma=gamma,
            mu_c=mu_c,
            c_star=c_star,
            rho=rho,
            mu_f=mu_f,
            eta=eta,
            sigma=sigma,
            mu_m=mu_m,
            m_star=m_star
        )

        label = (
            f"β={beta}, α={alpha}, τ={tau}, V₀={V0}, "
            f"γ={gamma}, μc={mu_c}, ρ={rho}, μf={mu_f}"
        )

        st.session_state.runs.append({
            'label': label,
            't': t,
            'V': V,
            'C': C,
            'F': F,
            'm': m
        })

with col2:
    if st.button("Wyczyść", use_container_width=True):
        st.session_state.runs = []
        st.rerun()


st.title("Wielowariantowa Analiza Modelu Marczuka")


st.markdown(
    """
    Aplikacja została utworzona w ramach realizacji pracy dyplomowej na Wydziale Informatyki Politechniki Białostockiej.
    Aplikacja umożliwia porównywanie wielu wariantów symulacji modelu.
    Każdy przebieg odpowiada jednemu zestawowi parametrów wybranemu w panelu bocznym.
    """
)

if not st.session_state.runs:

    st.info(
        "Ustaw parametry i kliknij **Dodaj przebieg**, aby rozpocząć rysowanie wykresów."
    )

else:

#rozmairy wykresów
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Zagęszczenie antygenu (V) - skala log",
            "Aktywność komórek odpornościowych (C)",
            "Stężenie przeciwciał (F)",
            "Uszkodzenie organu (m)",
            "Wykres fazowy V-F",
            "Wykres fazowy C-F"
        )
    )

    # Rysowanie wszystkich przebiegów
    for idx, run in enumerate(st.session_state.runs):

        color = COLORS[idx % len(COLORS)]

        # 1. Antygen V
        fig.add_trace(
            go.Scatter(
                x=run['t'],
                y=run['V'],
                mode='lines',
                name=run['label'],
                line=dict(color=color, width=2),
                legendgroup=str(idx)
            ),
            row=1,
            col=1
        )

        # 2. Komórki odpornościowe C
        fig.add_trace(
            go.Scatter(
                x=run['t'],
                y=run['C'],
                mode='lines',
                name=run['label'],
                line=dict(color=color, width=2),
                legendgroup=str(idx),
                showlegend=False
            ),
            row=1,
            col=2
        )

        # 3. Przeciwciała F
        fig.add_trace(
            go.Scatter(
                x=run['t'],
                y=run['F'],
                mode='lines',
                name=run['label'],
                line=dict(color=color, width=2),
                legendgroup=str(idx),
                showlegend=False
            ),
            row=2,
            col=1
        )

        # 4. Uszkodzenie organu m
        fig.add_trace(
            go.Scatter(
                x=run['t'],
                y=run['m'],
                mode='lines',
                name=run['label'],
                line=dict(color=color, width=2),
                legendgroup=str(idx),
                showlegend=False
            ),
            row=2,
            col=2
        )

        # 5. Wykres fazowy V-F
        fig.add_trace(
            go.Scatter(
                x=run['F'],
                y=run['V'],
                mode='lines',
                name=run['label'],
                line=dict(color=color, width=2),
                legendgroup=str(idx),
                showlegend=False
            ),
            row=3,
            col=1
        )

        # 6. Wykres fazowy C-F
        fig.add_trace(
            go.Scatter(
                x=run['F'],
                y=run['C'],
                mode='lines',
                name=run['label'],
                line=dict(color=color, width=2),
                legendgroup=str(idx),
                showlegend=False
            ),
            row=3,
            col=2
        )

    fig.update_yaxes(
        title_text="Zagęszczenie V",
        type="log",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Aktywność C",
        row=1,
        col=2
    )

    fig.update_yaxes(
        title_text="Stężenie F",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="Frakcja uszkodzenia m",
        range=[0, 1.05],
        row=2,
        col=2
    )

    fig.update_yaxes(
        title_text="Antygen V",
        type="log",
        row=3,
        col=1
    )

    fig.update_yaxes(
        title_text="Aktywność C",
        row=3,
        col=2
    )

    fig.update_xaxes(
        title_text="Czas (dni)",
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text="Czas (dni)",
        row=1,
        col=2
    )

    fig.update_xaxes(
        title_text="Czas (dni)",
        row=2,
        col=1
    )

    fig.update_xaxes(
        title_text="Czas (dni)",
        row=2,
        col=2
    )

    fig.update_xaxes(
        title_text="Przeciwciała F",
        row=3,
        col=1
    )

    fig.update_xaxes(
        title_text="Przeciwciała F",
        row=3,
        col=2
    )

    fig.update_layout(
        height=1100,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)
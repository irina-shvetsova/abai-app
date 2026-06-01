"""
AB·AI — Streamlit MVP
Автоматизация этапов A/B-тестирования с использованием методов ИИ.

Запуск:
    pip install streamlit anthropic scipy numpy matplotlib pandas
    streamlit run app.py

Структура приложения:
    1. Генерация гипотез  — HypothesisGenerator (Anthropic/YandexGPT)
    2. Планирование теста — calculate_sample_size (scipy.stats)
    3. Симуляция (Thompson Sampling) — ThompsonSamplingBandit
    4. Отчёт              — ResultsInterpreter + LLM-резюме
"""

import sys
import os
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import streamlit as st

# ---------------------------------------------------------------------------
# Встроенные версии модулей (не требуют отдельных файлов в той же папке)
# Если hypothesis_generator.py / adaptive_testing.py / results_interpreter.py
# лежат рядом — закомментируй эти блоки и раскомментируй импорты ниже.
# ---------------------------------------------------------------------------
# from hypothesis_generator import HypothesisGenerator, GenerationResult
# from adaptive_testing import ThompsonSamplingBandit, run_adaptive_experiment
# from results_interpreter import ResultsInterpreter, calculate_sample_size

# ============================================================
# БЛОК 1: Встроенный calculate_sample_size (без LLM)
# ============================================================

@dataclass
class SampleSizePlan:
    required_observations_per_group: int
    baseline_conversion: float
    minimum_detectable_effect: float
    target_conversion: float
    significance_level: float
    statistical_power: float
    estimated_duration_days: Optional[int]
    metric_type: str
    total_required_observations: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_required_observations = self.required_observations_per_group * 2


def calculate_sample_size(
    baseline_conversion: float,
    minimum_detectable_effect: float,
    significance_level: float = 0.05,
    statistical_power: float = 0.80,
    daily_traffic: Optional[int] = None,
) -> SampleSizePlan:
    z_alpha = stats.norm.ppf(1 - significance_level / 2)
    z_beta = stats.norm.ppf(statistical_power)
    z_sq = (z_alpha + z_beta) ** 2
    target_cr = baseline_conversion + minimum_detectable_effect
    variance_sum = (
        baseline_conversion * (1 - baseline_conversion)
        + target_cr * (1 - target_cr)
    )
    raw_n = z_sq * variance_sum / (minimum_detectable_effect ** 2)
    n_per_group = math.ceil(raw_n)
    duration = math.ceil(n_per_group * 2 / daily_traffic) if daily_traffic else None
    return SampleSizePlan(
        required_observations_per_group=n_per_group,
        baseline_conversion=baseline_conversion,
        minimum_detectable_effect=minimum_detectable_effect,
        target_conversion=target_cr,
        significance_level=significance_level,
        statistical_power=statistical_power,
        estimated_duration_days=duration,
        metric_type="proportional",
    )


# ============================================================
# БЛОК 2: Встроенный Thompson Sampling
# ============================================================

@dataclass
class ArmState:
    label: str
    success_count: int = 1
    failure_count: int = 1

    @property
    def total_observations(self) -> int:
        return (self.success_count - 1) + (self.failure_count - 1)

    @property
    def empirical_cr(self) -> float:
        denom = self.success_count + self.failure_count - 2
        return 0.0 if denom == 0 else (self.success_count - 1) / denom

    @property
    def posterior_mean(self) -> float:
        return self.success_count / (self.success_count + self.failure_count)


def run_thompson_simulation(
    true_rates: list[float],
    arm_labels: list[str],
    visitor_count: int = 10_000,
    snapshot_every: int = 100,
    seed: int = 42,
) -> dict:
    """Симулирует Thompson Sampling и возвращает словарь с метриками."""
    rng_bandit = np.random.default_rng(seed=seed)
    rng_sim = np.random.default_rng(seed=seed + 1)

    arms = [ArmState(label=lbl) for lbl in arm_labels]
    optimal_rate = max(true_rates)
    optimal_idx = int(np.argmax(true_rates))

    steps_log, regret_log, traffic_log = [], [], []
    cumulative_selections = [0] * len(arms)
    running_regret = 0.0

    for step in range(1, visitor_count + 1):
        # Сэмплируем θ из Beta(α, β) для каждого варианта
        samples = [rng_bandit.beta(a.success_count, a.failure_count) for a in arms]
        chosen = int(np.argmax(samples))

        # Симулируем исход
        converted = bool(rng_sim.random() < true_rates[chosen])

        # Обновляем posterior
        if converted:
            arms[chosen].success_count += 1
        else:
            arms[chosen].failure_count += 1

        cumulative_selections[chosen] += 1
        running_regret += optimal_rate - true_rates[chosen]

        if step % snapshot_every == 0 or step == visitor_count:
            steps_log.append(step)
            regret_log.append(running_regret)
            traffic_log.append([c / step for c in cumulative_selections])

    # Классический A/B: равномерное 50/50
    classic_regret_series = []
    classic_running = 0.0
    rng_classic = np.random.default_rng(seed=seed + 99)
    n_arms = len(arms)
    for step in range(1, visitor_count + 1):
        chosen = rng_classic.integers(0, n_arms)
        classic_running += optimal_rate - true_rates[chosen]
        if step % snapshot_every == 0 or step == visitor_count:
            classic_regret_series.append(classic_running)

    winner_idx = int(np.argmax([a.posterior_mean for a in arms]))

    return {
        "steps": steps_log,
        "ts_regret": regret_log,
        "classic_regret": classic_regret_series,
        "traffic_shares": traffic_log,
        "final_arms": arms,
        "winner_label": arm_labels[winner_idx],
        "winner_idx": winner_idx,
        "optimal_idx": optimal_idx,
        "total_regret_ts": running_regret,
        "total_regret_classic": classic_running,
    }


# ============================================================
# БЛОК 3: LLM-клиент (Anthropic или YandexGPT)
# ============================================================

def _call_anthropic(api_key: str, prompt: str, system: str = "") -> str:
    """Вызов Claude через Anthropic SDK."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": "claude-sonnet-4-20250514", "max_tokens": 1500, "messages": messages}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text
    except Exception as exc:
        return f"[Ошибка Anthropic API: {exc}]"


def _call_yandex(folder_id: str, api_key: str, prompt: str, system: str = "") -> str:
    """Вызов YandexGPT через REST API (Foundation Models)."""
    try:
        import urllib.request
        messages = []
        if system:
            messages.append({"role": "system", "text": system})
        messages.append({"role": "user", "text": prompt})
        payload = json.dumps({
            "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": 1500},
            "messages": messages,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Api-Key {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["result"]["alternatives"][0]["message"]["text"]
    except Exception as exc:
        return f"[Ошибка YandexGPT API: {exc}]"


def call_llm(prompt: str, system: str = "") -> str:
    """Вызов YandexGPT. Ключи: st.secrets (продакшн) → st.session_state (локально)."""
    folder_id = (
        st.secrets.get("YANDEX_FOLDER_ID", "")
        or st.session_state.get("yandex_folder_id", "")
    )
    api_key = (
        st.secrets.get("YANDEX_API_KEY", "")
        or st.session_state.get("yandex_api_key", "")
    )
    if not folder_id or not api_key:
        return "[Не указаны YANDEX_FOLDER_ID или YANDEX_API_KEY — перейди в раздел Настройки]"
    return _call_yandex(folder_id, api_key, prompt, system)


# ============================================================
# БЛОК 4: Промпты
# ============================================================

def build_hypothesis_prompt(
    product_name: str,
    target_metric: str,
    baseline_cr: float,
    problem_area: str,
    audience: str,
    history: str,
    count: int,
) -> str:
    return f"""Ты — опытный продуктовый аналитик. Сформулируй {count} конкретных, тестируемых гипотез для A/B-теста.

КОНТЕКСТ:
- Продукт: {product_name}
- Проблема: {problem_area}
- Метрика: {target_metric}
- Базовая конверсия: {baseline_cr:.1%}
- Аудитория: {audience if audience else 'не указана'}

ИСТОРИЯ ТЕСТОВ:
{history if history else 'Нет данных об истории тестов.'}

ТРЕБОВАНИЯ:
1. Одно конкретное изменение на гипотезу.
2. Эффект измерим по метрике.
3. Не повторять уже протестированные изменения.

Верни ТОЛЬКО JSON-массив без текста вокруг:
[
  {{
    "title": "Краткое название (≤60 символов)",
    "change": "Что именно меняется",
    "expected": "Ожидаемый эффект (+X%)",
    "reason": "Обоснование",
    "confidence": 0.75,
    "priority": "высокий"
  }}
]"""


def build_report_prompt(stats_json: dict) -> str:
    return f"""Ты — аналитик данных. Напиши бизнес-резюме результатов A/B-теста на русском языке.

ДАННЫЕ:
{json.dumps(stats_json, ensure_ascii=False, indent=2)}

Структура ответа (3–5 предложений):
1. Итог: победил ли вариант B, насколько значимо.
2. Практический смысл: что это означает для бизнеса.
3. Рекомендация: внедрить / не внедрять / продолжить тест.

Только текст, без заголовков и списков."""


# ============================================================
# STREAMLIT: КОНФИГУРАЦИЯ И СТИЛИ
# ============================================================

st.set_page_config(
    page_title="AB·AI — A/B-тестирование с ИИ",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# Акцентный цвет: индиго #4F46E5 (один на всё приложение)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* ── ГЛОБАЛЬНЫЕ СБРОСЫ ── */
    #MainMenu, footer, header { visibility: hidden; }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    /* ── ФОНЫ ── */
    .stApp, .stApp > div, .main { background: #ffffff !important; }

    /* ── САЙДБАР ── */
    [data-testid="stSidebar"] {
        background: #f9f9fb !important;
        border-right: 1px solid #e5e5ea !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #1c1c1e !important;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown em { color: #8e8e93 !important; font-size: 12px !important; }
    [data-testid="stSidebar"] hr { border-color: #e5e5ea !important; }

    /* Radio — минималистичная навигация */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        color: #3a3a3c !important;
        cursor: pointer;
        display: block !important;
        width: 100% !important;
        margin-bottom: 2px !important;
        transition: background .15s !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #ededf0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #eef0fd !important;
        color: #4F46E5 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] p { color: inherit !important; font-size: inherit !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] svg { display: none !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label > div:first-child { display: none !important; }

    /* ── ТИПОГРАФИКА ── */
    h1 { font-size: 22px !important; font-weight: 600 !important; color: #1c1c1e !important; letter-spacing: -0.02em !important; margin-bottom: 4px !important; }
    h2 { font-size: 17px !important; font-weight: 600 !important; color: #1c1c1e !important; letter-spacing: -0.01em !important; }
    h3 { font-size: 14px !important; font-weight: 500 !important; color: #3a3a3c !important; }
    p, label, .stMarkdown { color: #3a3a3c !important; font-size: 14px !important; }

    /* ── КНОПКИ ── */
    .stButton > button {
        background: #4F46E5 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 1px 3px rgba(79,70,229,.25) !important;
        transition: all .15s !important;
    }
    .stButton > button:hover {
        background: #4338CA !important;
        box-shadow: 0 2px 8px rgba(79,70,229,.35) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ── ИНПУТЫ ── */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: #ffffff !important;
        color: #1c1c1e !important;
        border: 1px solid #d1d1d6 !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,.12) !important;
        outline: none !important;
    }
    .stTextInput label, .stNumberInput label, .stTextArea label,
    .stSelectbox label, .stSlider label {
        color: #3a3a3c !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* ── SELECTBOX ── */
    [data-testid="stSelectbox"] > div {
        background: #ffffff !important;
        border: 1px solid #d1d1d6 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSelectbox"] > div > div { color: #1c1c1e !important; }

    /* ── СЛАЙДЕРЫ ── */
    [data-baseweb="slider"] [role="slider"] { background: #4F46E5 !important; }

    /* ── МЕТРИКИ ── */
    [data-testid="metric-container"] {
        background: #f9f9fb !important;
        border: 1px solid #e5e5ea !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        box-shadow: none !important;
    }
    [data-testid="metric-container"] label {
        color: #8e8e93 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #1c1c1e !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="metric-container"] [data-testid="metric-delta"] { color: #34C759 !important; font-size: 12px !important; }

    /* ── СПИННЕР ── */
    .stSpinner > div { border-top-color: #4F46E5 !important; }

    /* ── АЛЕРТЫ ── */
    .stSuccess { background: #f0fdf4 !important; border: 1px solid #bbf7d0 !important; color: #166534 !important; border-radius: 10px !important; }
    .stError   { background: #fff1f2 !important; border: 1px solid #fecdd3 !important; color: #9f1239 !important; border-radius: 10px !important; }
    .stInfo    { background: #eef0fd !important; border: 1px solid #c7d2fe !important; color: #3730a3 !important; border-radius: 10px !important; }
    .stWarning { background: #fffbeb !important; border: 1px solid #fde68a !important; color: #92400e !important; border-radius: 10px !important; }

    /* ── КАСТОМНЫЕ КОМПОНЕНТЫ ── */
    .info-box {
        background: #eef0fd;
        border-left: 3px solid #4F46E5;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 13px;
        color: #3730a3;
        margin: 8px 0;
        line-height: 1.5;
    }
    .hypo-card {
        background: #ffffff;
        border: 1px solid #e5e5ea;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
        transition: box-shadow .15s;
    }
    .hypo-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.1); }
    .hypo-title {
        font-size: 14px;
        font-weight: 600;
        color: #1c1c1e;
        margin-bottom: 6px;
    }
    .hypo-change { font-size: 13px; color: #3a3a3c; margin-bottom: 4px; line-height: 1.5; }
    .hypo-meta   { font-size: 12px; color: #8e8e93; margin-top: 4px; }
    .badge-high { background: #eef0fd; color: #4F46E5; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 500; }
    .badge-mid  { background: #fff7ed; color: #c2410c; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 500; }
    .badge-low  { background: #f4f4f5; color: #71717a; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 500; }
    .llm-summary {
        background: #f9f9fb;
        border-left: 3px solid #4F46E5;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-size: 14px;
        line-height: 1.7;
        color: #3a3a3c;
        margin-top: 8px;
    }

    /* ── РАЗДЕЛИТЕЛИ ── */
    hr, .stDivider { border-color: #e5e5ea !important; }

    /* ── ГРАФИКИ matplotlib — светлый фон ── */
    .stImage img { border-radius: 10px; border: 1px solid #e5e5ea; }

    /* ── КОЛЛАПС САЙДБАРА ── */
    [data-testid="collapsedControl"] {
        background: #f9f9fb !important;
        border-right: 1px solid #e5e5ea !important;
    }
</style>
""", unsafe_allow_html=True)

# Принудительно открываем сайдбар через JS
st.markdown("""
<script>
    // Открываем сайдбар если он закрыт
    window.addEventListener('load', function() {
        setTimeout(function() {
            const btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (btn) btn.click();
        }, 500);
    });
</script>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## AB·AI")
    st.markdown("*Автоматизация A/B-тестов*")
    st.divider()

    page = st.radio(
        "// навигация",
        options=["Гипотезы", "Планирование", "Симуляция", "Отчёт"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown(
        "<div style='font-size:11px; color:#999;'>Курсовая работа · 2025<br>"
        "Автоматизация A/B-тестирования с ИИ</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# СТРАНИЦА 1: ГЕНЕРАЦИЯ ГИПОТЕЗ
# ============================================================

if page == "Гипотезы":
    st.markdown("### Генерация гипотез")
    st.markdown(
        "<div class='info-box'>LLM анализирует контекст продукта и историю тестов, "
        "формулирует конкретные, тестируемые гипотезы с оценкой уверенности.</div>",
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("**Контекст продукта**")
        product_name = st.text_input("Продукт / экран", value="Ozon Express — корзина")
        target_metric = st.selectbox(
            "Целевая метрика",
            ["Конверсия в заказ", "CTR кнопки", "Средний чек", "Время до оплаты", "Другое"],
        )
        baseline_cr = st.slider("Базовая конверсия (%)", min_value=0.5, max_value=30.0, value=3.2, step=0.1) / 100
        problem_area = st.text_input("Проблемная зона", value="Кнопка оформления заказа")
        audience = st.text_input("Целевая аудитория (необязательно)", value="Мобильные пользователи 25–45 лет")
        hypo_count = st.slider("Количество гипотез", 2, 7, 3)

        st.markdown("**История предыдущих тестов (необязательно)**")
        history_raw = st.text_area(
            "Каждый тест с новой строки: название — результат",
            value="Изменение цвета кнопки — не победила\nДобавление таймера — победила, +8%",
            height=90,
        )

        generate_btn = st.button("✨ Сгенерировать гипотезы", use_container_width=True)

    with col_result:
        if generate_btn:
            with st.spinner("Запрос к LLM..."):
                prompt = build_hypothesis_prompt(
                    product_name=product_name,
                    target_metric=target_metric,
                    baseline_cr=baseline_cr,
                    problem_area=problem_area,
                    audience=audience,
                    history=history_raw,
                    count=hypo_count,
                )
                raw = call_llm(prompt)

            # Парсим JSON-ответ
            try:
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = "\n".join(cleaned.split("\n")[1:])
                    cleaned = cleaned.rstrip("`").strip()
                hypotheses = json.loads(cleaned)
                st.session_state["last_hypotheses"] = hypotheses
            except Exception:
                st.warning("LLM вернул неструктурированный ответ. Показываем как текст.")
                st.text_area("Ответ LLM", raw, height=300)
                hypotheses = []

            if hypotheses:
                st.markdown(f"**Сгенерировано: {len(hypotheses)} гипотез**")
                for i, h in enumerate(hypotheses, 1):
                    priority = h.get("priority", "средний")
                    badge_class = "badge-high" if priority == "высокий" else ("badge-mid" if priority == "средний" else "badge-low")
                    conf = h.get("confidence", 0.5)
                    st.markdown(f"""
<div class="hypo-card">
  <div class="hypo-title">#{i} &nbsp; {h.get('title', '—')}</div>
  <div class="hypo-change">→ {h.get('change', '')}</div>
  <div class="hypo-change" style="color:#534AB7">{h.get('expected', '')}</div>
  <div class="hypo-meta" style="margin-top:6px; font-style:italic">{h.get('reason', '')}</div>
  <div style="margin-top:8px">
    <span class="{badge_class}">{priority}</span>
    &nbsp; <span style="font-size:12px; color:#888">Уверенность: {conf:.0%}</span>
  </div>
</div>""", unsafe_allow_html=True)

        elif "last_hypotheses" in st.session_state:
            st.markdown("*Последний результат:*")
            for i, h in enumerate(st.session_state["last_hypotheses"], 1):
                conf = h.get("confidence", 0.5)
                st.markdown(f"""
<div class="hypo-card">
  <div class="hypo-title">#{i} &nbsp; {h.get('title', '—')}</div>
  <div class="hypo-change">→ {h.get('change', '')}</div>
  <div class="hypo-meta" style="margin-top:6px; font-style:italic">{h.get('reason', '')}</div>
  <div style="margin-top:6px"><span style="font-size:12px; color:#888">Уверенность: {conf:.0%}</span></div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#aaa; font-size:13px; padding-top:40px; text-align:center'>"
                "Заполни форму слева и нажми «Сгенерировать»</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# СТРАНИЦА 2: ПЛАНИРОВАНИЕ ТЕСТА
# ============================================================

elif page == "Планирование":
    st.markdown("### Планирование теста")
    st.markdown(
        "<div class='info-box'>Рассчитывается минимально необходимая выборка по формуле "
        "нормального приближения к биномиальному распределению. Ошибка I рода (α) и мощность (1−β) "
        "задаются вручную.</div>",
        unsafe_allow_html=True,
    )

    col_params, col_plan = st.columns([1, 1], gap="large")

    with col_params:
        st.markdown("**Параметры теста**")
        p_baseline = st.slider("Базовая конверсия (%)", 0.5, 30.0, 3.2, 0.1) / 100
        mde_pct = st.slider("MDE — минимальный эффект (абс., п.п.)", 0.1, 5.0, 0.8, 0.1)
        mde = mde_pct / 100

        alpha = st.select_slider("Уровень значимости α", options=[0.01, 0.05, 0.10], value=0.05)
        power = st.select_slider("Мощность теста 1−β", options=[0.70, 0.80, 0.90], value=0.80)
        daily_traffic = st.number_input("Суточный трафик (сессий/день)", min_value=100, value=1000, step=100)

        calc_btn = st.button("📊 Рассчитать", use_container_width=True)

    with col_plan:
        if calc_btn or True:  # показываем расчёт сразу при изменении слайдеров
            plan = calculate_sample_size(
                baseline_conversion=p_baseline,
                minimum_detectable_effect=mde,
                significance_level=alpha,
                statistical_power=power,
                daily_traffic=int(daily_traffic),
            )
            st.session_state["last_plan"] = plan

            st.markdown("**Результат**")
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m1.metric("Выборка (группа)", f"{plan.required_observations_per_group:,}")
            m2.metric("Всего наблюдений", f"{plan.total_required_observations:,}")
            m3.metric("Длительность", f"{plan.estimated_duration_days} дн." if plan.estimated_duration_days else "—")
            m4.metric("Целевая конверсия B", f"{plan.target_conversion:.2%}")

            # Power analysis plot
            st.markdown("**Power analysis: выборка vs MDE**")
            mde_range = np.linspace(0.002, 0.05, 40)
            n_range = []
            for m in mde_range:
                try:
                    p = calculate_sample_size(p_baseline, m, alpha, power)
                    n_range.append(p.required_observations_per_group)
                except Exception:
                    n_range.append(np.nan)

            fig, ax = plt.subplots(figsize=(5.5, 3))
            ax.plot([m * 100 for m in mde_range], n_range, color="#534AB7", linewidth=2)
            ax.axvline(mde * 100, color="#D85A30", linestyle="--", linewidth=1.5, label=f"Выбранный MDE = {mde_pct:.1f}%")
            ax.set_xlabel("MDE (абс., п.п.)", fontsize=11)
            ax.set_ylabel("Наблюдений / группа", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ============================================================
# СТРАНИЦА 3: СИМУЛЯЦИЯ THOMPSON SAMPLING
# ============================================================

elif page == "Симуляция":
    st.markdown("### Адаптивное тестирование")
    st.markdown(
        "<div class='info-box'>Симуляция Multi-Armed Bandit: каждый вариант моделируется "
        "Beta(α, β)-распределением. Трафик автоматически перераспределяется в пользу "
        "лидирующего варианта — в отличие от фиксированного 50/50 в классическом A/B.</div>",
        unsafe_allow_html=True,
    )

    col_cfg, col_vis = st.columns([1, 1.5], gap="large")

    with col_cfg:
        st.markdown("**Параметры симуляции**")
        cr_a = st.slider("Истинная конверсия варианта A (%)", 1.0, 20.0, 3.2, 0.1) / 100
        cr_b = st.slider("Истинная конверсия варианта B (%)", 1.0, 20.0, 4.0, 0.1) / 100
        n_visitors = st.select_slider(
            "Число посетителей",
            options=[1_000, 5_000, 10_000, 25_000, 50_000],
            value=10_000,
        )
        sim_seed = st.number_input("Seed (воспроизводимость)", min_value=0, value=42, step=1)

        sim_btn = st.button("▶ Запустить симуляцию", use_container_width=True)

    with col_vis:
        run_key = f"{cr_a}_{cr_b}_{n_visitors}_{sim_seed}"
        if sim_btn or st.session_state.get("sim_run_key") == run_key:
            if sim_btn:
                with st.spinner("Симуляция..."):
                    result = run_thompson_simulation(
                        true_rates=[cr_a, cr_b],
                        arm_labels=["Вариант A", "Вариант B"],
                        visitor_count=n_visitors,
                        snapshot_every=max(1, n_visitors // 100),
                        seed=int(sim_seed),
                    )
                st.session_state["sim_result"] = result
                st.session_state["sim_run_key"] = run_key

            result = st.session_state.get("sim_result")
            if result:
                m1, m2, m3 = st.columns(3)
                regret_reduction = (result["total_regret_classic"] - result["total_regret_ts"]) / max(result["total_regret_classic"], 1e-9) * 100
                winner_cr = [cr_a, cr_b][result["winner_idx"]]
                m1.metric("Снижение regret", f"{regret_reduction:.1f}%", "vs классический A/B")
                m2.metric("Победитель", result["winner_label"])
                m3.metric("Posterior mean победителя", f"{winner_cr:.2%}")

                steps = result["steps"]
                ts_r = result["ts_regret"]
                cl_r = result["classic_regret"]
                traffic = result["traffic_shares"]
                traffic_b = [t[1] for t in traffic]

                def _style_ax(a):
                    a.set_facecolor("#ffffff")
                    a.spines["top"].set_visible(False)
                    a.spines["right"].set_visible(False)
                    a.spines["left"].set_color("#e5e5ea")
                    a.spines["bottom"].set_color("#e5e5ea")
                    a.tick_params(colors="#8e8e93")
                    a.grid(True, alpha=0.4, color="#e5e5ea")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), facecolor="#ffffff")
                _style_ax(ax1); _style_ax(ax2)

                ax1.plot(steps, ts_r, color="#4F46E5", linewidth=2, label="Thompson Sampling")
                ax1.plot(steps, cl_r, color="#EF4444", linewidth=1.5, linestyle="--", label="Классический A/B")
                ax1.set_title("Cumulative regret", fontsize=12, color="#1c1c1e", fontweight="600")
                ax1.set_xlabel("Посетители", color="#8e8e93")
                ax1.set_ylabel("Regret", color="#8e8e93")
                ax1.legend(fontsize=9, framealpha=0)

                ax2.plot(steps, [t * 100 for t in traffic_b], color="#4F46E5", linewidth=2, label="Вариант B")
                ax2.plot(steps, [(1 - t) * 100 for t in traffic_b], color="#34C759", linewidth=2, label="Вариант A")
                ax2.axhline(50, color="#d1d1d6", linestyle="--", linewidth=1)
                ax2.set_title("Доля трафика (%)", fontsize=12, color="#1c1c1e", fontweight="600")
                ax2.set_xlabel("Посетители", color="#8e8e93")
                ax2.set_ylabel("%", color="#8e8e93")
                ax2.legend(fontsize=9, framealpha=0)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("**Posterior Beta-распределения (финал)**")
                fig2, ax3 = plt.subplots(figsize=(9, 2.8), facecolor="#ffffff")
                _style_ax(ax3)
                x = np.linspace(0, 0.12, 500)
                colors_plot = ["#34C759", "#4F46E5"]
                labels_plot = ["Вариант A", "Вариант B"]
                for arm_idx, arm in enumerate(result["final_arms"]):
                    y = stats.beta.pdf(x, arm.success_count, arm.failure_count)
                    ax3.plot(x, y, color=colors_plot[arm_idx], linewidth=2, label=f"{labels_plot[arm_idx]} (mean={arm.posterior_mean:.3%})")
                    ax3.fill_between(x, y, alpha=0.12, color=colors_plot[arm_idx])
                ax3.set_xlabel("Конверсия θ", color="#8e8e93")
                ax3.set_ylabel("Плотность", color="#8e8e93")
                ax3.legend(fontsize=10, framealpha=0)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
        else:
            st.markdown(
                "<div style='color:#aaa; font-size:13px; padding-top:40px; text-align:center'>"
                "Задай параметры и нажми «Запустить симуляцию»</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# СТРАНИЦА 4: ОТЧЁТ
# ============================================================

elif page == "Отчёт":
    st.markdown("### Анализ результатов")
    st.markdown(
        "<div class='info-box'>Введи сырые результаты теста — система посчитает статистику "
        "и сгенерирует бизнес-резюме через LLM.</div>",
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("**Результаты теста**")
        ctrl_visitors = st.number_input("Контроль — посетители", min_value=10, value=3000)
        ctrl_conv = st.number_input("Контроль — конверсии", min_value=0, value=150)
        treat_visitors = st.number_input("Вариант B — посетители", min_value=10, value=3000)
        treat_conv = st.number_input("Вариант B — конверсии", min_value=0, value=185)
        metric_name = st.text_input("Название метрики", value="конверсия в заказ")
        daily_rev_per_conv = st.number_input(
            "Средняя выручка с конверсии (₽)", min_value=0, value=2500, step=100,
            help="Используется для расчёта денежного эффекта"
        )
        report_btn = st.button("🔍 Рассчитать и сгенерировать отчёт", use_container_width=True)

    with col_out:
        if report_btn:
            # Статистика
            p_ctrl = ctrl_conv / ctrl_visitors
            p_treat = treat_conv / treat_visitors
            lift_abs = p_treat - p_ctrl
            lift_rel = lift_abs / p_ctrl if p_ctrl > 0 else 0

            # z-test для двух пропорций
            pooled = (ctrl_conv + treat_conv) / (ctrl_visitors + treat_visitors)
            se = math.sqrt(pooled * (1 - pooled) * (1 / ctrl_visitors + 1 / treat_visitors))
            z_score = lift_abs / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            # Cohen's d (приближение для пропорций)
            cohens_d = lift_abs / math.sqrt(pooled * (1 - pooled)) if pooled > 0 else 0

            # 95% CI
            se_diff = math.sqrt(p_ctrl * (1 - p_ctrl) / ctrl_visitors + p_treat * (1 - p_treat) / treat_visitors)
            ci_lo = lift_abs - 1.96 * se_diff
            ci_hi = lift_abs + 1.96 * se_diff

            # Экономический эффект
            monthly_traffic = ctrl_visitors  # приближение: объём = месячный трафик
            extra_conversions = lift_abs * monthly_traffic
            monthly_revenue = extra_conversions * daily_rev_per_conv

            # Метрики
            m1, m2, m3, m4 = st.columns(4)
            sig_label = "✅ значимо" if p_value < 0.05 else "❌ не значимо"
            m1.metric("p-value", f"{p_value:.4f}", sig_label)
            m2.metric("Lift", f"{lift_rel:+.1%}")
            m3.metric("Cohen's d", f"{cohens_d:.3f}")
            m4.metric("95% CI (абс.)", f"[{ci_lo:+.3%}, {ci_hi:+.3%}]")

            st.metric("Доп. выручка/мес. (оценка)", f"{monthly_revenue:,.0f} ₽")

            # LLM-резюме
            stats_for_llm = {
                "метрика": metric_name,
                "конверсия_контроль": f"{p_ctrl:.3%}",
                "конверсия_B": f"{p_treat:.3%}",
                "lift_абсолютный": f"{lift_abs:+.3%}",
                "lift_относительный": f"{lift_rel:+.1%}",
                "p_value": round(p_value, 4),
                "cohens_d": round(cohens_d, 3),
                "CI_95": f"[{ci_lo:+.3%}, {ci_hi:+.3%}]",
                "значимость": "да" if p_value < 0.05 else "нет",
                "доп_выручка_в_месяц_руб": int(monthly_revenue),
            }

            with st.spinner("LLM генерирует бизнес-резюме..."):
                summary = call_llm(build_report_prompt(stats_for_llm))

            st.markdown("**LLM-резюме**")
            st.markdown(f"<div class='llm-summary'>{summary}</div>", unsafe_allow_html=True)

            st.session_state["last_report"] = {"stats": stats_for_llm, "summary": summary}

        elif "last_report" in st.session_state:
            r = st.session_state["last_report"]
            st.markdown("*Последний отчёт:*")
            st.json(r["stats"])
            st.markdown(f"<div class='llm-summary'>{r['summary']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#aaa; font-size:13px; padding-top:40px; text-align:center'>"
                "Введи данные теста и нажми «Рассчитать»</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# СТРАНИЦА 5: НАСТРОЙКИ
# ============================================================
"""
AB·AI — Streamlit MVP  (v2 — Premium Design + Onboarding)
Автоматизация этапов A/B-тестирования с использованием методов ИИ.

Запуск:
    pip install streamlit anthropic scipy numpy matplotlib pandas
    streamlit run app.py

Что нового в v2:
    • Визуал максимально приближён к landing_premium_v2.html
      (цвета, типографика, компоненты — единая дизайн-система)
    • Система онбординга для новых пользователей:
      - Приветственный экран при первом запуске
      - 5-шаговый guided tour с тултипами (Intro.js-style через JS)
      - Прогресс-бар, кнопки «Далее / Назад / Пропустить»
      - Состояние сохраняется в localStorage браузера
      - Кнопка «Показать тур снова» в сайдбаре
    • Улучшенный UX страниц: «живой» расчёт на странице Планирования,
      компактные метрики, отзывчивые поля ввода
"""

import sys
import os
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────
# БЛОК 1: Расчёт размера выборки
# ────────────────────────────────────────────────────────────────────────────

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
    z_beta  = stats.norm.ppf(statistical_power)
    z_sq    = (z_alpha + z_beta) ** 2
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


# ────────────────────────────────────────────────────────────────────────────
# БЛОК 2: Thompson Sampling
# ────────────────────────────────────────────────────────────────────────────

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
    rng_bandit = np.random.default_rng(seed=seed)
    rng_sim    = np.random.default_rng(seed=seed + 1)

    arms          = [ArmState(label=lbl) for lbl in arm_labels]
    optimal_rate  = max(true_rates)
    optimal_idx   = int(np.argmax(true_rates))

    steps_log, regret_log, traffic_log = [], [], []
    cumulative_selections = [0] * len(arms)
    running_regret = 0.0

    for step in range(1, visitor_count + 1):
        samples = [rng_bandit.beta(a.success_count, a.failure_count) for a in arms]
        chosen  = int(np.argmax(samples))
        converted = bool(rng_sim.random() < true_rates[chosen])

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
        "steps":                steps_log,
        "ts_regret":            regret_log,
        "classic_regret":       classic_regret_series,
        "traffic_shares":       traffic_log,
        "final_arms":           arms,
        "winner_label":         arm_labels[winner_idx],
        "winner_idx":           winner_idx,
        "optimal_idx":          optimal_idx,
        "total_regret_ts":      running_regret,
        "total_regret_classic": classic_running,
    }


# ────────────────────────────────────────────────────────────────────────────
# БЛОК 3: LLM-клиент (YandexGPT / Anthropic)
# ────────────────────────────────────────────────────────────────────────────

def _call_anthropic(api_key: str, prompt: str, system: str = "") -> str:
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        kwargs   = {"model": "claude-sonnet-4-20250514", "max_tokens": 1500, "messages": messages}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text
    except Exception as exc:
        return f"[Ошибка Anthropic API: {exc}]"


def _call_yandex(folder_id: str, api_key: str, prompt: str, system: str = "") -> str:
    try:
        import urllib.request
        messages = []
        if system:
            messages.append({"role": "system", "text": system})
        messages.append({"role": "user", "text": prompt})
        payload = json.dumps({
            "modelUri":         f"gpt://{folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.6, "maxTokens": 1500},
            "messages":         messages,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Api-Key {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["result"]["alternatives"][0]["message"]["text"]
    except Exception as exc:
        return f"[Ошибка YandexGPT API: {exc}]"


def call_llm(prompt: str, system: str = "") -> str:
    folder_id = (
        st.secrets.get("YANDEX_FOLDER_ID", "")
        or st.session_state.get("yandex_folder_id", "")
    )
    api_key = (
        st.secrets.get("YANDEX_API_KEY", "")
        or st.session_state.get("yandex_api_key", "")
    )
    if folder_id and api_key:
        return _call_yandex(folder_id, api_key, prompt, system)

    anthropic_key = (
        st.secrets.get("ANTHROPIC_API_KEY", "")
        or st.session_state.get("anthropic_api_key", "")
    )
    if anthropic_key:
        return _call_anthropic(anthropic_key, prompt, system)

    return "[Не указаны API-ключи — перейди в раздел Настройки]"


# ────────────────────────────────────────────────────────────────────────────
# БЛОК 4: Промпты
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# STREAMLIT: КОНФИГУРАЦИЯ
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AB·AI — A/B-тестирование с ИИ",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# ────────────────────────────────────────────────────────────────────────────
# CSS: Дизайн-система — полностью соответствует landing_premium_v2.html
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── СБРОС И ГЛОБАЛЬНЫЕ ── */
#MainMenu, footer, header { visibility: hidden; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased; }
.block-container { padding-top: 1.75rem !important; padding-bottom: 3rem !important;
                   max-width: 1160px !important; }

/* ── ФОНЫ ── */
.stApp, .stApp > div, .main { background: #ffffff !important; }

/* ── САЙДБАР — точно как в лендинге ── */
[data-testid="stSidebar"] {
    background: #F9FAFB !important;
    border-right: 1px solid #E5E7EB !important;
    min-width: 220px !important; max-width: 240px !important;
}

/* Логотип */
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 15px !important; font-weight: 600 !important;
    color: #0F0F10 !important; letter-spacing: -0.02em !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown em {
    color: #9CA3AF !important; font-size: 11px !important;
}
[data-testid="stSidebar"] hr { border-color: #E5E7EB !important; margin: 12px 0 !important; }

/* Навигационные пункты — Radio */
[data-testid="stSidebar"] [data-testid="stRadio"] { margin-top: 4px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    color: #4B5563 !important;
    cursor: pointer !important;
    display: block !important; width: 100% !important;
    margin-bottom: 2px !important;
    transition: background .15s !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #EDEDF0 !important; color: #0F0F10 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #EEF2FF !important;
    color: #4F46E5 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] p {
    color: inherit !important; font-size: inherit !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] svg { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label > div:first-child { display: none !important; }

/* Метка навигации */
[data-testid="stSidebar"] [data-testid="stRadio"] > div > label:first-child {
    font-size: 10px !important;
    font-weight: 600 !important;
    color: #9CA3AF !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    padding: 4px 14px 2px !important;
    cursor: default !important;
    background: transparent !important;
}

/* ── ТИПОГРАФИКА — точно как в лендинге ── */
h1 { font-size: 22px !important; font-weight: 600 !important; color: #0F0F10 !important;
     letter-spacing: -0.025em !important; line-height: 1.25 !important; margin-bottom: 6px !important; }
h2 { font-size: 17px !important; font-weight: 600 !important; color: #0F0F10 !important;
     letter-spacing: -0.015em !important; }
h3 { font-size: 14px !important; font-weight: 500 !important; color: #1c1c1e !important; }
p, label, div { font-size: 14px !important; color: #4B5563 !important; line-height: 1.6 !important; }

/* ── КНОПКИ — стиль лендинга ── */
.stButton > button {
    background: #0F0F10 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 1px 2px rgba(15,15,16,.08), 0 4px 16px rgba(15,15,16,.12) !important;
    transition: all .15s !important;
    letter-spacing: -0.01em !important;
}
.stButton > button:hover {
    background: #4F46E5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 1px 2px rgba(79,70,229,.1), 0 8px 24px rgba(79,70,229,.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── ИНПУТЫ — стиль лендинга ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #ffffff !important;
    color: #0F0F10 !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    padding: 9px 13px !important;
    transition: border-color .15s, box-shadow .15s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.12) !important;
    outline: none !important;
}
.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stSlider label {
    color: #4B5563 !important; font-size: 13px !important;
    font-weight: 500 !important; letter-spacing: -0.005em !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div {
    background: #ffffff !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] > div > div { color: #0F0F10 !important; }

/* ── SELECT_SLIDER ── */
[data-baseweb="slider"] [role="slider"] {
    background: #4F46E5 !important;
    border-color: #4F46E5 !important;
}
[data-baseweb="slider"] [data-testid="stSliderThumb"] {
    background: #4F46E5 !important;
}

/* ── МЕТРИКИ — карточки как в лендинге ── */
[data-testid="metric-container"] {
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    box-shadow: none !important;
}
[data-testid="metric-container"] label {
    color: #9CA3AF !important; font-size: 12px !important;
    font-weight: 500 !important; text-transform: none !important; letter-spacing: 0 !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #0F0F10 !important; font-size: 26px !important;
    font-weight: 700 !important; letter-spacing: -0.03em !important;
}
[data-testid="metric-container"] [data-testid="metric-delta"] {
    color: #059669 !important; font-size: 12px !important;
}

/* ── СПИННЕР ── */
.stSpinner > div { border-top-color: #4F46E5 !important; }

/* ── АЛЕРТЫ ── */
.stSuccess { background: #ECFDF5 !important; border: 1px solid #A7F3D0 !important;
             color: #065F46 !important; border-radius: 10px !important; }
.stError   { background: #FFF1F2 !important; border: 1px solid #FECDD3 !important;
             color: #9F1239 !important; border-radius: 10px !important; }
.stInfo    { background: #EEF2FF !important; border: 1px solid #C7D2FE !important;
             color: #3730A3 !important; border-radius: 10px !important; }
.stWarning { background: #FFFBEB !important; border: 1px solid #FDE68A !important;
             color: #92400E !important; border-radius: 10px !important; }

/* ── КАСТОМНЫЕ КОМПОНЕНТЫ — точно из landing ── */

/* Info box — синяя плашка */
.ab-info-box {
    background: #EEF2FF;
    border-left: 3px solid #4F46E5;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 13px !important;
    color: #3730A3 !important;
    margin: 10px 0 18px !important;
    line-height: 1.55 !important;
}

/* Section label — надпись над блоком */
.ab-section-label {
    font-size: 10px !important;
    font-weight: 600 !important;
    color: #9CA3AF !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    margin-bottom: 12px !important;
    display: block;
}

/* Hypothesis card */
.ab-hypo-card {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
    transition: box-shadow .15s;
}
.ab-hypo-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.08); }
.ab-hypo-title { font-size: 14px !important; font-weight: 600 !important;
                 color: #0F0F10 !important; margin-bottom: 6px !important; }
.ab-hypo-change { font-size: 13px !important; color: #4B5563 !important;
                  line-height: 1.55 !important; margin-bottom: 3px !important; }
.ab-hypo-expected { font-size: 13px !important; color: #4F46E5 !important; margin-bottom: 3px !important; }
.ab-hypo-meta { font-size: 12px !important; color: #9CA3AF !important;
                margin-top: 4px !important; font-style: italic; }
.ab-hypo-footer { display: flex; align-items: center; gap: 8px; margin-top: 8px; }

/* Badges */
.ab-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px !important;
    font-weight: 500 !important;
    line-height: 1.5;
}
.ab-badge-high { background: #EEF2FF; color: #4F46E5 !important; }
.ab-badge-mid  { background: #FFF7ED; color: #C2410C !important; }
.ab-badge-low  { background: #F4F4F5; color: #71717A !important; }
.ab-badge-conf { background: #F9FAFB; color: #6B7280 !important; border: 1px solid #E5E7EB; }

/* LLM summary */
.ab-llm-summary {
    background: #F9FAFB;
    border-left: 3px solid #4F46E5;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: #4B5563 !important;
    margin-top: 10px;
}

/* ── РЕЗУЛЬТАТЫ: stat row ── */
.ab-stat-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin-bottom: 12px;
}
.ab-stat-card {
    flex: 1; min-width: 120px;
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 13px 15px;
}
.ab-stat-label { font-size: 11px !important; color: #9CA3AF !important; font-weight: 500 !important; margin-bottom: 4px !important; }
.ab-stat-value { font-size: 22px !important; font-weight: 700 !important; color: #0F0F10 !important;
                 letter-spacing: -0.03em !important; line-height: 1.15 !important; }
.ab-stat-delta { font-size: 11px !important; margin-top: 2px !important; }
.ab-stat-green { color: #059669 !important; }
.ab-stat-red   { color: #DC2626 !important; }
.ab-stat-indigo{ color: #4F46E5 !important; }

/* Revenue highlight */
.ab-revenue-card {
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 12px;
    padding: 13px 18px;
    margin: 10px 0;
    display: flex; align-items: baseline; gap: 8px;
}
.ab-revenue-label { font-size: 12px !important; color: #4338CA !important; font-weight: 500 !important; }
.ab-revenue-value { font-size: 26px !important; font-weight: 700 !important; color: #3730A3 !important;
                    letter-spacing: -0.03em !important; }

/* ── PLOT ── */
.stImage img { border-radius: 10px !important; border: 1px solid #E5E7EB !important; }

/* ── РАЗДЕЛИТЕЛИ ── */
hr, .stDivider { border-color: #E5E7EB !important; }

/* ── КОЛЛАПС САЙДБАРА ── */
[data-testid="collapsedControl"] {
    background: #F9FAFB !important;
    border-right: 1px solid #E5E7EB !important;
}

/* ── PLACEHOLDER TEXT ── */
.ab-placeholder {
    text-align: center;
    padding: 52px 20px;
    color: #9CA3AF;
    font-size: 13px !important;
    border: 1.5px dashed #E5E7EB;
    border-radius: 12px;
}

/* ── ONBOARDING TOUR ── */
.ab-tour-overlay {
    position: fixed !important;
    inset: 0 !important;
    background: rgba(15,15,16,.55) !important;
    z-index: 9990 !important;
    pointer-events: none !important;
}
.ab-tour-tooltip {
    position: fixed !important;
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 22px 24px 18px !important;
    width: 320px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,.2), 0 2px 8px rgba(0,0,0,.08) !important;
    z-index: 9999 !important;
    font-size: 14px !important;
    color: #0F0F10 !important;
    pointer-events: all !important;
    animation: ab-tour-in .25s cubic-bezier(.4,0,.2,1) !important;
}
@keyframes ab-tour-in {
    from { opacity: 0; transform: translateY(6px) scale(.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.ab-tour-step-label {
    font-size: 10px !important; font-weight: 600 !important;
    color: #4F46E5 !important; text-transform: uppercase !important;
    letter-spacing: .08em !important; margin-bottom: 8px !important; display: block;
}
.ab-tour-title {
    font-size: 15px !important; font-weight: 600 !important;
    color: #0F0F10 !important; letter-spacing: -0.015em !important;
    margin-bottom: 8px !important; line-height: 1.35 !important;
}
.ab-tour-desc {
    font-size: 13px !important; color: #4B5563 !important;
    line-height: 1.6 !important; margin-bottom: 16px !important;
}
.ab-tour-progress {
    display: flex; gap: 4px; margin-bottom: 16px;
}
.ab-tour-pip {
    height: 3px; flex: 1; border-radius: 99px; background: #E5E7EB;
}
.ab-tour-pip-done { background: #4F46E5 !important; }
.ab-tour-actions {
    display: flex; align-items: center; gap: 8px;
}
.ab-tour-btn-primary {
    background: #0F0F10 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 8px 16px !important; font-size: 13px !important;
    font-weight: 500 !important; cursor: pointer !important;
    transition: background .15s !important; flex: 1 !important;
    font-family: inherit !important;
}
.ab-tour-btn-primary:hover { background: #4F46E5 !important; }
.ab-tour-btn-back {
    background: transparent !important; color: #4B5563 !important;
    border: 1px solid #E5E7EB !important; border-radius: 8px !important;
    padding: 8px 13px !important; font-size: 13px !important;
    cursor: pointer !important; white-space: nowrap !important;
    font-family: inherit !important;
}
.ab-tour-btn-back:hover { background: #F9FAFB !important; color: #0F0F10 !important; }
.ab-tour-btn-skip {
    background: none !important; border: none !important;
    color: #9CA3AF !important; font-size: 12px !important;
    cursor: pointer !important; margin-left: auto !important;
    padding: 4px !important; font-family: inherit !important;
}
.ab-tour-btn-skip:hover { color: #4B5563 !important; }

/* Welcome modal */
.ab-welcome-backdrop {
    position: fixed !important; inset: 0 !important;
    background: rgba(15,15,16,.55) !important;
    z-index: 9995 !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
}
.ab-welcome-modal {
    background: #ffffff !important;
    border-radius: 18px !important;
    padding: 36px 36px 30px !important;
    width: 440px !important;
    max-width: 90vw !important;
    box-shadow: 0 24px 64px rgba(0,0,0,.22) !important;
    text-align: center !important;
    animation: ab-tour-in .3s cubic-bezier(.4,0,.2,1) !important;
}
.ab-welcome-logo {
    width: 56px; height: 56px;
    background: #0F0F10;
    border-radius: 15px;
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; font-weight: 700; color: white;
    letter-spacing: -0.03em;
    margin: 0 auto 20px;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# ONBOARDING: HTML/JS система
# ────────────────────────────────────────────────────────────────────────────

ONBOARDING_JS = """
<script>
(function() {
  'use strict';

  /* ─── КОНФИГУРАЦИЯ ТУРА ─── */
  const STEPS = [
    {
      title: "Добро пожаловать в AB·AI",
      desc:  "Это полный цикл автоматизации A/B-теста: от генерации гипотез до бизнес-отчёта. Тур займёт ~1 минуту.",
      anchor: null,   /* центральный tooltip */
      placement: "center"
    },
    {
      title: "Навигация по модулям",
      desc:  "Четыре раздела образуют pipeline: Гипотезы → Планирование → Симуляция → Отчёт. Переходи между ними через левое меню.",
      anchor: "[data-testid='stSidebar']",
      placement: "right"
    },
    {
      title: "Генерация гипотез с ИИ",
      desc:  "В разделе «Гипотезы» укажи продукт и проблемную зону — LLM предложит конкретные тестируемые изменения с оценкой уверенности.",
      anchor: "[data-testid='stSidebar']",
      placement: "right"
    },
    {
      title: "Расчёт выборки",
      desc:  "«Планирование» авторасчитает необходимый объём данных по заданным α, β и MDE — без формул вручную.",
      anchor: "[data-testid='stSidebar']",
      placement: "right"
    },
    {
      title: "Thompson Sampling",
      desc:  "«Симуляция» сравнивает адаптивный алгоритм Thompson Sampling с классическим A/B 50/50 и показывает снижение regret.",
      anchor: "[data-testid='stSidebar']",
      placement: "right"
    },
    {
      title: "Анализ и LLM-отчёт",
      desc:  "«Отчёт» считает p-value, Cohen's d, 95% ДИ и генерирует текстовое бизнес-резюме через языковую модель.",
      anchor: "[data-testid='stSidebar']",
      placement: "right"
    }
  ];

  const LS_KEY = "abai_onboarding_done_v2";
  let currentStep = 0;
  let overlay, tooltip, highlight;

  /* ─── ПРОВЕРКА: уже видел? ─── */
  function isDone() {
    try { return !!localStorage.getItem(LS_KEY); } catch(e) { return false; }
  }
  function markDone() {
    try { localStorage.setItem(LS_KEY, "1"); } catch(e) {}
  }

  /* ─── OVERLAY ─── */
  function createOverlay() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.id = "ab-tour-overlay";
    overlay.style.cssText = [
      "position:fixed","inset:0","background:rgba(15,15,16,.5)",
      "z-index:9990","pointer-events:none","transition:opacity .25s"
    ].join(";");
    document.body.appendChild(overlay);
  }

  /* ─── HIGHLIGHT ─── */
  function createHighlight() {
    if (highlight) return;
    highlight = document.createElement("div");
    highlight.id = "ab-tour-hl";
    highlight.style.cssText = [
      "position:fixed","border-radius:10px",
      "box-shadow:0 0 0 9999px rgba(15,15,16,.5)",
      "z-index:9991","pointer-events:none",
      "transition:all .3s cubic-bezier(.4,0,.2,1)"
    ].join(";");
    document.body.appendChild(highlight);
  }

  function positionHighlight(anchor) {
    if (!highlight) return;
    if (!anchor) {
      highlight.style.display = "none";
      return;
    }
    const r = anchor.getBoundingClientRect();
    const P = 6;
    highlight.style.display = "block";
    highlight.style.top    = (r.top  - P) + "px";
    highlight.style.left   = (r.left - P) + "px";
    highlight.style.width  = (r.width  + P*2) + "px";
    highlight.style.height = (r.height + P*2) + "px";
  }

  /* ─── TOOLTIP ─── */
  function buildTooltip() {
    tooltip = document.createElement("div");
    tooltip.id = "ab-tour-tip";
    tooltip.style.cssText = [
      "position:fixed","background:#fff",
      "border-radius:14px","padding:22px 24px 18px",
      "width:310px","box-shadow:0 12px 40px rgba(0,0,0,.2),0 2px 8px rgba(0,0,0,.08)",
      "z-index:9999","font-family:Inter,system-ui,sans-serif","pointer-events:all",
      "animation:ab-tt-in .25s cubic-bezier(.4,0,.2,1)"
    ].join(";");

    const style = document.createElement("style");
    style.textContent = "@keyframes ab-tt-in{from{opacity:0;transform:translateY(6px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}";
    document.head.appendChild(style);
    document.body.appendChild(tooltip);
  }

  function renderTooltip(step) {
    const s = STEPS[step];
    const total = STEPS.length;

    /* pip HTML */
    let pips = "";
    for (let i = 0; i < total; i++) {
      const done = i <= step;
      pips += `<div style="height:3px;flex:1;border-radius:99px;background:${done?"#4F46E5":"#E5E7EB"};transition:background .3s"></div>`;
    }

    const isFirst = step === 0;
    const isLast  = step === total - 1;

    tooltip.innerHTML = `
      <span style="font-size:10px;font-weight:600;color:#4F46E5;text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:8px">
        Шаг ${step+1} из ${total}
      </span>
      <div style="font-size:15px;font-weight:600;color:#0F0F10;letter-spacing:-.015em;margin-bottom:8px;line-height:1.35">${s.title}</div>
      <div style="font-size:13px;color:#4B5563;line-height:1.6;margin-bottom:14px">${s.desc}</div>
      <div style="display:flex;gap:4px;margin-bottom:16px">${pips}</div>
      <div style="display:flex;align-items:center;gap:8px">
        ${!isFirst ? `<button id="ab-back" style="background:transparent;color:#4B5563;border:1px solid #E5E7EB;border-radius:8px;padding:8px 13px;font-size:13px;cursor:pointer;font-family:inherit;white-space:nowrap">← Назад</button>` : ""}
        <button id="ab-next" style="background:#0F0F10;color:#fff;border:none;border-radius:8px;padding:8px 16px;font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;flex:1;transition:background .15s">
          ${isLast ? "Готово ✓" : "Далее →"}
        </button>
        ${!isLast ? `<button id="ab-skip" style="background:none;border:none;color:#9CA3AF;font-size:12px;cursor:pointer;font-family:inherit;padding:4px;margin-left:auto">Пропустить</button>` : ""}
      </div>`;

    tooltip.querySelector("#ab-next").onmouseenter = e => e.target.style.background = "#4F46E5";
    tooltip.querySelector("#ab-next").onmouseleave = e => e.target.style.background = "#0F0F10";
    tooltip.querySelector("#ab-next").onclick = () => {
      if (isLast) { endTour(true); } else { goStep(step + 1); }
    };
    const back = tooltip.querySelector("#ab-back");
    if (back) back.onclick = () => goStep(step - 1);
    const skip = tooltip.querySelector("#ab-skip");
    if (skip) skip.onclick = () => endTour(false);
  }

  function positionTooltip(anchorEl, placement) {
    if (!tooltip) return;
    const TW = 310, TH = 240, PAD = 12;
    const VW = window.innerWidth, VH = window.innerHeight;

    if (!anchorEl || placement === "center") {
      tooltip.style.top  = Math.round(VH/2 - TH/2) + "px";
      tooltip.style.left = Math.round(VW/2 - TW/2) + "px";
      return;
    }

    const r = anchorEl.getBoundingClientRect();
    let top, left;

    if (placement === "right") {
      top  = r.top;
      left = r.right + PAD;
    } else if (placement === "left") {
      top  = r.top;
      left = r.left - TW - PAD;
    } else if (placement === "bottom") {
      top  = r.bottom + PAD;
      left = r.left;
    } else {
      top  = r.top - TH - PAD;
      left = r.left;
    }

    top  = Math.max(8, Math.min(top,  VH - TH - 8));
    left = Math.max(8, Math.min(left, VW - TW - 8));

    tooltip.style.top  = top  + "px";
    tooltip.style.left = left + "px";
  }

  /* ─── STEP LOGIC ─── */
  function goStep(idx) {
    currentStep = idx;
    const s    = STEPS[idx];
    const anch = s.anchor ? document.querySelector(s.anchor) : null;

    /* перерисовываем tooltip */
    if (tooltip) tooltip.remove();
    buildTooltip();
    renderTooltip(idx);
    positionTooltip(anch, s.placement);
    positionHighlight(anch);
  }

  /* ─── НАЧАЛО И КОНЕЦ ─── */
  function startTour() {
    createOverlay();
    createHighlight();
    buildTooltip();
    goStep(0);
  }

  function endTour(completed) {
    if (overlay)    { overlay.remove();    overlay    = null; }
    if (highlight)  { highlight.remove();  highlight  = null; }
    if (tooltip)    { tooltip.remove();    tooltip    = null; }
    if (completed) markDone();

    if (completed) {
      showDoneBanner();
    }
  }

  /* ─── ФИНАЛЬНЫЙ БАННЕР ─── */
  function showDoneBanner() {
    const banner = document.createElement("div");
    banner.style.cssText = [
      "position:fixed","bottom:24px","right:24px",
      "background:#0F0F10","color:#fff",
      "border-radius:12px","padding:14px 20px",
      "font-family:Inter,system-ui,sans-serif",
      "font-size:14px","z-index:9998",
      "box-shadow:0 8px 24px rgba(0,0,0,.25)",
      "display:flex","align-items:center","gap:12px",
      "animation:ab-tt-in .3s cubic-bezier(.4,0,.2,1)"
    ].join(";");
    banner.innerHTML = `
      <span style="font-size:18px">✓</span>
      <span>Тур завершён! Теперь вы знаете AB·AI.</span>
      <button onclick="this.parentNode.remove()" style="background:#ffffff22;border:none;color:#fff;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px;font-family:inherit">×</button>`;
    document.body.appendChild(banner);
    setTimeout(() => banner.remove(), 5000);
  }

  /* ─── WELCOME SCREEN ─── */
  function showWelcome() {
    const modal = document.createElement("div");
    modal.style.cssText = [
      "position:fixed","inset:0","background:rgba(15,15,16,.55)",
      "z-index:9995","display:flex","align-items:center","justify-content:center"
    ].join(";");
    modal.innerHTML = `
      <div style="background:#fff;border-radius:18px;padding:36px 36px 28px;width:420px;
                  max-width:90vw;box-shadow:0 24px 64px rgba(0,0,0,.22);text-align:center;
                  font-family:Inter,system-ui,sans-serif;
                  animation:ab-tt-in .3s cubic-bezier(.4,0,.2,1)">
        <div style="width:52px;height:52px;background:#0F0F10;border-radius:14px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:18px;font-weight:700;color:#fff;letter-spacing:-.03em;
                    margin:0 auto 20px">AB</div>
        <div style="font-size:20px;font-weight:600;color:#0F0F10;letter-spacing:-.02em;margin-bottom:10px">
          Добро пожаловать в AB·AI
        </div>
        <div style="font-size:14px;color:#4B5563;line-height:1.65;margin-bottom:22px">
          Инструмент автоматизации A/B-тестирования с ИИ.<br>
          Хотите пройти быстрый тур по функциям?
        </div>
        <div style="text-align:left;margin-bottom:22px;display:flex;flex-direction:column;gap:9px">
          ${["LLM-генерация тестируемых гипотез",
             "Авторасчёт выборки (размер, мощность, длительность)",
             "Thompson Sampling — адаптивное распределение трафика",
             "Автоматический бизнес-отчёт через языковую модель"]
            .map(f => `<div style="display:flex;align-items:center;gap:10px;font-size:13px;color:#4B5563">
              <div style="width:6px;height:6px;border-radius:50%;background:#4F46E5;flex-shrink:0"></div>${f}
            </div>`).join("")}
        </div>
        <button id="ab-welcome-start" style="width:100%;background:#0F0F10;color:#fff;border:none;
                border-radius:10px;padding:12px 20px;font-size:14px;font-weight:500;cursor:pointer;
                margin-bottom:10px;font-family:inherit;
                box-shadow:0 1px 2px rgba(0,0,0,.08),0 4px 16px rgba(0,0,0,.12);
                transition:background .15s">
          Начать тур →
        </button>
        <div>
          <button id="ab-welcome-skip" style="background:none;border:none;color:#9CA3AF;
                  font-size:13px;cursor:pointer;font-family:inherit">
            Пропустить, я разберусь сам
          </button>
        </div>
      </div>`;

    document.body.appendChild(modal);
    modal.querySelector("#ab-welcome-start").onclick = () => {
      modal.remove();
      startTour();
    };
    modal.querySelector("#ab-welcome-start").onmouseenter = e => e.target.style.background = "#4F46E5";
    modal.querySelector("#ab-welcome-start").onmouseleave = e => e.target.style.background = "#0F0F10";
    modal.querySelector("#ab-welcome-skip").onclick = () => {
      modal.remove();
      markDone();
    };
  }

  /* ─── ИНИЦИАЛИЗАЦИЯ ─── */
  function init() {
    if (!isDone()) {
      showWelcome();
    }

    /* Кнопка «Показать тур снова» — ищем по тексту в сайдбаре */
    const observer = new MutationObserver(() => {
      const btns = document.querySelectorAll("button");
      btns.forEach(btn => {
        if (btn.textContent.includes("Показать тур") && !btn._abBound) {
          btn._abBound = true;
          btn.onclick = e => { e.stopPropagation(); startTour(); };
        }
      });
    });
    observer.observe(document.body, { childList: true, subtree: true });

    /* Кнопка из st.button с ключом ab_restart_tour */
    window._abStartTour = startTour;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    setTimeout(init, 800);
  }
})();
</script>
"""

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AB·AI")
    st.markdown("*Автоматизация A/B-тестов*")
    st.divider()

    page = st.radio(
        "// навигация",
        options=["Гипотезы", "Планирование", "Симуляция", "Отчёт", "Настройки"],
        label_visibility="visible",
    )

    st.divider()

    # Кнопка перезапуска тура
    if st.button("◎ Показать тур снова", key="ab_restart_tour", use_container_width=True):
        pass  # JS перехватит клик через MutationObserver

    st.markdown(
        "<div style='font-size:11px; color:#9CA3AF; line-height:1.6; margin-top:4px'>"
        "Курсовая работа · 2025<br>"
        "Автоматизация A/B-тестирования с ИИ</div>",
        unsafe_allow_html=True,
    )

# Инжектируем JS онбординга один раз
st.markdown(ONBOARDING_JS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 1: ГЕНЕРАЦИЯ ГИПОТЕЗ
# ────────────────────────────────────────────────────────────────────────────

if page == "Гипотезы":
    st.markdown("### Генерация гипотез")
    st.markdown(
        "<div class='ab-info-box'>LLM анализирует контекст продукта и историю тестов — "
        "формулирует конкретные, тестируемые гипотезы с оценкой уверенности.</div>",
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<span class='ab-section-label'>Контекст продукта</span>", unsafe_allow_html=True)
        product_name  = st.text_input("Продукт / экран", value="Ozon Express — корзина",
                                       placeholder="Название продукта или экрана")
        target_metric = st.selectbox(
            "Целевая метрика",
            ["Конверсия в заказ", "CTR кнопки", "Средний чек", "Время до оплаты", "Другое"],
        )
        baseline_cr  = st.slider("Базовая конверсия (%)", 0.5, 30.0, 3.2, 0.1) / 100
        problem_area = st.text_input("Проблемная зона", value="Кнопка оформления заказа")
        audience     = st.text_input("Целевая аудитория (необязательно)",
                                      value="Мобильные пользователи 25–45 лет")
        hypo_count   = st.slider("Количество гипотез", 2, 7, 3)

        st.markdown("<span class='ab-section-label'>История тестов</span>", unsafe_allow_html=True)
        history_raw = st.text_area(
            "Каждый тест с новой строки: название — результат",
            value="Изменение цвета кнопки — не победила\nДобавление таймера — победила, +8%",
            height=90,
            label_visibility="collapsed",
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
                st.markdown(f"<span class='ab-section-label'>Сгенерировано: {len(hypotheses)} гипотез</span>",
                            unsafe_allow_html=True)
                for i, h in enumerate(hypotheses, 1):
                    priority    = h.get("priority", "средний")
                    badge_class = ("ab-badge-high" if priority == "высокий"
                                   else "ab-badge-mid" if priority == "средний"
                                   else "ab-badge-low")
                    conf = h.get("confidence", 0.5)
                    st.markdown(f"""
<div class="ab-hypo-card">
  <div class="ab-hypo-title">#{i} &nbsp; {h.get('title', '—')}</div>
  <div class="ab-hypo-change">→ {h.get('change', '')}</div>
  <div class="ab-hypo-expected">{h.get('expected', '')}</div>
  <div class="ab-hypo-meta">{h.get('reason', '')}</div>
  <div class="ab-hypo-footer">
    <span class="ab-badge {badge_class}">{priority}</span>
    <span class="ab-badge ab-badge-conf">Уверенность: {conf:.0%}</span>
  </div>
</div>""", unsafe_allow_html=True)

        elif "last_hypotheses" in st.session_state:
            st.markdown("<span class='ab-section-label'>Последний результат</span>", unsafe_allow_html=True)
            for i, h in enumerate(st.session_state["last_hypotheses"], 1):
                conf = h.get("confidence", 0.5)
                st.markdown(f"""
<div class="ab-hypo-card">
  <div class="ab-hypo-title">#{i} &nbsp; {h.get('title', '—')}</div>
  <div class="ab-hypo-change">→ {h.get('change', '')}</div>
  <div class="ab-hypo-meta">{h.get('reason', '')}</div>
  <div class="ab-hypo-footer">
    <span class="ab-badge ab-badge-conf">Уверенность: {conf:.0%}</span>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='ab-placeholder'>◆<br><br>Заполни форму слева и нажми «Сгенерировать»</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 2: ПЛАНИРОВАНИЕ ТЕСТА
# ────────────────────────────────────────────────────────────────────────────

elif page == "Планирование":
    st.markdown("### Планирование эксперимента")
    st.markdown(
        "<div class='ab-info-box'>Авторасчёт минимально необходимой выборки по формуле нормального "
        "приближения к биномиальному распределению. Все параметры задаются вручную — "
        "результат пересчитывается мгновенно.</div>",
        unsafe_allow_html=True,
    )

    col_params, col_plan = st.columns([1, 1], gap="large")

    with col_params:
        st.markdown("<span class='ab-section-label'>Параметры теста</span>", unsafe_allow_html=True)
        p_baseline   = st.slider("Базовая конверсия (%)", 0.5, 30.0, 3.2, 0.1) / 100
        mde_pct      = st.slider("MDE — минимальный обнаруживаемый эффект (абс., п.п.)", 0.1, 5.0, 0.8, 0.1)
        mde          = mde_pct / 100
        alpha        = st.select_slider("Уровень значимости α", options=[0.01, 0.05, 0.10], value=0.05)
        power        = st.select_slider("Мощность теста 1−β", options=[0.70, 0.80, 0.90], value=0.80)
        daily_traffic = st.number_input("Суточный трафик (сессий/день)", min_value=100, value=1000, step=100)

    with col_plan:
        plan = calculate_sample_size(
            baseline_conversion=p_baseline,
            minimum_detectable_effect=mde,
            significance_level=alpha,
            statistical_power=power,
            daily_traffic=int(daily_traffic),
        )
        st.session_state["last_plan"] = plan

        st.markdown("<span class='ab-section-label'>Результат расчёта</span>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Выборка / группа", f"{plan.required_observations_per_group:,}")
        m2.metric("Всего наблюдений", f"{plan.total_required_observations:,}")
        m3.metric(
            "Длительность теста",
            f"{plan.estimated_duration_days} дн." if plan.estimated_duration_days else "—",
        )
        m4.metric("Целевая конверсия B", f"{plan.target_conversion:.2%}")

        # Power analysis plot
        st.markdown("<span class='ab-section-label'>Power analysis: выборка vs MDE</span>",
                    unsafe_allow_html=True)
        mde_range = np.linspace(0.002, 0.05, 60)
        n_range   = []
        for m in mde_range:
            try:
                p = calculate_sample_size(p_baseline, m, alpha, power)
                n_range.append(p.required_observations_per_group)
            except Exception:
                n_range.append(np.nan)

        fig, ax = plt.subplots(figsize=(5.5, 3), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E5E7EB")
        ax.spines["bottom"].set_color("#E5E7EB")
        ax.tick_params(colors="#9CA3AF", labelsize=10)
        ax.grid(True, alpha=0.35, color="#E5E7EB")

        ax.plot([m * 100 for m in mde_range], n_range,
                color="#4F46E5", linewidth=2.5)
        ax.fill_between([m * 100 for m in mde_range], n_range,
                        alpha=0.08, color="#4F46E5")
        ax.axvline(mde * 100, color="#DC2626", linestyle="--", linewidth=1.5,
                   label=f"MDE = {mde_pct:.1f} п.п.")
        ax.set_xlabel("MDE (абс., п.п.)", fontsize=11, color="#9CA3AF")
        ax.set_ylabel("Наблюдений / группа", fontsize=11, color="#9CA3AF")
        ax.legend(fontsize=10, framealpha=0)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 3: СИМУЛЯЦИЯ THOMPSON SAMPLING
# ────────────────────────────────────────────────────────────────────────────

elif page == "Симуляция":
    st.markdown("### Адаптивное тестирование")
    st.markdown(
        "<div class='ab-info-box'>Симуляция Multi-Armed Bandit с Thompson Sampling: "
        "каждый вариант моделируется Beta(α, β)-распределением — трафик автоматически "
        "перераспределяется в пользу лидирующего варианта.</div>",
        unsafe_allow_html=True,
    )

    col_cfg, col_vis = st.columns([1, 1.5], gap="large")

    with col_cfg:
        st.markdown("<span class='ab-section-label'>Параметры симуляции</span>", unsafe_allow_html=True)
        cr_a       = st.slider("Истинная конверсия варианта A (%)", 1.0, 20.0, 3.2, 0.1) / 100
        cr_b       = st.slider("Истинная конверсия варианта B (%)", 1.0, 20.0, 4.0, 0.1) / 100
        n_visitors = st.select_slider(
            "Число посетителей",
            options=[1_000, 5_000, 10_000, 25_000, 50_000],
            value=10_000,
        )
        sim_seed = st.number_input("Seed (воспроизводимость)", min_value=0, value=42, step=1)
        sim_btn  = st.button("▶ Запустить симуляцию", use_container_width=True)

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
                st.session_state["sim_result"]  = result
                st.session_state["sim_run_key"] = run_key

            result = st.session_state.get("sim_result")
            if result:
                rr = (result["total_regret_classic"] - result["total_regret_ts"]) / max(result["total_regret_classic"], 1e-9) * 100
                winner_cr = [cr_a, cr_b][result["winner_idx"]]
                m1, m2, m3 = st.columns(3)
                m1.metric("Снижение regret", f"{rr:.1f}%", "vs классический A/B")
                m2.metric("Победитель", result["winner_label"])
                m3.metric("Posterior mean победителя", f"{winner_cr:.2%}")

                steps     = result["steps"]
                ts_r      = result["ts_regret"]
                cl_r      = result["classic_regret"]
                traffic   = result["traffic_shares"]
                traffic_b = [t[1] for t in traffic]

                def _style_ax(a):
                    a.set_facecolor("#ffffff")
                    a.spines["top"].set_visible(False)
                    a.spines["right"].set_visible(False)
                    a.spines["left"].set_color("#E5E7EB")
                    a.spines["bottom"].set_color("#E5E7EB")
                    a.tick_params(colors="#9CA3AF", labelsize=10)
                    a.grid(True, alpha=0.3, color="#E5E7EB")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), facecolor="#ffffff")
                _style_ax(ax1); _style_ax(ax2)

                ax1.plot(steps, ts_r, color="#4F46E5", linewidth=2.5, label="Thompson Sampling")
                ax1.plot(steps, cl_r, color="#EF4444", linewidth=1.8, linestyle="--", label="Классический A/B")
                ax1.fill_between(steps, ts_r, cl_r, alpha=0.07, color="#4F46E5")
                ax1.set_title("Cumulative regret", fontsize=12, color="#0F0F10", fontweight="600", pad=8)
                ax1.set_xlabel("Посетители", color="#9CA3AF", fontsize=10)
                ax1.set_ylabel("Regret", color="#9CA3AF", fontsize=10)
                ax1.legend(fontsize=9, framealpha=0)

                ax2.plot(steps, [t * 100 for t in traffic_b],       color="#4F46E5", linewidth=2.5, label="Вариант B")
                ax2.plot(steps, [(1-t) * 100 for t in traffic_b],   color="#059669", linewidth=2.5, label="Вариант A")
                ax2.axhline(50, color="#E5E7EB", linestyle="--", linewidth=1.2)
                ax2.set_title("Доля трафика (%)", fontsize=12, color="#0F0F10", fontweight="600", pad=8)
                ax2.set_xlabel("Посетители", color="#9CA3AF", fontsize=10)
                ax2.set_ylabel("%", color="#9CA3AF", fontsize=10)
                ax2.legend(fontsize=9, framealpha=0)

                fig.tight_layout(pad=1.5)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("<span class='ab-section-label'>Posterior Beta-распределения (финал)</span>",
                            unsafe_allow_html=True)
                fig2, ax3 = plt.subplots(figsize=(9, 2.8), facecolor="#ffffff")
                _style_ax(ax3)
                x = np.linspace(0, 0.12, 500)
                for arm_idx, arm in enumerate(result["final_arms"]):
                    color  = ["#059669", "#4F46E5"][arm_idx]
                    label_ = f"{'Вариант A' if arm_idx==0 else 'Вариант B'} (mean={arm.posterior_mean:.3%})"
                    y = stats.beta.pdf(x, arm.success_count, arm.failure_count)
                    ax3.plot(x, y, color=color, linewidth=2.5, label=label_)
                    ax3.fill_between(x, y, alpha=0.10, color=color)
                ax3.set_xlabel("Конверсия θ", color="#9CA3AF", fontsize=10)
                ax3.set_ylabel("Плотность", color="#9CA3AF", fontsize=10)
                ax3.legend(fontsize=10, framealpha=0)
                fig2.tight_layout(pad=1.5)
                st.pyplot(fig2)
                plt.close(fig2)
        else:
            st.markdown(
                "<div class='ab-placeholder'>◆<br><br>Задай параметры и нажми «Запустить симуляцию»</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 4: АНАЛИЗ РЕЗУЛЬТАТОВ / ОТЧЁТ
# ────────────────────────────────────────────────────────────────────────────

elif page == "Отчёт":
    st.markdown("### Анализ результатов")
    st.markdown(
        "<div class='ab-info-box'>Введи сырые результаты теста — система посчитает статистику "
        "и сгенерирует бизнес-резюме через LLM.</div>",
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("<span class='ab-section-label'>Результаты теста</span>", unsafe_allow_html=True)
        ctrl_visitors      = st.number_input("Контроль — посетители",  min_value=10, value=3000)
        ctrl_conv          = st.number_input("Контроль — конверсии",   min_value=0,  value=150)
        treat_visitors     = st.number_input("Вариант B — посетители", min_value=10, value=3000)
        treat_conv         = st.number_input("Вариант B — конверсии",  min_value=0,  value=185)
        metric_name        = st.text_input("Название метрики",          value="конверсия в заказ")
        daily_rev_per_conv = st.number_input(
            "Средняя выручка с конверсии (₽)", min_value=0, value=2500, step=100,
            help="Используется для расчёта денежного эффекта",
        )
        report_btn = st.button("Рассчитать и сгенерировать отчёт", use_container_width=True)

    with col_out:
        if report_btn:
            # ── Статистика ──
            p_ctrl   = ctrl_conv  / ctrl_visitors
            p_treat  = treat_conv / treat_visitors
            lift_abs = p_treat - p_ctrl
            lift_rel = lift_abs / p_ctrl if p_ctrl > 0 else 0

            pooled = (ctrl_conv + treat_conv) / (ctrl_visitors + treat_visitors)
            se     = math.sqrt(pooled * (1 - pooled) * (1 / ctrl_visitors + 1 / treat_visitors))
            z_score = lift_abs / se if se > 0 else 0
            p_value  = 2 * (1 - stats.norm.cdf(abs(z_score)))
            cohens_d = lift_abs / math.sqrt(pooled * (1 - pooled)) if pooled > 0 else 0

            se_diff = math.sqrt(
                p_ctrl  * (1 - p_ctrl)  / ctrl_visitors
                + p_treat * (1 - p_treat) / treat_visitors
            )
            ci_lo = lift_abs - 1.96 * se_diff
            ci_hi = lift_abs + 1.96 * se_diff

            extra_conversions = lift_abs * ctrl_visitors
            monthly_revenue   = extra_conversions * daily_rev_per_conv

            sig_label = "✅ значимо" if p_value < 0.05 else "❌ не значимо"
            sig_color = "#059669"    if p_value < 0.05 else "#DC2626"

            # ── Карточки метрик ──
            st.markdown("<span class='ab-section-label'>Результаты</span>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("p-value",   f"{p_value:.4f}", sig_label)
            m2.metric("Lift",      f"{lift_rel:+.1%}")
            m3.metric("Cohen's d", f"{cohens_d:.3f}")
            m4.metric("95% CI",    f"[{ci_lo:+.2%}, {ci_hi:+.2%}]")

            # ── Выручка ──
            st.markdown(
                f"<div class='ab-revenue-card'>"
                f"<span class='ab-revenue-label'>Доп. выручка / мес. (оценка)</span>"
                f"<span class='ab-revenue-value'>{monthly_revenue:,.0f} ₽</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── LLM-резюме ──
            stats_for_llm = {
                "метрика":             metric_name,
                "конверсия_контроль":  f"{p_ctrl:.3%}",
                "конверсия_B":         f"{p_treat:.3%}",
                "lift_абсолютный":     f"{lift_abs:+.3%}",
                "lift_относительный":  f"{lift_rel:+.1%}",
                "p_value":             round(p_value, 4),
                "cohens_d":            round(cohens_d, 3),
                "CI_95":               f"[{ci_lo:+.3%}, {ci_hi:+.3%}]",
                "значимость":          "да" if p_value < 0.05 else "нет",
                "доп_выручка_в_месяц_руб": int(monthly_revenue),
            }

            with st.spinner("LLM генерирует бизнес-резюме..."):
                summary = call_llm(build_report_prompt(stats_for_llm))

            st.markdown("<span class='ab-section-label'>LLM-резюме</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='ab-llm-summary'>{summary}</div>", unsafe_allow_html=True)

            st.session_state["last_report"] = {"stats": stats_for_llm, "summary": summary}

        elif "last_report" in st.session_state:
            r = st.session_state["last_report"]
            st.markdown("<span class='ab-section-label'>Последний отчёт</span>", unsafe_allow_html=True)
            st.json(r["stats"])
            st.markdown(f"<div class='ab-llm-summary'>{r['summary']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='ab-placeholder'>◆<br><br>Введи данные теста и нажми «Рассчитать»</div>",
                unsafe_allow_html=True,
            )


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 5: НАСТРОЙКИ
# ────────────────────────────────────────────────────────────────────────────

elif page == "Настройки":
    st.markdown("### Настройки")
    st.markdown(
        "<div class='ab-info-box'>Укажи API-ключи для LLM. Ключи хранятся только в памяти сессии "
        "и не передаются третьим сторонам. В продакшн-деплое используй <code>st.secrets</code>.</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("<span class='ab-section-label'>YandexGPT</span>", unsafe_allow_html=True)
        yandex_folder = st.text_input(
            "Folder ID (Yandex Cloud)",
            value=st.session_state.get("yandex_folder_id", ""),
            type="password",
            placeholder="b1g...",
        )
        yandex_key = st.text_input(
            "API Key (YandexGPT)",
            value=st.session_state.get("yandex_api_key", ""),
            type="password",
            placeholder="AQVN...",
        )
        if st.button("Сохранить ключи YandexGPT", use_container_width=True):
            st.session_state["yandex_folder_id"] = yandex_folder
            st.session_state["yandex_api_key"]   = yandex_key
            st.success("Ключи YandexGPT сохранены в сессии.")

    with col_r:
        st.markdown("<span class='ab-section-label'>Anthropic (Claude)</span>", unsafe_allow_html=True)
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=st.session_state.get("anthropic_api_key", ""),
            type="password",
            placeholder="sk-ant-...",
        )
        if st.button("Сохранить ключ Anthropic", use_container_width=True):
            st.session_state["anthropic_api_key"] = anthropic_key
            st.success("Ключ Anthropic сохранён в сессии.")

        st.markdown(
            "<span class='ab-section-label' style='margin-top:20px;display:block'>Приоритет</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:13px;color:#4B5563;line-height:1.65'>"
            "Если заданы оба ключа — используется <strong>YandexGPT</strong>.<br>"
            "Для переключения на Claude удали ключи Yandex.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("<span class='ab-section-label'>О приложении</span>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#4B5563;line-height:1.8'>"
        "Стек: Python 3.11 · scipy · numpy · matplotlib · Streamlit<br>"
        "LLM: YandexGPT / Anthropic Claude<br>"
        "Алгоритмы: Thompson Sampling, Z-тест для двух пропорций, Cohen's d<br>"
        "Курсовая работа · НИУ ВШЭ / ИИМУП · 2025"
        "</div>",
        unsafe_allow_html=True,
    )

"""
AB·AI — Streamlit MVP  (v3 — fixes)
Автоматизация этапов A/B-тестирования с использованием методов ИИ.

Запуск:
    pip install streamlit anthropic scipy numpy matplotlib pandas
    streamlit run app.py

Что исправлено в v3:
    • Онбординг работает через streamlit.components.html (window.parent),
      что позволяет JS манипулировать реальным DOM, а не iframe-sandbox
    • Страница «Настройки» убрана из навигации (API-ключи — в st.secrets)
    • Навигация: убрана метка "// навигация", точка выровнена с текстом
    • Selectbox: белый фон, светлое выпадающее меню
    • Кнопки: светлый hover-стиль, не чёрный фон у disabled-state
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
import streamlit.components.v1 as components

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
    def posterior_mean(self) -> float:
        return self.success_count / (self.success_count + self.failure_count)


def run_thompson_simulation(
    true_rates: list,
    arm_labels: list,
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


# ────────────────────────────────────────────────────────────────────────────
# БЛОК 3: LLM
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
    folder_id = st.secrets.get("YANDEX_FOLDER_ID", "") or st.session_state.get("yandex_folder_id", "")
    api_key   = st.secrets.get("YANDEX_API_KEY", "")   or st.session_state.get("yandex_api_key", "")
    if folder_id and api_key:
        return _call_yandex(folder_id, api_key, prompt, system)
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "") or st.session_state.get("anthropic_api_key", "")
    if anthropic_key:
        return _call_anthropic(anthropic_key, prompt, system)
    return "[API-ключ не указан. Добавь YANDEX_API_KEY + YANDEX_FOLDER_ID или ANTHROPIC_API_KEY в .streamlit/secrets.toml]"


# ────────────────────────────────────────────────────────────────────────────
# БЛОК 4: Промпты
# ────────────────────────────────────────────────────────────────────────────

def build_hypothesis_prompt(product_name, target_metric, baseline_cr, problem_area, audience, history, count):
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
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AB·AI — A/B-тестирование с ИИ",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# ────────────────────────────────────────────────────────────────────────────
# CSS
# ────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

#MainMenu, footer, header { visibility: hidden; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased; }
.block-container { padding-top: 1.75rem !important; padding-bottom: 3rem !important;
                   max-width: 1160px !important; }

.stApp, .stApp > div, .main { background: #ffffff !important; }

/* ── САЙДБАР ── */
[data-testid="stSidebar"] {
    background: #F9FAFB !important;
    border-right: 1px solid #E5E7EB !important;
    min-width: 220px !important; max-width: 240px !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 15px !important; font-weight: 600 !important;
    color: #0F0F10 !important; letter-spacing: -0.02em !important; margin-bottom: 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown em { color: #9CA3AF !important; font-size: 11px !important; }
[data-testid="stSidebar"] hr { border-color: #E5E7EB !important; margin: 12px 0 !important; }

/* Навигация — Radio */
/* Скрываем только кружок радиокнопки, НЕ весь baseweb-radio блок */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child { display: none !important; }
/* Скрываем заголовок-label группы радио (не сами пункты) */
[data-testid="stSidebar"] [data-testid="stRadio"] > div > div:first-child > label { display: none !important; }
/* Каждый пункт меню */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important; border: none !important;
    border-radius: 8px !important; padding: 9px 14px !important;
    font-size: 13px !important; font-weight: 400 !important;
    color: #3a3a3c !important; cursor: pointer !important;
    display: flex !important; align-items: center !important; gap: 8px !important;
    width: 100% !important; margin-bottom: 2px !important;
    transition: background .12s !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #EDEDF0 !important; color: #0F0F10 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #EEF2FF !important; color: #4F46E5 !important; font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] p { color: inherit !important; font-size: inherit !important; }
/* Индикатор активной страницы — левая полоска */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)::before {
    content: '' !important; width: 3px !important; height: 14px !important;
    background: #4F46E5 !important; border-radius: 2px !important; flex-shrink: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label::before {
    content: '' !important; width: 3px !important; height: 14px !important;
    background: transparent !important; border-radius: 2px !important; flex-shrink: 0 !important;
}
/* Заголовок группы "// навигация" */
[data-testid="stSidebar"] [data-testid="stRadio"] > div > p {
    font-size: 10px !important; font-weight: 600 !important; color: #9CA3AF !important;
    text-transform: uppercase !important; letter-spacing: .08em !important;
    padding: 4px 14px !important; margin-bottom: 2px !important;
}

/* ── ТИПОГРАФИКА ── */
h1 { font-size: 22px !important; font-weight: 600 !important; color: #0F0F10 !important;
     letter-spacing: -0.025em !important; line-height: 1.25 !important; margin-bottom: 6px !important; }
h3 { font-size: 14px !important; font-weight: 500 !important; color: #1c1c1e !important; }
p, label { font-size: 14px !important; color: #4B5563 !important; line-height: 1.6 !important; }

/* ── КНОПКИ ── */
.stButton > button {
    background: #0F0F10 !important; color: #ffffff !important;
    border: none !important; border-radius: 10px !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 1px 2px rgba(15,15,16,.08), 0 4px 12px rgba(15,15,16,.1) !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    background: #4F46E5 !important; transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(79,70,229,.15), 0 8px 24px rgba(79,70,229,.25) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
/* Кнопка «Показать тур» — вторичный стиль */
[data-testid="stSidebar"] .stButton > button {
    background: #EEF2FF !important; color: #4F46E5 !important;
    border: 1px solid #C7D2FE !important;
    box-shadow: none !important; font-size: 12px !important;
    padding: 0.45rem 1rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #E0E7FF !important; color: #3730A3 !important;
    transform: none !important; box-shadow: none !important;
}

/* ── ИНПУТЫ ── */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #ffffff !important; color: #0F0F10 !important;
    border: 1px solid #E5E7EB !important; border-radius: 8px !important;
    font-size: 14px !important; padding: 9px 13px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: #4F46E5 !important; box-shadow: 0 0 0 3px rgba(79,70,229,.12) !important;
    outline: none !important;
}
.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stSlider label {
    color: #4B5563 !important; font-size: 13px !important; font-weight: 500 !important;
}

/* ── SELECTBOX — белый фон, светлое меню ── */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    color: #0F0F10 !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #D1D5DB !important;
}
[data-testid="stSelectbox"] svg { color: #9CA3AF !important; }
/* Выпадающий список */
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[role="listbox"] {
    background: #ffffff !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.1) !important;
}
[role="option"] {
    background: #ffffff !important;
    color: #0F0F10 !important;
    font-size: 14px !important;
}
[role="option"]:hover,
[aria-selected="true"] {
    background: #EEF2FF !important;
    color: #4F46E5 !important;
}
/* Принудительно: все li/div внутри baseweb-popover — белые */
[data-baseweb="popover"] * {
    background-color: transparent !important;
    color: #0F0F10 !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background-color: #EEF2FF !important;
    color: #4F46E5 !important;
}
[data-baseweb="popover"] { background: #ffffff !important; }

/* ── СЛАЙДЕРЫ ── */
[data-baseweb="slider"] [role="slider"] { background: #4F46E5 !important; border-color: #4F46E5 !important; }

/* ── МЕТРИКИ ── */
[data-testid="metric-container"] {
    background: #F9FAFB !important; border: 1px solid #E5E7EB !important;
    border-radius: 12px !important; padding: 16px 18px !important; box-shadow: none !important;
}
[data-testid="metric-container"] label {
    color: #9CA3AF !important; font-size: 12px !important; font-weight: 500 !important;
    text-transform: none !important; letter-spacing: 0 !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #0F0F10 !important; font-size: 26px !important;
    font-weight: 700 !important; letter-spacing: -0.03em !important;
}
[data-testid="metric-container"] [data-testid="metric-delta"] { color: #059669 !important; font-size: 12px !important; }

.stSpinner > div { border-top-color: #4F46E5 !important; }
.stSuccess { background: #ECFDF5 !important; border: 1px solid #A7F3D0 !important; color: #065F46 !important; border-radius: 10px !important; }
.stError   { background: #FFF1F2 !important; border: 1px solid #FECDD3 !important; color: #9F1239 !important; border-radius: 10px !important; }
.stInfo    { background: #EEF2FF !important; border: 1px solid #C7D2FE !important; color: #3730A3 !important; border-radius: 10px !important; }
.stWarning { background: #FFFBEB !important; border: 1px solid #FDE68A !important; color: #92400E !important; border-radius: 10px !important; }
hr, .stDivider { border-color: #E5E7EB !important; }
[data-testid="collapsedControl"] { background: #F9FAFB !important; border-right: 1px solid #E5E7EB !important; }
.stImage img { border-radius: 10px !important; border: 1px solid #E5E7EB !important; }

/* ── КАСТОМНЫЕ КОМПОНЕНТЫ ── */
.ab-info-box {
    background: #EEF2FF; border-left: 3px solid #4F46E5;
    border-radius: 0 8px 8px 0; padding: 10px 16px;
    font-size: 13px !important; color: #3730A3 !important;
    margin: 10px 0 18px !important; line-height: 1.55 !important;
}
.ab-section-label {
    font-size: 10px !important; font-weight: 600 !important; color: #9CA3AF !important;
    text-transform: uppercase !important; letter-spacing: .08em !important;
    margin-bottom: 12px !important; display: block;
}
.ab-hypo-card {
    background: #ffffff; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 16px 18px; margin-bottom: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,.04); transition: box-shadow .15s;
}
.ab-hypo-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.08); }
.ab-hypo-title    { font-size: 14px !important; font-weight: 600 !important; color: #0F0F10 !important; margin-bottom: 6px !important; }
.ab-hypo-change   { font-size: 13px !important; color: #4B5563 !important; line-height: 1.55 !important; margin-bottom: 3px !important; }
.ab-hypo-expected { font-size: 13px !important; color: #4F46E5 !important; margin-bottom: 3px !important; }
.ab-hypo-meta     { font-size: 12px !important; color: #9CA3AF !important; margin-top: 4px !important; font-style: italic; }
.ab-hypo-footer   { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
.ab-badge         { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 11px !important; font-weight: 500 !important; }
.ab-badge-high    { background: #EEF2FF; color: #4F46E5 !important; }
.ab-badge-mid     { background: #FFF7ED; color: #C2410C !important; }
.ab-badge-low     { background: #F4F4F5; color: #71717A !important; }
.ab-badge-conf    { background: #F9FAFB; color: #6B7280 !important; border: 1px solid #E5E7EB; }
.ab-llm-summary {
    background: #F9FAFB; border-left: 3px solid #4F46E5;
    border-radius: 0 10px 10px 0; padding: 14px 18px;
    font-size: 14px !important; line-height: 1.7 !important; color: #4B5563 !important; margin-top: 10px;
}
.ab-revenue-card {
    background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 12px;
    padding: 13px 18px; margin: 10px 0; display: flex; align-items: baseline; gap: 8px;
}
.ab-revenue-label { font-size: 12px !important; color: #4338CA !important; font-weight: 500 !important; }
.ab-revenue-value { font-size: 26px !important; font-weight: 700 !important; color: #3730A3 !important; letter-spacing: -0.03em !important; }
.ab-placeholder {
    text-align: center; padding: 52px 20px; color: #9CA3AF;
    font-size: 13px !important; border: 1.5px dashed #E5E7EB; border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# ОНБОРДИНГ — через components.html (window.parent обходит iframe sandbox)
# ────────────────────────────────────────────────────────────────────────────

ONBOARDING_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;overflow:hidden;background:transparent">
<script>
(function() {
  var doc = window.parent.document;

  /* ─────────────────────────────────────────────────────────────────
     ШАГИ:
     anchors  — массив CSS-селекторов; highlight обернёт их все в один bbox
     pos      — куда ставить тултип: right / left / bottom / top / center
     Шаги 2,5,6,7: конкретные пункты nav (nth-of-type)
     Шаг 3: весь левый столбец формы (.stColumn:first-child)
     Шаг 4: кнопка «Сгенерировать» (.stButton button)
  ───────────────────────────────────────────────────────────────── */
  var STEPS = [
    {
      title: "Добро пожаловать в AB·AI",
      desc: "Платформа для A/B-тестирования с ИИ. Тур покажет каждый экран и что с ним делать — займёт около минуты.",
      anchors: [],
      pos: "center"
    },
    {
      title: "Меню разделов",
      desc: "Четыре раздела — полный цикл теста: Гипотезы → Планирование → Симуляция → Отчёт. Нажимай по порядку.",
      anchors: ["[data-testid='stSidebar'] [data-testid='stRadio']"],
      pos: "right"
    },
    {
      title: "Заполни контекст продукта",
      desc: "Введи название экрана, выбери метрику, опиши проблемную зону. Чем точнее — тем сильнее гипотезы от LLM.",
      anchors: [
        "[data-testid='stSidebar'] ~ section .stColumn:first-child .stTextInput",
        "[data-testid='stSidebar'] ~ section .stColumn:first-child .stSelectbox",
        "[data-testid='stSidebar'] ~ section .stColumn:first-child .stSlider"
      ],
      groupAll: true,
      pos: "right"
    },
    {
      title: "Нажми «Сгенерировать гипотезы»",
      desc: "После заполнения нажми эту кнопку. LLM вернёт 2–7 конкретных идей с ожидаемым эффектом и оценкой уверенности.",
      anchors: ["[data-testid='stSidebar'] ~ section .stButton button"],
      pos: "top"
    },
    {
      title: "Раздел «Планирование»",
      desc: "Здесь передвинь ползунок MDE — система мгновенно покажет, сколько пользователей собрать и сколько дней займёт тест.",
      anchors: ["[data-testid='stSidebar'] [data-testid='stRadio'] label:nth-child(2)"],
      pos: "right"
    },
    {
      title: "Раздел «Симуляция»",
      desc: "Задай конверсии A и B и нажми ▶. Увидишь, как Thompson Sampling экономит трафик по сравнению с обычным 50/50.",
      anchors: ["[data-testid='stSidebar'] [data-testid='stRadio'] label:nth-child(3)"],
      pos: "right"
    },
    {
      title: "Раздел «Отчёт»",
      desc: "Введи числа контроля и варианта B, нажми «Рассчитать». LLM напишет бизнес-резюме и скажет: внедрять или нет.",
      anchors: ["[data-testid='stSidebar'] [data-testid='stRadio'] label:nth-child(4)"],
      pos: "right"
    }
  ];

  var LS_KEY = "abai_tour_v5";
  var curStep = 0;
  var overlay, hl, tip, arrowEl;

  function isDone(){ try{ return !!localStorage.getItem(LS_KEY); }catch(e){ return false; } }
  function markDone(){ try{ localStorage.setItem(LS_KEY,"1"); }catch(e){} }
  function qs(s){ return doc.querySelector(s); }

  /* ── СТИЛИ ── */
  function injectStyles(){
    if(doc.getElementById("ab-css")) return;
    var s=doc.createElement("style"); s.id="ab-css";
    s.textContent=
      "@keyframes ab-in{from{opacity:0;transform:translateY(4px) scale(.97)}to{opacity:1;transform:none}}"+
      "@keyframes ab-ring{0%,100%{outline-color:#4F46E5}50%{outline-color:#818CF8}}"+
      /* Overlay: затемнение без clip — highlight добавит outline поверх */
      "#ab-ov{position:fixed;inset:0;background:rgba(15,15,16,.48);z-index:9980;pointer-events:none}"+
      /* Highlight: прозрачный блок с outline, поверх overlay */
      "#ab-hl{position:fixed;z-index:9985;pointer-events:none;border-radius:10px;"+
        "outline:3px solid #4F46E5;outline-offset:4px;"+
        "box-shadow:0 0 0 9999px rgba(15,15,16,.48);"+
        "animation:ab-ring 2s ease-in-out infinite;"+
        "transition:top .28s cubic-bezier(.4,0,.2,1),left .28s cubic-bezier(.4,0,.2,1),"+
        "width .28s cubic-bezier(.4,0,.2,1),height .28s cubic-bezier(.4,0,.2,1)}"+
      "#ab-tip{position:fixed;background:#fff;border-radius:14px;padding:20px 22px 16px;width:296px;"+
        "box-shadow:0 16px 48px rgba(0,0,0,.2),0 2px 8px rgba(0,0,0,.08);z-index:9999;"+
        "font-family:Inter,system-ui,sans-serif;"+
        "transition:top .28s cubic-bezier(.4,0,.2,1),left .28s cubic-bezier(.4,0,.2,1);"+
        "animation:ab-in .2s cubic-bezier(.4,0,.2,1)}"+
      /* Стрелка-коннектор */
      "#ab-arr{position:fixed;z-index:9998;pointer-events:none;"+
        "transition:top .28s cubic-bezier(.4,0,.2,1),left .28s cubic-bezier(.4,0,.2,1)}"+
      "#ab-arr.l::before,#ab-arr.r::before,#ab-arr.t::before,#ab-arr.b::before"+
        "{content:'';position:absolute;width:0;height:0}"+
      "#ab-arr.l::before{border-top:9px solid transparent;border-bottom:9px solid transparent;border-right:11px solid #fff;top:-9px;left:0}"+
      "#ab-arr.r::before{border-top:9px solid transparent;border-bottom:9px solid transparent;border-left:11px solid #fff;top:-9px;left:0}"+
      "#ab-arr.t::before{border-left:9px solid transparent;border-right:9px solid transparent;border-bottom:11px solid #fff;top:0;left:-9px}"+
      "#ab-arr.b::before{border-left:9px solid transparent;border-right:9px solid transparent;border-top:11px solid #fff;top:0;left:-9px}"+
      "#ab-welcome-wrap{position:fixed;inset:0;background:rgba(15,15,16,.52);z-index:9995;display:flex;align-items:center;justify-content:center}"+
      "#ab-welcome{background:#fff;border-radius:18px;padding:34px 34px 26px;width:428px;max-width:92vw;"+
        "box-shadow:0 24px 64px rgba(0,0,0,.22);text-align:center;font-family:Inter,system-ui,sans-serif;"+
        "animation:ab-in .3s cubic-bezier(.4,0,.2,1)}"+
      ".abp{background:#0F0F10;color:#fff;border:none;border-radius:9px;padding:9px 17px;"+
        "font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;transition:background .12s}"+
      ".abp:hover{background:#4F46E5}"+
      ".abs{background:transparent;color:#4B5563;border:1px solid #E5E7EB;border-radius:8px;"+
        "padding:7px 12px;font-size:13px;cursor:pointer;font-family:inherit}"+
      ".abs:hover{background:#F9FAFB;color:#0F0F10}"+
      "#ab-toast{position:fixed;bottom:24px;right:24px;background:#0F0F10;color:#fff;"+
        "border-radius:12px;padding:12px 18px;font-family:Inter,system-ui,sans-serif;font-size:13px;"+
        "z-index:9999;box-shadow:0 8px 24px rgba(0,0,0,.22);display:flex;align-items:center;gap:10px;"+
        "animation:ab-in .3s cubic-bezier(.4,0,.2,1)}";
    doc.head.appendChild(s);
  }

  /* ── BOUNDING BOX по нескольким элементам ── */
  function getBBox(anchors, groupAll){
    var els=[], rects=[];
    if(!anchors || !anchors.length) return null;

    if(groupAll){
      /* берём все совпадения каждого селектора */
      anchors.forEach(function(sel){
        doc.querySelectorAll(sel).forEach(function(el){ els.push(el); });
      });
    } else {
      /* берём первый из каждого */
      anchors.forEach(function(sel){
        var el=qs(sel); if(el) els.push(el);
      });
    }
    if(!els.length) return null;

    var minT=Infinity,minL=Infinity,maxB=-Infinity,maxR=-Infinity;
    els.forEach(function(el){
      var r=el.getBoundingClientRect();
      if(r.width===0 && r.height===0) return; /* скрытые пропускаем */
      if(r.top    < minT) minT=r.top;
      if(r.left   < minL) minL=r.left;
      if(r.bottom > maxB) maxB=r.bottom;
      if(r.right  > maxR) maxR=r.right;
    });
    if(minT===Infinity) return null;
    return {top:minT, left:minL, width:maxR-minL, height:maxB-minT,
            right:maxR, bottom:maxB, cx:minL+(maxR-minL)/2, cy:minT+(maxB-minT)/2};
  }

  /* ── ПОЗИЦИОНИРОВАНИЕ HL ── */
  function posHL(bbox){
    if(!hl) return;
    if(!bbox){ hl.style.display="none"; return; }
    var P=6;
    hl.style.display="block";
    hl.style.top    =(bbox.top  -P)+"px";
    hl.style.left   =(bbox.left -P)+"px";
    hl.style.width  =(bbox.width+P*2)+"px";
    hl.style.height =(bbox.height+P*2)+"px";
  }

  /* ── ПОЗИЦИОНИРОВАНИЕ ТУЛТИПА + СТРЕЛКИ ── */
  function posTip(bbox, pos){
    var TW=296, TH=230, PAD=18, VW=window.innerWidth, VH=window.innerHeight;
    var top, left, aC="", aTop, aLeft;

    arrowEl.className=""; arrowEl.style.display="none";

    if(!bbox || pos==="center"){
      top=Math.round(VH/2-TH/2); left=Math.round(VW/2-TW/2);
    } else {
      if(pos==="right"){
        left=bbox.right+PAD; top=Math.round(bbox.cy-TH/2);
        aC="l"; aLeft=bbox.right+PAD-12; aTop=Math.round(bbox.cy);
      } else if(pos==="left"){
        left=bbox.left-TW-PAD; top=Math.round(bbox.cy-TH/2);
        aC="r"; aLeft=bbox.left-PAD+1; aTop=Math.round(bbox.cy);
      } else if(pos==="bottom"){
        top=bbox.bottom+PAD; left=Math.round(bbox.cx-TW/2);
        aC="t"; aLeft=Math.round(bbox.cx); aTop=bbox.bottom+PAD-12;
      } else { /* top */
        top=bbox.top-TH-PAD; left=Math.round(bbox.cx-TW/2);
        aC="b"; aLeft=Math.round(bbox.cx); aTop=bbox.top-PAD+1;
      }
      top  =Math.max(8,Math.min(top,  VH-TH-8));
      left =Math.max(8,Math.min(left, VW-TW-8));
      if(aC){
        arrowEl.className=aC; arrowEl.style.display="block";
        arrowEl.style.top=aTop+"px"; arrowEl.style.left=aLeft+"px";
      }
    }
    tip.style.top=top+"px"; tip.style.left=left+"px";
  }

  /* ── ПРОГРЕСС-ТОЧКИ ── */
  function pips(cur,tot){
    var o="<div style='display:flex;gap:4px;margin-bottom:14px'>";
    for(var i=0;i<tot;i++) o+="<div style='height:3px;flex:1;border-radius:99px;background:"+(i<=cur?"#4F46E5":"#E5E7EB")+";transition:background .3s'></div>";
    return o+"</div>";
  }

  /* ── РЕНДЕР ТУЛТИПА ── */
  function renderTip(idx){
    var s=STEPS[idx],tot=STEPS.length,isF=idx===0,isL=idx===tot-1;
    tip.innerHTML=
      "<span style='font-size:10px;font-weight:600;color:#4F46E5;text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:8px'>Шаг "+(idx+1)+" из "+tot+"</span>"+
      "<div style='font-size:15px;font-weight:600;color:#0F0F10;letter-spacing:-.015em;margin-bottom:7px;line-height:1.3'>"+s.title+"</div>"+
      "<div style='font-size:13px;color:#4B5563;line-height:1.6;margin-bottom:12px'>"+s.desc+"</div>"+
      pips(idx,tot)+
      "<div style='display:flex;align-items:center;gap:8px'>"+
        (!isF?"<button class='abs' id='ab-bk'>← Назад</button>":"")+
        "<button class='abp' id='ab-nx' style='flex:1'>"+(isL?"Готово ✓":"Далее →")+"</button>"+
        (!isL?"<button style='background:none;border:none;color:#9CA3AF;font-size:12px;cursor:pointer;font-family:inherit;padding:4px' id='ab-sk'>Пропустить</button>":"")+
      "</div>";
    tip.querySelector("#ab-nx").onclick=function(){ isL?endTour(true):goStep(idx+1); };
    var bk=tip.querySelector("#ab-bk"); if(bk) bk.onclick=function(){ goStep(idx-1); };
    var sk=tip.querySelector("#ab-sk"); if(sk) sk.onclick=function(){ endTour(false); };
  }

  /* ── ШАГ ── */
  function goStep(idx){
    curStep=idx;
    var s=STEPS[idx];
    var bbox=getBBox(s.anchors, s.groupAll);

    /* Пересоздаём тултип для анимации */
    if(tip){ tip.remove(); }
    tip=doc.createElement("div"); tip.id="ab-tip"; doc.body.appendChild(tip);
    renderTip(idx);

    setTimeout(function(){
      posHL(bbox);
      posTip(bbox, s.pos);
    },15);
  }

  /* ── СТАРТ / КОНЕЦ ── */
  function startTour(){
    var w=doc.getElementById("ab-welcome-wrap"); if(w) w.remove();
    if(!overlay){ overlay=doc.createElement("div"); overlay.id="ab-ov"; doc.body.appendChild(overlay); }
    if(!hl)     { hl=doc.createElement("div");      hl.id="ab-hl";     doc.body.appendChild(hl); }
    if(!arrowEl){ arrowEl=doc.createElement("div"); arrowEl.id="ab-arr"; doc.body.appendChild(arrowEl); }
    goStep(0);
  }

  function endTour(done){
    ["ab-ov","ab-hl","ab-tip","ab-arr"].forEach(function(id){ var e=doc.getElementById(id); if(e)e.remove(); });
    overlay=null; hl=null; tip=null; arrowEl=null;
    if(done){ markDone(); showToast(); }
  }

  function showToast(){
    var t=doc.createElement("div"); t.id="ab-toast";
    t.innerHTML="<span style='font-size:16px'>✓</span><span>Тур завершён — приступайте!</span>"+
      "<button onclick='this.parentNode.remove()' style='background:#ffffff22;border:none;color:#fff;"+
      "border-radius:6px;padding:3px 9px;cursor:pointer;font-size:12px;font-family:inherit;margin-left:4px'>×</button>";
    doc.body.appendChild(t);
    setTimeout(function(){ if(t.parentNode)t.remove(); },5000);
  }

  /* ── WELCOME ── */
  function showWelcome(){
    var wrap=doc.createElement("div"); wrap.id="ab-welcome-wrap";
    wrap.innerHTML=
      "<div id='ab-welcome'>"+
      "<div style='width:52px;height:52px;background:#0F0F10;border-radius:14px;display:flex;align-items:center;"+
        "justify-content:center;font-size:18px;font-weight:700;color:#fff;letter-spacing:-.03em;margin:0 auto 20px'>AB</div>"+
      "<div style='font-size:20px;font-weight:600;color:#0F0F10;letter-spacing:-.02em;margin-bottom:10px'>Добро пожаловать в AB·AI</div>"+
      "<div style='font-size:14px;color:#4B5563;line-height:1.65;margin-bottom:22px'>"+
        "Платформа автоматизации A/B-тестирования с ИИ.<br>Пройдите короткий тур — займёт меньше минуты.</div>"+
      "<div style='text-align:left;margin-bottom:22px;display:flex;flex-direction:column;gap:9px'>"+
      ["Генерируй гипотезы — LLM предложит идеи за секунды",
       "Рассчитай выборку — без формул и таблиц",
       "Запусти адаптивный трафик с Thompson Sampling",
       "Получи бизнес-отчёт одним кликом"]
      .map(function(f){
        return "<div style='display:flex;align-items:center;gap:10px;font-size:13px;color:#4B5563'>"+
          "<div style='width:6px;height:6px;border-radius:50%;background:#4F46E5;flex-shrink:0'></div>"+f+"</div>";
      }).join("")+
      "</div>"+
      "<button class='abp' id='ab-go' style='width:100%;padding:12px 20px;font-size:14px;margin-bottom:10px'>Начать тур →</button>"+
      "<div><button style='background:none;border:none;color:#9CA3AF;font-size:13px;cursor:pointer;font-family:inherit' "+
        "id='ab-no'>Пропустить, разберусь сам</button></div></div>";
    doc.body.appendChild(wrap);
    doc.getElementById("ab-go").onclick=function(){ wrap.remove(); startTour(); };
    doc.getElementById("ab-no").onclick=function(){ wrap.remove(); markDone(); };
  }

  /* ── КНОПКА ПЕРЕЗАПУСКА ── */
  function bindRestart(){
    var obs=new MutationObserver(function(){
      doc.querySelectorAll("[data-testid='stSidebar'] button").forEach(function(btn){
        if(btn.textContent.trim().includes("Показать тур") && !btn._ab){
          btn._ab=true;
          btn.addEventListener("click",function(e){
            e.stopPropagation();
            ["ab-ov","ab-hl","ab-tip","ab-arr","ab-welcome-wrap"].forEach(function(id){
              var el=doc.getElementById(id); if(el)el.remove();
            });
            overlay=null; hl=null; tip=null; arrowEl=null;
            injectStyles(); startTour();
          });
        }
      });
    });
    obs.observe(doc.body,{childList:true,subtree:true});
  }

  injectStyles();
  bindRestart();
  if(!isDone()){ setTimeout(showWelcome,1000); }
})();
</script>
</body>
</html>
"""

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AB·AI")
    st.markdown("*Автоматизация A/B-тестов*")
    st.divider()

    # label "// навигация" нужен как заголовок группы — скрываем его через CSS
    page = st.radio(
        "// навигация",
        options=["Гипотезы", "Планирование", "Симуляция", "Отчёт"],
        label_visibility="visible",
    )

    st.divider()

    if st.button("◎ Показать тур", key="ab_restart_tour", use_container_width=True):
        pass  # JS перехватывает через MutationObserver

    st.markdown(
        "<div style='font-size:11px; color:#9CA3AF; line-height:1.6; margin-top:6px'>"
        "Курсовая работа · 2025<br>"
        "Автоматизация A/B-тестирования с ИИ</div>",
        unsafe_allow_html=True,
    )

# Инжектируем онбординг через components.html — JS работает в window.parent
components.html(ONBOARDING_HTML, height=0, scrolling=False)


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 1: ГЕНЕРАЦИЯ ГИПОТЕЗ
# ────────────────────────────────────────────────────────────────────────────

if page == "Гипотезы":
    st.markdown("### Генерация гипотез")
    st.markdown(
        "<div class='ab-info-box'>LLM анализирует контекст продукта и историю тестов — "
        "формулирует конкретные тестируемые гипотезы с оценкой уверенности.</div>",
        unsafe_allow_html=True,
    )

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<span class='ab-section-label'>Контекст продукта</span>", unsafe_allow_html=True)
        product_name  = st.text_input("Продукт / экран", value="Ozon Express — корзина")
        target_metric = st.selectbox(
            "Целевая метрика",
            ["Конверсия в заказ", "CTR кнопки", "Средний чек", "Время до оплаты", "Другое"],
        )
        baseline_cr  = st.slider("Базовая конверсия (%)", 0.5, 30.0, 3.2, 0.1) / 100
        problem_area = st.text_input("Проблемная зона", value="Кнопка оформления заказа")
        audience     = st.text_input("Целевая аудитория (необязательно)", value="Мобильные пользователи 25–45 лет")
        hypo_count   = st.slider("Количество гипотез", 2, 7, 3)

        st.markdown("<span class='ab-section-label'>История тестов</span>", unsafe_allow_html=True)
        history_raw = st.text_area(
            "history",
            value="Изменение цвета кнопки — не победила\nДобавление таймера — победила, +8%",
            height=90, label_visibility="collapsed",
        )

        generate_btn = st.button("✨ Сгенерировать гипотезы", use_container_width=True)

    with col_result:
        if generate_btn:
            with st.spinner("Запрос к LLM..."):
                prompt = build_hypothesis_prompt(product_name, target_metric, baseline_cr,
                                                  problem_area, audience, history_raw, hypo_count)
                raw = call_llm(prompt)
            try:
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = "\n".join(cleaned.split("\n")[1:]).rstrip("`").strip()
                hypotheses = json.loads(cleaned)
                st.session_state["last_hypotheses"] = hypotheses
            except Exception:
                st.warning("LLM вернул неструктурированный ответ.")
                st.text_area("Ответ LLM", raw, height=300)
                hypotheses = []

            if hypotheses:
                st.markdown(f"<span class='ab-section-label'>Сгенерировано: {len(hypotheses)} гипотез</span>",
                            unsafe_allow_html=True)
                for i, h in enumerate(hypotheses, 1):
                    priority = h.get("priority", "средний")
                    bc = "ab-badge-high" if priority == "высокий" else ("ab-badge-mid" if priority == "средний" else "ab-badge-low")
                    conf = h.get("confidence", 0.5)
                    st.markdown(f"""<div class="ab-hypo-card">
  <div class="ab-hypo-title">#{i} &nbsp; {h.get('title','—')}</div>
  <div class="ab-hypo-change">→ {h.get('change','')}</div>
  <div class="ab-hypo-expected">{h.get('expected','')}</div>
  <div class="ab-hypo-meta">{h.get('reason','')}</div>
  <div class="ab-hypo-footer">
    <span class="ab-badge {bc}">{priority}</span>
    <span class="ab-badge ab-badge-conf">Уверенность: {conf:.0%}</span>
  </div>
</div>""", unsafe_allow_html=True)

        elif "last_hypotheses" in st.session_state:
            st.markdown("<span class='ab-section-label'>Последний результат</span>", unsafe_allow_html=True)
            for i, h in enumerate(st.session_state["last_hypotheses"], 1):
                conf = h.get("confidence", 0.5)
                st.markdown(f"""<div class="ab-hypo-card">
  <div class="ab-hypo-title">#{i} &nbsp; {h.get('title','—')}</div>
  <div class="ab-hypo-change">→ {h.get('change','')}</div>
  <div class="ab-hypo-meta">{h.get('reason','')}</div>
  <div class="ab-hypo-footer"><span class="ab-badge ab-badge-conf">Уверенность: {conf:.0%}</span></div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ab-placeholder'>◆<br><br>Заполни форму слева и нажми «Сгенерировать»</div>",
                        unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 2: ПЛАНИРОВАНИЕ
# ────────────────────────────────────────────────────────────────────────────

elif page == "Планирование":
    st.markdown("### Планирование эксперимента")
    st.markdown(
        "<div class='ab-info-box'>Авторасчёт минимально необходимой выборки по формуле нормального "
        "приближения к биномиальному распределению. Результат обновляется мгновенно.</div>",
        unsafe_allow_html=True,
    )

    col_params, col_plan = st.columns([1, 1], gap="large")

    with col_params:
        st.markdown("<span class='ab-section-label'>Параметры теста</span>", unsafe_allow_html=True)
        p_baseline   = st.slider("Базовая конверсия (%)", 0.5, 30.0, 3.2, 0.1) / 100
        mde_pct      = st.slider("MDE — минимальный эффект (абс., п.п.)", 0.1, 5.0, 0.8, 0.1)
        mde          = mde_pct / 100
        alpha        = st.select_slider("Уровень значимости α", options=[0.01, 0.05, 0.10], value=0.05)
        power        = st.select_slider("Мощность теста 1−β", options=[0.70, 0.80, 0.90], value=0.80)
        daily_traffic = st.number_input("Суточный трафик (сессий/день)", min_value=100, value=1000, step=100)

    with col_plan:
        plan = calculate_sample_size(p_baseline, mde, alpha, power, int(daily_traffic))
        st.markdown("<span class='ab-section-label'>Результат расчёта</span>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        m1.metric("Выборка / группа", f"{plan.required_observations_per_group:,}")
        m2.metric("Всего наблюдений", f"{plan.total_required_observations:,}")
        m3.metric("Длительность", f"{plan.estimated_duration_days} дн." if plan.estimated_duration_days else "—")
        m4.metric("Целевая конверсия B", f"{plan.target_conversion:.2%}")

        st.markdown("<span class='ab-section-label'>Power analysis</span>", unsafe_allow_html=True)
        mde_range = np.linspace(0.002, 0.05, 60)
        n_range = []
        for m in mde_range:
            try:
                n_range.append(calculate_sample_size(p_baseline, m, alpha, power).required_observations_per_group)
            except Exception:
                n_range.append(np.nan)

        fig, ax = plt.subplots(figsize=(5.5, 3), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#E5E7EB"); ax.spines["bottom"].set_color("#E5E7EB")
        ax.tick_params(colors="#9CA3AF", labelsize=10)
        ax.grid(True, alpha=0.35, color="#E5E7EB")
        ax.plot([m * 100 for m in mde_range], n_range, color="#4F46E5", linewidth=2.5)
        ax.fill_between([m * 100 for m in mde_range], n_range, alpha=0.08, color="#4F46E5")
        ax.axvline(mde * 100, color="#DC2626", linestyle="--", linewidth=1.5, label=f"MDE = {mde_pct:.1f} п.п.")
        ax.set_xlabel("MDE (абс., п.п.)", fontsize=11, color="#9CA3AF")
        ax.set_ylabel("Наблюдений / группа", fontsize=11, color="#9CA3AF")
        ax.legend(fontsize=10, framealpha=0)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 3: СИМУЛЯЦИЯ
# ────────────────────────────────────────────────────────────────────────────

elif page == "Симуляция":
    st.markdown("### Адаптивное тестирование")
    st.markdown(
        "<div class='ab-info-box'>Симуляция Multi-Armed Bandit с Thompson Sampling: "
        "каждый вариант моделируется Beta(α,β)-распределением, трафик автоматически "
        "перераспределяется в пользу лидера.</div>",
        unsafe_allow_html=True,
    )

    col_cfg, col_vis = st.columns([1, 1.5], gap="large")

    with col_cfg:
        st.markdown("<span class='ab-section-label'>Параметры симуляции</span>", unsafe_allow_html=True)
        cr_a = st.slider("Истинная конверсия A (%)", 1.0, 20.0, 3.2, 0.1) / 100
        cr_b = st.slider("Истинная конверсия B (%)", 1.0, 20.0, 4.0, 0.1) / 100
        n_visitors = st.select_slider("Число посетителей",
                                       options=[1_000, 5_000, 10_000, 25_000, 50_000], value=10_000)
        sim_seed = st.number_input("Seed", min_value=0, value=42, step=1)
        sim_btn  = st.button("▶ Запустить симуляцию", use_container_width=True)

    with col_vis:
        run_key = f"{cr_a}_{cr_b}_{n_visitors}_{sim_seed}"
        if sim_btn or st.session_state.get("sim_run_key") == run_key:
            if sim_btn:
                with st.spinner("Симуляция..."):
                    result = run_thompson_simulation([cr_a, cr_b], ["Вариант A","Вариант B"],
                                                     n_visitors, max(1, n_visitors//100), int(sim_seed))
                st.session_state["sim_result"]  = result
                st.session_state["sim_run_key"] = run_key

            result = st.session_state.get("sim_result")
            if result:
                rr = (result["total_regret_classic"]-result["total_regret_ts"]) / max(result["total_regret_classic"],1e-9)*100
                winner_cr = [cr_a,cr_b][result["winner_idx"]]
                m1,m2,m3 = st.columns(3)
                m1.metric("Снижение regret", f"{rr:.1f}%", "vs A/B 50/50")
                m2.metric("Победитель", result["winner_label"])
                m3.metric("Posterior mean", f"{winner_cr:.2%}")

                def _ax(a):
                    a.set_facecolor("#ffffff")
                    for sp in ["top","right"]: a.spines[sp].set_visible(False)
                    a.spines["left"].set_color("#E5E7EB"); a.spines["bottom"].set_color("#E5E7EB")
                    a.tick_params(colors="#9CA3AF",labelsize=10); a.grid(True,alpha=0.3,color="#E5E7EB")

                steps=result["steps"]; ts_r=result["ts_regret"]; cl_r=result["classic_regret"]
                traffic_b=[t[1] for t in result["traffic_shares"]]

                fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,3.5),facecolor="#ffffff")
                _ax(ax1); _ax(ax2)
                ax1.plot(steps,ts_r,color="#4F46E5",linewidth=2.5,label="Thompson Sampling")
                ax1.plot(steps,cl_r,color="#EF4444",linewidth=1.8,linestyle="--",label="Классический A/B")
                ax1.fill_between(steps,ts_r,cl_r,alpha=0.07,color="#4F46E5")
                ax1.set_title("Cumulative regret",fontsize=12,color="#0F0F10",fontweight="600",pad=8)
                ax1.set_xlabel("Посетители",color="#9CA3AF",fontsize=10)
                ax1.legend(fontsize=9,framealpha=0)
                ax2.plot(steps,[t*100 for t in traffic_b],color="#4F46E5",linewidth=2.5,label="Вариант B")
                ax2.plot(steps,[(1-t)*100 for t in traffic_b],color="#059669",linewidth=2.5,label="Вариант A")
                ax2.axhline(50,color="#E5E7EB",linestyle="--",linewidth=1.2)
                ax2.set_title("Доля трафика (%)",fontsize=12,color="#0F0F10",fontweight="600",pad=8)
                ax2.set_xlabel("Посетители",color="#9CA3AF",fontsize=10)
                ax2.legend(fontsize=9,framealpha=0)
                fig.tight_layout(pad=1.5); st.pyplot(fig); plt.close(fig)

                st.markdown("<span class='ab-section-label'>Posterior Beta-распределения (финал)</span>", unsafe_allow_html=True)
                fig2,ax3=plt.subplots(figsize=(9,2.8),facecolor="#ffffff"); _ax(ax3)
                x=np.linspace(0,0.12,500)
                for ai,arm in enumerate(result["final_arms"]):
                    c=["#059669","#4F46E5"][ai]; lbl=f"{'Вариант A' if ai==0 else 'Вариант B'} (mean={arm.posterior_mean:.3%})"
                    y=stats.beta.pdf(x,arm.success_count,arm.failure_count)
                    ax3.plot(x,y,color=c,linewidth=2.5,label=lbl); ax3.fill_between(x,y,alpha=0.1,color=c)
                ax3.set_xlabel("Конверсия θ",color="#9CA3AF",fontsize=10)
                ax3.legend(fontsize=10,framealpha=0)
                fig2.tight_layout(pad=1.5); st.pyplot(fig2); plt.close(fig2)
        else:
            st.markdown("<div class='ab-placeholder'>◆<br><br>Задай параметры и нажми «Запустить симуляцию»</div>",
                        unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# СТРАНИЦА 4: ОТЧЁТ
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
        metric_name        = st.text_input("Название метрики", value="конверсия в заказ")
        daily_rev_per_conv = st.number_input("Средняя выручка с конверсии (₽)", min_value=0, value=2500, step=100)
        report_btn = st.button("Рассчитать и сгенерировать отчёт", use_container_width=True)

    with col_out:
        if report_btn:
            p_ctrl  = ctrl_conv  / ctrl_visitors
            p_treat = treat_conv / treat_visitors
            lift_abs = p_treat - p_ctrl
            lift_rel = lift_abs / p_ctrl if p_ctrl > 0 else 0
            pooled   = (ctrl_conv + treat_conv) / (ctrl_visitors + treat_visitors)
            se       = math.sqrt(pooled*(1-pooled)*(1/ctrl_visitors+1/treat_visitors))
            z_score  = lift_abs/se if se > 0 else 0
            p_value  = 2*(1-stats.norm.cdf(abs(z_score)))
            cohens_d = lift_abs/math.sqrt(pooled*(1-pooled)) if pooled > 0 else 0
            se_diff  = math.sqrt(p_ctrl*(1-p_ctrl)/ctrl_visitors + p_treat*(1-p_treat)/treat_visitors)
            ci_lo = lift_abs - 1.96*se_diff; ci_hi = lift_abs + 1.96*se_diff
            monthly_revenue = lift_abs * ctrl_visitors * daily_rev_per_conv

            st.markdown("<span class='ab-section-label'>Результаты</span>", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("p-value",   f"{p_value:.4f}", "✅ значимо" if p_value<0.05 else "❌ не значимо")
            m2.metric("Lift",      f"{lift_rel:+.1%}")
            m3.metric("Cohen's d", f"{cohens_d:.3f}")
            m4.metric("95% CI",    f"[{ci_lo:+.2%}, {ci_hi:+.2%}]")

            st.markdown(
                f"<div class='ab-revenue-card'>"
                f"<span class='ab-revenue-label'>Доп. выручка / мес. (оценка)</span>"
                f"<span class='ab-revenue-value'>{monthly_revenue:,.0f} ₽</span>"
                f"</div>", unsafe_allow_html=True,
            )

            stats_for_llm = {
                "метрика": metric_name,
                "конверсия_контроль": f"{p_ctrl:.3%}", "конверсия_B": f"{p_treat:.3%}",
                "lift_абсолютный": f"{lift_abs:+.3%}", "lift_относительный": f"{lift_rel:+.1%}",
                "p_value": round(p_value,4), "cohens_d": round(cohens_d,3),
                "CI_95": f"[{ci_lo:+.3%}, {ci_hi:+.3%}]",
                "значимость": "да" if p_value<0.05 else "нет",
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
            st.markdown("<div class='ab-placeholder'>◆<br><br>Введи данные теста и нажми «Рассчитать»</div>",
                        unsafe_allow_html=True)

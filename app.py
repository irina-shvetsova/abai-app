"""
AB·AI — Streamlit MVP  (v4 — all fixes)
"""
import json, math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st
import streamlit.components.v1 as components

# ── Sample size ──────────────────────────────────────────────────────────────
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
    def __post_init__(self):
        self.total_required_observations = self.required_observations_per_group * 2

def calculate_sample_size(baseline_conversion, minimum_detectable_effect,
                           significance_level=0.05, statistical_power=0.80,
                           daily_traffic=None):
    z_alpha = stats.norm.ppf(1 - significance_level / 2)
    z_beta  = stats.norm.ppf(statistical_power)
    target_cr = baseline_conversion + minimum_detectable_effect
    variance_sum = (baseline_conversion*(1-baseline_conversion) + target_cr*(1-target_cr))
    raw_n = (z_alpha+z_beta)**2 * variance_sum / minimum_detectable_effect**2
    n = math.ceil(raw_n)
    dur = math.ceil(n*2/daily_traffic) if daily_traffic else None
    return SampleSizePlan(n, baseline_conversion, minimum_detectable_effect,
                          target_cr, significance_level, statistical_power, dur, "proportional")

# ── Thompson Sampling ────────────────────────────────────────────────────────
@dataclass
class ArmState:
    label: str
    success_count: int = 1
    failure_count: int = 1
    @property
    def posterior_mean(self):
        return self.success_count / (self.success_count + self.failure_count)

def run_thompson_simulation(true_rates, arm_labels, visitor_count=10000, snapshot_every=100, seed=42):
    rng_b = np.random.default_rng(seed); rng_s = np.random.default_rng(seed+1)
    arms = [ArmState(l) for l in arm_labels]
    opt = max(true_rates); opt_i = int(np.argmax(true_rates))
    steps_log, reg_log, traf_log = [], [], []
    sel = [0]*len(arms); run_reg = 0.0
    for step in range(1, visitor_count+1):
        chosen = int(np.argmax([rng_b.beta(a.success_count, a.failure_count) for a in arms]))
        if rng_s.random() < true_rates[chosen]: arms[chosen].success_count += 1
        else: arms[chosen].failure_count += 1
        sel[chosen] += 1; run_reg += opt - true_rates[chosen]
        if step % snapshot_every == 0 or step == visitor_count:
            steps_log.append(step); reg_log.append(run_reg)
            traf_log.append([c/step for c in sel])
    cl_reg = []; cl_run = 0.0; rng_c = np.random.default_rng(seed+99)
    for step in range(1, visitor_count+1):
        cl_run += opt - true_rates[rng_c.integers(0, len(arms))]
        if step % snapshot_every == 0 or step == visitor_count: cl_reg.append(cl_run)
    w = int(np.argmax([a.posterior_mean for a in arms]))
    return {"steps":steps_log,"ts_regret":reg_log,"classic_regret":cl_reg,
            "traffic_shares":traf_log,"final_arms":arms,"winner_label":arm_labels[w],
            "winner_idx":w,"total_regret_ts":run_reg,"total_regret_classic":cl_run}

# ── LLM ──────────────────────────────────────────────────────────────────────
def _call_anthropic(key, prompt, system=""):
    try:
        import anthropic; c = anthropic.Anthropic(api_key=key)
        kw = {"model":"claude-sonnet-4-20250514","max_tokens":1500,"messages":[{"role":"user","content":prompt}]}
        if system: kw["system"] = system
        return c.messages.create(**kw).content[0].text
    except Exception as e: return f"[Anthropic error: {e}]"

def _call_yandex(folder, key, prompt, system=""):
    try:
        import urllib.request
        msgs = []
        if system: msgs.append({"role":"system","text":system})
        msgs.append({"role":"user","text":prompt})
        data = json.dumps({"modelUri":f"gpt://{folder}/yandexgpt/latest",
            "completionOptions":{"stream":False,"temperature":0.6,"maxTokens":1500},"messages":msgs}).encode()
        req = urllib.request.Request("https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            data=data,headers={"Content-Type":"application/json","Authorization":f"Api-Key {key}"},method="POST")
        with urllib.request.urlopen(req,timeout=30) as r:
            return json.loads(r.read())["result"]["alternatives"][0]["message"]["text"]
    except Exception as e: return f"[YandexGPT error: {e}]"

def call_llm(prompt, system=""):
    fid = st.secrets.get("YANDEX_FOLDER_ID","") or st.session_state.get("yandex_folder_id","")
    key = st.secrets.get("YANDEX_API_KEY","")   or st.session_state.get("yandex_api_key","")
    if fid and key: return _call_yandex(fid, key, prompt, system)
    akey = st.secrets.get("ANTHROPIC_API_KEY","") or st.session_state.get("anthropic_api_key","")
    if akey: return _call_anthropic(akey, prompt, system)
    return "[API-ключ не указан — добавь в .streamlit/secrets.toml]"

def hyp_prompt(product, metric, cr, problem, audience, history, count):
    return f"""Ты — опытный продуктовый аналитик. Сформулируй {count} гипотез для A/B-теста.
КОНТЕКСТ: Продукт: {product}, Проблема: {problem}, Метрика: {metric}, Конверсия: {cr:.1%}, Аудитория: {audience or 'не указана'}
ИСТОРИЯ: {history or 'нет данных'}
Верни ТОЛЬКО JSON-массив:
[{{"title":"...","change":"...","expected":"...","reason":"...","confidence":0.75,"priority":"высокий"}}]"""

def report_prompt(data):
    return f"""Напиши бизнес-резюме A/B-теста на русском (3–5 предложений): итог, смысл для бизнеса, рекомендация.
Данные: {json.dumps(data,ensure_ascii=False,indent=2)}
Только текст, без заголовков."""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AB·AI", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded", menu_items={})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

#MainMenu,footer,header{visibility:hidden}
*{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;-webkit-font-smoothing:antialiased}
.block-container{padding-top:1.75rem !important;padding-bottom:3rem !important;max-width:1160px !important}
.stApp,.stApp>div,.main{background:#ffffff !important}

/* ── САЙДБАР ── */
[data-testid="stSidebar"]{background:#F9FAFB !important;border-right:1px solid #E5E7EB !important;min-width:220px !important;max-width:240px !important}
[data-testid="stSidebar"] .stMarkdown h2{font-size:15px !important;font-weight:600 !important;color:#0F0F10 !important;letter-spacing:-0.02em !important;margin-bottom:0 !important}
[data-testid="stSidebar"] .stMarkdown p,[data-testid="stSidebar"] .stMarkdown em{color:#9CA3AF !important;font-size:11px !important}
[data-testid="stSidebar"] hr{border-color:#E5E7EB !important;margin:12px 0 !important}

/* ── НАВИГАЦИЯ: скрываем только кружок ── */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"]>div:first-child{display:none !important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{
    background:transparent !important;border:none !important;
    border-radius:8px !important;padding:9px 14px !important;
    font-size:13px !important;font-weight:400 !important;
    /* ВАЖНО: color с максимальным specificity чтобы перебить "p,label" */
    color:#3a3a3c !important;
    cursor:pointer !important;display:flex !important;align-items:center !important;
    gap:8px !important;width:100% !important;margin-bottom:2px !important;
    transition:background .12s !important}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#EDEDF0 !important;color:#0F0F10 !important}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:#EEF2FF !important;color:#4F46E5 !important;font-weight:500 !important}
[data-testid="stSidebar"] [data-testid="stRadio"] p{color:inherit !important;font-size:inherit !important}
/* Активная полоска */
[data-testid="stSidebar"] [data-testid="stRadio"] label::before{content:'' !important;width:3px !important;height:14px !important;background:transparent !important;border-radius:2px !important;flex-shrink:0 !important}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked)::before{background:#4F46E5 !important}
/* Метка группы "// навигация" */
[data-testid="stSidebar"] [data-testid="stRadio"]>div>p{font-size:10px !important;font-weight:600 !important;color:#9CA3AF !important;text-transform:uppercase !important;letter-spacing:.08em !important;padding:4px 14px !important;margin-bottom:2px !important}

/* ── ТИПОГРАФИКА ── */
h1{font-size:22px !important;font-weight:600 !important;color:#0F0F10 !important;letter-spacing:-0.025em !important;line-height:1.25 !important;margin-bottom:6px !important}
h3{font-size:14px !important;font-weight:500 !important;color:#1c1c1e !important}
/* НЕ применяем глобальный color к label — это ломает навигацию */
p{font-size:14px !important;color:#4B5563 !important;line-height:1.6 !important}

/* ── КНОПКИ ── */
.stButton>button{background:#0F0F10 !important;color:#fff !important;border:none !important;border-radius:10px !important;font-size:14px !important;font-weight:500 !important;padding:0.6rem 1.5rem !important;box-shadow:0 1px 2px rgba(15,15,16,.08),0 4px 12px rgba(15,15,16,.1) !important;transition:all .15s !important}
.stButton>button:hover{background:#4F46E5 !important;transform:translateY(-1px) !important;box-shadow:0 2px 6px rgba(79,70,229,.15),0 8px 24px rgba(79,70,229,.25) !important}
.stButton>button:active{transform:translateY(0) !important}
[data-testid="stSidebar"] .stButton>button{background:#EEF2FF !important;color:#4F46E5 !important;border:1px solid #C7D2FE !important;box-shadow:none !important;font-size:12px !important;padding:0.45rem 1rem !important}
[data-testid="stSidebar"] .stButton>button:hover{background:#E0E7FF !important;color:#3730A3 !important;transform:none !important;box-shadow:none !important}

/* ── ИНПУТЫ ── */
.stTextInput input,.stNumberInput input,.stTextArea textarea{background:#fff !important;color:#0F0F10 !important;border:1px solid #E5E7EB !important;border-radius:8px !important;font-size:14px !important;padding:9px 13px !important}
.stTextInput input:focus,.stTextArea textarea:focus,.stNumberInput input:focus{border-color:#4F46E5 !important;box-shadow:0 0 0 3px rgba(79,70,229,.12) !important;outline:none !important}
.stTextInput label,.stNumberInput label,.stTextArea label,.stSelectbox label,.stSlider label{color:#4B5563 !important;font-size:13px !important;font-weight:500 !important}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"]>div>div{background:#fff !important;border:1px solid #E5E7EB !important;border-radius:8px !important;color:#0F0F10 !important}
[data-testid="stSelectbox"] svg{color:#9CA3AF !important}
[data-baseweb="popover"],[data-baseweb="popover"] *{background:#fff !important;color:#0F0F10 !important}
[role="option"]:hover,[aria-selected="true"]{background:#EEF2FF !important;color:#4F46E5 !important}

/* ── СЛАЙДЕРЫ ── */
[data-baseweb="slider"] [role="slider"]{background:#4F46E5 !important;border-color:#4F46E5 !important}

/* ── МЕТРИКИ ── */
[data-testid="metric-container"]{background:#F9FAFB !important;border:1px solid #E5E7EB !important;border-radius:12px !important;padding:16px 18px !important;box-shadow:none !important}
[data-testid="metric-container"] label{color:#9CA3AF !important;font-size:12px !important;font-weight:500 !important;text-transform:none !important;letter-spacing:0 !important}
[data-testid="metric-container"] [data-testid="metric-value"]{color:#0F0F10 !important;font-size:26px !important;font-weight:700 !important;letter-spacing:-0.03em !important}
[data-testid="metric-container"] [data-testid="metric-delta"]{color:#059669 !important;font-size:12px !important}

/* ── ПРОЧЕЕ ── */
.stSpinner>div{border-top-color:#4F46E5 !important}
.stSuccess{background:#ECFDF5 !important;border:1px solid #A7F3D0 !important;color:#065F46 !important;border-radius:10px !important}
.stError{background:#FFF1F2 !important;border:1px solid #FECDD3 !important;color:#9F1239 !important;border-radius:10px !important}
.stInfo{background:#EEF2FF !important;border:1px solid #C7D2FE !important;color:#3730A3 !important;border-radius:10px !important}
hr,.stDivider{border-color:#E5E7EB !important}
[data-testid="collapsedControl"]{background:#F9FAFB !important;border-right:1px solid #E5E7EB !important}
.stImage img{border-radius:10px !important;border:1px solid #E5E7EB !important}

/* ── КАСТОМНЫЕ КОМПОНЕНТЫ ── */
.ab-info-box{background:#EEF2FF;border-left:3px solid #4F46E5;border-radius:0 8px 8px 0;padding:10px 16px;font-size:13px !important;color:#3730A3 !important;margin:10px 0 18px !important;line-height:1.55 !important}
.ab-section-label{font-size:10px !important;font-weight:600 !important;color:#9CA3AF !important;text-transform:uppercase !important;letter-spacing:.08em !important;margin-bottom:12px !important;display:block}
.ab-hypo-card{background:#fff;border:1px solid #E5E7EB;border-radius:12px;padding:16px 18px;margin-bottom:10px;box-shadow:0 1px 2px rgba(0,0,0,.04);transition:box-shadow .15s}
.ab-hypo-card:hover{box-shadow:0 4px 14px rgba(0,0,0,.08)}
.ab-hypo-title{font-size:14px !important;font-weight:600 !important;color:#0F0F10 !important;margin-bottom:6px !important}
.ab-hypo-change{font-size:13px !important;color:#4B5563 !important;line-height:1.55 !important;margin-bottom:3px !important}
.ab-hypo-expected{font-size:13px !important;color:#4F46E5 !important;margin-bottom:3px !important}
.ab-hypo-meta{font-size:12px !important;color:#9CA3AF !important;margin-top:4px !important;font-style:italic}
.ab-hypo-footer{display:flex;align-items:center;gap:8px;margin-top:8px}
.ab-badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px !important;font-weight:500 !important}
.ab-badge-high{background:#EEF2FF;color:#4F46E5 !important}
.ab-badge-mid{background:#FFF7ED;color:#C2410C !important}
.ab-badge-low{background:#F4F4F5;color:#71717A !important}
.ab-badge-conf{background:#F9FAFB;color:#6B7280 !important;border:1px solid #E5E7EB}
.ab-llm-summary{background:#F9FAFB;border-left:3px solid #4F46E5;border-radius:0 10px 10px 0;padding:14px 18px;font-size:14px !important;line-height:1.7 !important;color:#4B5563 !important;margin-top:10px}
.ab-revenue-card{background:#EEF2FF;border:1px solid #C7D2FE;border-radius:12px;padding:13px 18px;margin:10px 0;display:flex;align-items:baseline;gap:8px}
.ab-revenue-label{font-size:12px !important;color:#4338CA !important;font-weight:500 !important}
.ab-revenue-value{font-size:26px !important;font-weight:700 !important;color:#3730A3 !important;letter-spacing:-0.03em !important}
.ab-placeholder{text-align:center;padding:52px 20px;color:#9CA3AF;font-size:13px !important;border:1.5px dashed #E5E7EB;border-radius:12px}
</style>
""", unsafe_allow_html=True)

# ── ОНБОРДИНГ ────────────────────────────────────────────────────────────────
# Ключевые исправления:
# 1. window.parent.innerHeight вместо window.innerHeight (iframe height=0)
# 2. Рамка рисуется отдельным div с border (не outline) — не обрезается overflow:hidden
# 3. Затемнение = псевдо-clip: четыре полосы вокруг highlight-area
ONBOARDING_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;overflow:hidden;background:transparent">
<script>
(function(){
  var P = window.parent;
  var doc = P.document;

  var STEPS = [
    {title:"Добро пожаловать в AB·AI",
     desc:"Платформа для автоматизации A/B-тестирования с ИИ. Короткий тур познакомит с каждым экраном — займёт около минуты.",
     sel:[],pos:"center"},
    {title:"Меню разделов",
     desc:"Четыре раздела — полный цикл теста: Гипотезы → Планирование → Симуляция → Отчёт. Переключай их здесь.",
     sel:["[data-testid='stSidebar'] [data-testid='stRadio']"],pos:"right"},
    {title:"Заполни контекст продукта",
     desc:"Введи название экрана, выбери метрику и опиши проблемную зону. Чем точнее — тем сильнее гипотезы от LLM.",
     sel:["[data-testid='stSidebar'] ~ section .stColumn:first-child"],pos:"right",group:true},
    {title:"Нажми «Сгенерировать гипотезы»",
     desc:"Нажми эту кнопку после заполнения формы. LLM вернёт 2–7 конкретных гипотез с ожидаемым эффектом и уверенностью.",
     sel:["[data-testid='stSidebar'] ~ section .stButton:first-of-type button"],pos:"top"},
    {title:"Раздел «Планирование»",
     desc:"Здесь передвинь ползунок MDE — система мгновенно покажет, сколько пользователей нужно и сколько дней займёт тест.",
     sel:["[data-testid='stSidebar'] [data-testid='stRadio'] [data-baseweb='radio']:nth-child(2) label"],pos:"right"},
    {title:"Раздел «Симуляция»",
     desc:"Задай конверсии A и B, нажми ▶. Увидишь, как Thompson Sampling экономит трафик по сравнению с обычным A/B 50/50.",
     sel:["[data-testid='stSidebar'] [data-testid='stRadio'] [data-baseweb='radio']:nth-child(3) label"],pos:"right"},
    {title:"Раздел «Отчёт»",
     desc:"Введи числа контроля и варианта B, нажми «Рассчитать». LLM напишет резюме и скажет: внедрять изменение или нет.",
     sel:["[data-testid='stSidebar'] [data-testid='stRadio'] [data-baseweb='radio']:nth-child(4) label"],pos:"right"}
  ];

  var LS = "abai_v6";
  var cur = 0;
  var dimT,dimR,dimB,dimL, hlBox, tip, arw;

  function done(){ try{return !!localStorage.getItem(LS);}catch(e){return false;} }
  function mark(){ try{localStorage.setItem(LS,"1");}catch(e){} }
  function qs(s){ return doc.querySelector(s); }

  /* ─ INJECT STYLES ─ */
  function injectCSS(){
    if(doc.getElementById("ab-css6")) return;
    var s=doc.createElement("style"); s.id="ab-css6";
    s.textContent=
      "@keyframes ab-in{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}"+
      /* Четыре полосы затемнения — не обрезаются overflow */
      ".ab-dim{position:fixed;background:rgba(15,15,16,.5);z-index:9980;pointer-events:all;transition:all .25s cubic-bezier(.4,0,.2,1)}"+
      /* Рамка highlight — отдельный div с border */
      "#ab-hl{position:fixed;z-index:9985;pointer-events:none;border-radius:10px;"+
        "border:2.5px solid #4F46E5;box-shadow:0 0 0 4px rgba(79,70,229,.18);"+
        "transition:all .25s cubic-bezier(.4,0,.2,1)}"+
      "#ab-tip{position:fixed;background:#fff;border-radius:14px;padding:20px 22px 16px;width:292px;"+
        "box-shadow:0 16px 48px rgba(0,0,0,.2),0 2px 8px rgba(0,0,0,.07);z-index:9999;"+
        "font-family:Inter,system-ui,sans-serif;animation:ab-in .2s cubic-bezier(.4,0,.2,1)}"+
      "#ab-arw{position:fixed;z-index:9998;pointer-events:none;width:0;height:0}"+
      "#ab-arw.r{border-top:9px solid transparent;border-bottom:9px solid transparent;border-left:11px solid #fff}"+
      "#ab-arw.l{border-top:9px solid transparent;border-bottom:9px solid transparent;border-right:11px solid #fff}"+
      "#ab-arw.b{border-left:9px solid transparent;border-right:9px solid transparent;border-top:11px solid #fff}"+
      "#ab-arw.t{border-left:9px solid transparent;border-right:9px solid transparent;border-bottom:11px solid #fff}"+
      "#ab-ww{position:fixed;inset:0;background:rgba(15,15,16,.5);z-index:9995;display:flex;align-items:center;justify-content:center}"+
      "#ab-wm{background:#fff;border-radius:18px;padding:34px 34px 26px;width:426px;max-width:92vw;"+
        "box-shadow:0 24px 64px rgba(0,0,0,.22);text-align:center;font-family:Inter,system-ui,sans-serif;animation:ab-in .3s}"+
      ".abp{background:#0F0F10;color:#fff;border:none;border-radius:9px;padding:9px 17px;"+
        "font-size:13px;font-weight:500;cursor:pointer;font-family:inherit;transition:background .12s}"+
      ".abp:hover{background:#4F46E5}"+
      ".abs{background:transparent;color:#4B5563;border:1px solid #E5E7EB;border-radius:8px;"+
        "padding:7px 12px;font-size:13px;cursor:pointer;font-family:inherit}"+
      ".abs:hover{background:#F9FAFB}"+
      "#ab-toast{position:fixed;bottom:24px;right:24px;background:#0F0F10;color:#fff;"+
        "border-radius:12px;padding:12px 18px;font-family:Inter,system-ui,sans-serif;font-size:13px;"+
        "z-index:9999;display:flex;align-items:center;gap:10px;animation:ab-in .3s}";
    doc.head.appendChild(s);
  }

  /* ─ DIM: четыре полосы затемнения вокруг highlight ─ */
  function makeDim(){
    var ids=["ab-dt","ab-dr","ab-db","ab-dl"];
    ids.forEach(function(id){
      var d=doc.createElement("div"); d.id=id; d.className="ab-dim"; doc.body.appendChild(d);
    });
    dimT=doc.getElementById("ab-dt"); dimR=doc.getElementById("ab-dr");
    dimB=doc.getElementById("ab-db"); dimL=doc.getElementById("ab-dl");
  }
  function removeDim(){
    ["ab-dt","ab-dr","ab-db","ab-dl"].forEach(function(id){ var e=doc.getElementById(id); if(e)e.remove(); });
    dimT=dimR=dimB=dimL=null;
  }

  /* ─ BBOX ─ */
  function bbox(sels, group){
    var els=[];
    sels.forEach(function(s){
      if(group) doc.querySelectorAll(s).forEach(function(e){els.push(e);});
      else { var e=qs(s); if(e) els.push(e); }
    });
    if(!els.length) return null;
    var mt=1e9,ml=1e9,mb=-1e9,mr=-1e9;
    els.forEach(function(el){
      var r=el.getBoundingClientRect();
      if(r.width===0&&r.height===0) return;
      if(r.top<mt)mt=r.top; if(r.left<ml)ml=r.left;
      if(r.bottom>mb)mb=r.bottom; if(r.right>mr)mr=r.right;
    });
    if(mt===1e9) return null;
    return {t:mt,l:ml,b:mb,r:mr,w:mr-ml,h:mb-mt,cx:ml+(mr-ml)/2,cy:mt+(mb-mt)/2};
  }

  /* ─ POSITION HIGHLIGHT + DIM ─ */
  function setHL(bb){
    if(!bb){
      if(hlBox){hlBox.style.display="none";}
      if(dimT){dimT.style.display="none";dimR.style.display="none";dimB.style.display="none";dimL.style.display="none";}
      return;
    }
    var P=7, VW=P.innerWidth||doc.documentElement.clientWidth, VH=P.innerHeight||doc.documentElement.clientHeight;
    var t=bb.t-P, l=bb.l-P, w=bb.w+P*2, h=bb.h+P*2;
    /* Highlight box */
    hlBox.style.display="block"; hlBox.style.top=t+"px"; hlBox.style.left=l+"px"; hlBox.style.width=w+"px"; hlBox.style.height=h+"px";
    /* Dim: top */
    dimT.style.display="block"; dimT.style.top="0"; dimT.style.left="0"; dimT.style.right="0"; dimT.style.height=t+"px";
    /* Dim: bottom */
    dimB.style.display="block"; dimB.style.top=(t+h)+"px"; dimB.style.left="0"; dimB.style.right="0"; dimB.style.bottom="0"; dimB.style.height=(VH-t-h)+"px";
    /* Dim: left */
    dimL.style.display="block"; dimL.style.top=t+"px"; dimL.style.left="0"; dimL.style.width=l+"px"; dimL.style.height=h+"px";
    /* Dim: right */
    dimR.style.display="block"; dimR.style.top=t+"px"; dimR.style.left=(l+w)+"px"; dimR.style.right="0"; dimR.style.height=h+"px";
  }

  /* ─ POSITION TOOLTIP ─ */
  function setTip(bb, pos){
    var TW=292, TH=230, PAD=16;
    /* ВАЖНО: берём размеры из parent window, а не из iframe */
    var VW=P.innerWidth||doc.documentElement.clientWidth;
    var VH=P.innerHeight||doc.documentElement.clientHeight;
    var t,l,ac="";

    arw.className=""; arw.style.display="none";

    if(!bb||pos==="center"){
      t=Math.round(VH/2-TH/2); l=Math.round(VW/2-TW/2);
    } else {
      if(pos==="right"){
        l=bb.r+PAD; t=Math.round(bb.cy-TH/2);
        ac="l"; arw.style.left=(bb.r+PAD-12)+"px"; arw.style.top=Math.round(bb.cy-9)+"px";
      } else if(pos==="left"){
        l=bb.l-TW-PAD; t=Math.round(bb.cy-TH/2);
        ac="r"; arw.style.left=(bb.l-PAD)+"px"; arw.style.top=Math.round(bb.cy-9)+"px";
      } else if(pos==="bottom"){
        t=bb.b+PAD; l=Math.round(bb.cx-TW/2);
        ac="t"; arw.style.left=Math.round(bb.cx-9)+"px"; arw.style.top=(bb.b+PAD-12)+"px";
      } else { /* top */
        t=bb.t-TH-PAD; l=Math.round(bb.cx-TW/2);
        ac="b"; arw.style.left=Math.round(bb.cx-9)+"px"; arw.style.top=(bb.t-PAD)+"px";
      }
      t=Math.max(8,Math.min(t,VH-TH-8));
      l=Math.max(8,Math.min(l,VW-TW-8));
      if(ac){ arw.className=ac; arw.style.display="block"; }
    }
    tip.style.top=t+"px"; tip.style.left=l+"px";
  }

  /* ─ PROGRESS PIPS ─ */
  function pips(c,tot){
    var o="<div style='display:flex;gap:4px;margin-bottom:14px'>";
    for(var i=0;i<tot;i++) o+="<div style='height:3px;flex:1;border-radius:99px;background:"+(i<=c?"#4F46E5":"#E5E7EB")+"'></div>";
    return o+"</div>";
  }

  /* ─ RENDER TOOLTIP ─ */
  function render(idx){
    var s=STEPS[idx],tot=STEPS.length,f=idx===0,l=idx===tot-1;
    tip.innerHTML=
      "<span style='font-size:10px;font-weight:600;color:#4F46E5;text-transform:uppercase;letter-spacing:.08em;display:block;margin-bottom:8px'>Шаг "+(idx+1)+" из "+tot+"</span>"+
      "<div style='font-size:15px;font-weight:600;color:#0F0F10;letter-spacing:-.015em;margin-bottom:7px;line-height:1.3'>"+s.title+"</div>"+
      "<div style='font-size:13px;color:#4B5563;line-height:1.6;margin-bottom:12px'>"+s.desc+"</div>"+
      pips(idx,tot)+
      "<div style='display:flex;align-items:center;gap:8px'>"+
        (!f?"<button class='abs' id='ab-bk'>← Назад</button>":"")+
        "<button class='abp' id='ab-nx' style='flex:1'>"+(l?"Готово ✓":"Далее →")+"</button>"+
        (!l?"<button style='background:none;border:none;color:#9CA3AF;font-size:12px;cursor:pointer;font-family:inherit;padding:4px' id='ab-sk'>Пропустить</button>":"")+
      "</div>";
    tip.querySelector("#ab-nx").onclick=function(){l?endTour(true):go(idx+1);};
    var bk=tip.querySelector("#ab-bk"); if(bk) bk.onclick=function(){go(idx-1);};
    var sk=tip.querySelector("#ab-sk"); if(sk) sk.onclick=function(){endTour(false);};
  }

  /* ─ GO STEP ─ */
  function go(idx){
    cur=idx;
    var s=STEPS[idx];
    var bb = s.sel.length ? bbox(s.sel, s.group) : null;
    if(tip){tip.remove();}
    tip=doc.createElement("div"); tip.id="ab-tip"; doc.body.appendChild(tip);
    render(idx);
    setTimeout(function(){ setHL(bb); setTip(bb,s.pos); },20);
  }

  /* ─ START / END ─ */
  function start(){
    var w=doc.getElementById("ab-ww"); if(w)w.remove();
    makeDim();
    hlBox=doc.createElement("div"); hlBox.id="ab-hl"; doc.body.appendChild(hlBox);
    arw=doc.createElement("div"); arw.id="ab-arw"; doc.body.appendChild(arw);
    go(0);
  }

  function endTour(ok){
    removeDim();
    ["ab-hl","ab-tip","ab-arw"].forEach(function(id){ var e=doc.getElementById(id); if(e)e.remove(); });
    hlBox=null; tip=null; arw=null;
    if(ok){ mark(); toast(); }
  }

  function toast(){
    var t=doc.createElement("div"); t.id="ab-toast";
    t.innerHTML="<span style='font-size:16px'>✓</span><span>Тур завершён — приступайте!</span>"+
      "<button onclick='this.parentNode.remove()' style='background:#fff3;border:none;color:#fff;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:12px;font-family:inherit;margin-left:4px'>×</button>";
    doc.body.appendChild(t);
    setTimeout(function(){if(t.parentNode)t.remove();},5000);
  }

  /* ─ WELCOME ─ */
  function welcome(){
    var wrap=doc.createElement("div"); wrap.id="ab-ww";
    wrap.innerHTML=
      "<div id='ab-wm'>"+
      "<div style='width:52px;height:52px;background:#0F0F10;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:700;color:#fff;letter-spacing:-.03em;margin:0 auto 20px'>AB</div>"+
      "<div style='font-size:20px;font-weight:600;color:#0F0F10;letter-spacing:-.02em;margin-bottom:10px'>Добро пожаловать в AB·AI</div>"+
      "<div style='font-size:14px;color:#4B5563;line-height:1.65;margin-bottom:22px'>Платформа автоматизации A/B-тестирования с ИИ.<br>Пройди короткий тур — займёт меньше минуты.</div>"+
      "<div style='text-align:left;margin-bottom:22px;display:flex;flex-direction:column;gap:9px'>"+
      ["Генерируй гипотезы — LLM предложит идеи за секунды",
       "Рассчитай выборку — без формул и таблиц",
       "Запусти адаптивный трафик с Thompson Sampling",
       "Получи бизнес-отчёт одним кликом"]
      .map(function(f){return "<div style='display:flex;align-items:center;gap:10px;font-size:13px;color:#4B5563'><div style='width:6px;height:6px;border-radius:50%;background:#4F46E5;flex-shrink:0'></div>"+f+"</div>";})
      .join("")+"</div>"+
      "<button class='abp' id='ab-go' style='width:100%;padding:12px 20px;font-size:14px;margin-bottom:10px'>Начать тур →</button>"+
      "<div><button style='background:none;border:none;color:#9CA3AF;font-size:13px;cursor:pointer;font-family:inherit' id='ab-no'>Пропустить, разберусь сам</button></div></div>";
    doc.body.appendChild(wrap);
    doc.getElementById("ab-go").onclick=function(){wrap.remove();start();};
    doc.getElementById("ab-no").onclick=function(){wrap.remove();mark();};
  }

  /* ─ BIND RESTART BUTTON ─ */
  function bindRestart(){
    new MutationObserver(function(){
      doc.querySelectorAll("[data-testid='stSidebar'] button").forEach(function(btn){
        if(btn.textContent.trim().includes("Показать тур")&&!btn._ab){
          btn._ab=true;
          btn.addEventListener("click",function(e){
            e.stopPropagation();
            removeDim();
            ["ab-hl","ab-tip","ab-arw","ab-ww"].forEach(function(id){ var el=doc.getElementById(id); if(el)el.remove(); });
            hlBox=null; tip=null; arw=null;
            injectCSS(); start();
          });
        }
      });
    }).observe(doc.body,{childList:true,subtree:true});
  }

  injectCSS(); bindRestart();
  if(!done()) setTimeout(welcome,1000);
})();
</script></body></html>"""

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AB·AI")
    st.markdown("*Автоматизация A/B-тестов*")
    st.divider()

    page = st.radio("// навигация", ["Гипотезы","Планирование","Симуляция","Отчёт"],
                    label_visibility="visible")

    st.divider()
    if st.button("◎ Показать тур", key="ab_restart_tour", use_container_width=True):
        pass
    st.markdown("<div style='font-size:11px;color:#9CA3AF;line-height:1.6;margin-top:6px'>"
                "Курсовая работа · 2025<br>Автоматизация A/B-тестирования с ИИ</div>",
                unsafe_allow_html=True)

components.html(ONBOARDING_HTML, height=0, scrolling=False)

# ── СТРАНИЦА 1: ГИПОТЕЗЫ ─────────────────────────────────────────────────────
if page == "Гипотезы":
    st.markdown("### Генерация гипотез")
    st.markdown("<div class='ab-info-box'>LLM анализирует контекст продукта и историю тестов — формулирует конкретные тестируемые гипотезы с оценкой уверенности.</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1,1], gap="large")
    with col_form:
        st.markdown("<span class='ab-section-label'>Контекст продукта</span>", unsafe_allow_html=True)
        product_name  = st.text_input("Продукт / экран", value="Ozon Express — корзина")
        target_metric = st.selectbox("Целевая метрика", ["Конверсия в заказ","CTR кнопки","Средний чек","Время до оплаты","Другое"])
        baseline_cr   = st.slider("Базовая конверсия (%)", 0.5, 30.0, 3.2, 0.1) / 100
        problem_area  = st.text_input("Проблемная зона", value="Кнопка оформления заказа")
        audience      = st.text_input("Целевая аудитория (необязательно)", value="Мобильные пользователи 25–45 лет")
        hypo_count    = st.slider("Количество гипотез", 2, 7, 3)
        st.markdown("<span class='ab-section-label'>История тестов</span>", unsafe_allow_html=True)
        history_raw   = st.text_area("history", value="Изменение цвета кнопки — не победила\nДобавление таймера — победила, +8%", height=90, label_visibility="collapsed")
        gen_btn = st.button("✨ Сгенерировать гипотезы", use_container_width=True)

    with col_result:
        if gen_btn:
            with st.spinner("Запрос к LLM..."):
                raw = call_llm(hyp_prompt(product_name,target_metric,baseline_cr,problem_area,audience,history_raw,hypo_count))
            try:
                cl = raw.strip()
                if cl.startswith("```"): cl = "\n".join(cl.split("\n")[1:]).rstrip("`").strip()
                hyps = json.loads(cl); st.session_state["last_hyps"] = hyps
            except: st.warning("LLM вернул неструктурированный ответ."); st.text_area("Ответ", raw, height=300); hyps = []
            if hyps:
                st.markdown(f"<span class='ab-section-label'>Сгенерировано: {len(hyps)} гипотез</span>", unsafe_allow_html=True)
                for i,h in enumerate(hyps,1):
                    p=h.get("priority","средний"); bc="ab-badge-high" if p=="высокий" else ("ab-badge-mid" if p=="средний" else "ab-badge-low")
                    st.markdown(f"""<div class="ab-hypo-card"><div class="ab-hypo-title">#{i} &nbsp; {h.get('title','—')}</div><div class="ab-hypo-change">→ {h.get('change','')}</div><div class="ab-hypo-expected">{h.get('expected','')}</div><div class="ab-hypo-meta">{h.get('reason','')}</div><div class="ab-hypo-footer"><span class="ab-badge {bc}">{p}</span><span class="ab-badge ab-badge-conf">Уверенность: {h.get('confidence',0.5):.0%}</span></div></div>""", unsafe_allow_html=True)
        elif "last_hyps" in st.session_state:
            st.markdown("<span class='ab-section-label'>Последний результат</span>", unsafe_allow_html=True)
            for i,h in enumerate(st.session_state["last_hyps"],1):
                st.markdown(f"""<div class="ab-hypo-card"><div class="ab-hypo-title">#{i} &nbsp; {h.get('title','—')}</div><div class="ab-hypo-change">→ {h.get('change','')}</div><div class="ab-hypo-meta">{h.get('reason','')}</div><div class="ab-hypo-footer"><span class="ab-badge ab-badge-conf">Уверенность: {h.get('confidence',0.5):.0%}</span></div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ab-placeholder'>◆<br><br>Заполни форму слева и нажми «Сгенерировать»</div>", unsafe_allow_html=True)

# ── СТРАНИЦА 2: ПЛАНИРОВАНИЕ ─────────────────────────────────────────────────
elif page == "Планирование":
    st.markdown("### Планирование эксперимента")
    st.markdown("<div class='ab-info-box'>Авторасчёт минимальной выборки. Результат обновляется мгновенно при изменении любого параметра.</div>", unsafe_allow_html=True)
    col_p, col_r = st.columns([1,1], gap="large")
    with col_p:
        st.markdown("<span class='ab-section-label'>Параметры</span>", unsafe_allow_html=True)
        p_base = st.slider("Базовая конверсия (%)", 0.5, 30.0, 3.2, 0.1) / 100
        mde_p  = st.slider("MDE (абс., п.п.)", 0.1, 5.0, 0.8, 0.1); mde = mde_p/100
        alpha  = st.select_slider("Уровень значимости α", [0.01,0.05,0.10], value=0.05)
        power  = st.select_slider("Мощность 1−β", [0.70,0.80,0.90], value=0.80)
        dtraf  = st.number_input("Суточный трафик", min_value=100, value=1000, step=100)
    with col_r:
        plan = calculate_sample_size(p_base, mde, alpha, power, int(dtraf))
        st.markdown("<span class='ab-section-label'>Результат</span>", unsafe_allow_html=True)
        m1,m2=st.columns(2); m3,m4=st.columns(2)
        m1.metric("Выборка / группа", f"{plan.required_observations_per_group:,}")
        m2.metric("Всего наблюдений", f"{plan.total_required_observations:,}")
        m3.metric("Длительность", f"{plan.estimated_duration_days} дн." if plan.estimated_duration_days else "—")
        m4.metric("Целевая конверсия B", f"{plan.target_conversion:.2%}")
        st.markdown("<span class='ab-section-label'>Power analysis</span>", unsafe_allow_html=True)
        mde_r=np.linspace(0.002,0.05,60); n_r=[]
        for m in mde_r:
            try: n_r.append(calculate_sample_size(p_base,m,alpha,power).required_observations_per_group)
            except: n_r.append(np.nan)
        fig,ax=plt.subplots(figsize=(5.5,3),facecolor="#fff"); ax.set_facecolor("#fff")
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#E5E7EB"); ax.spines["bottom"].set_color("#E5E7EB")
        ax.tick_params(colors="#9CA3AF",labelsize=10); ax.grid(True,alpha=0.3,color="#E5E7EB")
        ax.plot([m*100 for m in mde_r],n_r,color="#4F46E5",linewidth=2.5)
        ax.fill_between([m*100 for m in mde_r],n_r,alpha=0.08,color="#4F46E5")
        ax.axvline(mde_p,color="#DC2626",linestyle="--",linewidth=1.5,label=f"MDE={mde_p:.1f}п.п.")
        ax.set_xlabel("MDE (п.п.)",fontsize=11,color="#9CA3AF"); ax.set_ylabel("Наблюдений/группу",fontsize=11,color="#9CA3AF")
        ax.legend(fontsize=10,framealpha=0); fig.tight_layout(pad=1.5); st.pyplot(fig); plt.close(fig)

# ── СТРАНИЦА 3: СИМУЛЯЦИЯ ────────────────────────────────────────────────────
elif page == "Симуляция":
    st.markdown("### Адаптивное тестирование")
    st.markdown("<div class='ab-info-box'>Thompson Sampling: трафик автоматически перераспределяется в пользу лидирующего варианта.</div>", unsafe_allow_html=True)
    col_c,col_v=st.columns([1,1.5],gap="large")
    with col_c:
        st.markdown("<span class='ab-section-label'>Параметры</span>", unsafe_allow_html=True)
        cr_a=st.slider("Конверсия A (%)",1.0,20.0,3.2,0.1)/100
        cr_b=st.slider("Конверсия B (%)",1.0,20.0,4.0,0.1)/100
        nv=st.select_slider("Посетители",[1000,5000,10000,25000,50000],value=10000)
        sd=st.number_input("Seed",min_value=0,value=42,step=1)
        sim_btn=st.button("▶ Запустить симуляцию",use_container_width=True)
    with col_v:
        rk=f"{cr_a}_{cr_b}_{nv}_{sd}"
        if sim_btn or st.session_state.get("sim_rk")==rk:
            if sim_btn:
                with st.spinner("Симуляция..."):
                    res=run_thompson_simulation([cr_a,cr_b],["Вариант A","Вариант B"],nv,max(1,nv//100),int(sd))
                st.session_state["sim_res"]=res; st.session_state["sim_rk"]=rk
            res=st.session_state.get("sim_res")
            if res:
                rr=(res["total_regret_classic"]-res["total_regret_ts"])/max(res["total_regret_classic"],1e-9)*100
                wc=[cr_a,cr_b][res["winner_idx"]]
                m1,m2,m3=st.columns(3)
                m1.metric("Снижение regret",f"{rr:.1f}%","vs A/B 50/50")
                m2.metric("Победитель",res["winner_label"])
                m3.metric("Posterior mean",f"{wc:.2%}")
                def ax_style(a):
                    a.set_facecolor("#fff")
                    for sp in ["top","right"]: a.spines[sp].set_visible(False)
                    a.spines["left"].set_color("#E5E7EB"); a.spines["bottom"].set_color("#E5E7EB")
                    a.tick_params(colors="#9CA3AF",labelsize=10); a.grid(True,alpha=0.3,color="#E5E7EB")
                st_l=res["steps"]; ts=res["ts_regret"]; cl=res["classic_regret"]; tb=[t[1] for t in res["traffic_shares"]]
                fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.5),facecolor="#fff"); ax_style(a1); ax_style(a2)
                a1.plot(st_l,ts,color="#4F46E5",linewidth=2.5,label="Thompson Sampling")
                a1.plot(st_l,cl,color="#EF4444",linewidth=1.8,linestyle="--",label="Классический A/B")
                a1.fill_between(st_l,ts,cl,alpha=0.07,color="#4F46E5")
                a1.set_title("Cumulative regret",fontsize=12,color="#0F0F10",fontweight="600",pad=8); a1.legend(fontsize=9,framealpha=0)
                a2.plot(st_l,[t*100 for t in tb],color="#4F46E5",linewidth=2.5,label="Вариант B")
                a2.plot(st_l,[(1-t)*100 for t in tb],color="#059669",linewidth=2.5,label="Вариант A")
                a2.axhline(50,color="#E5E7EB",linestyle="--",linewidth=1.2)
                a2.set_title("Доля трафика (%)",fontsize=12,color="#0F0F10",fontweight="600",pad=8); a2.legend(fontsize=9,framealpha=0)
                fig.tight_layout(pad=1.5); st.pyplot(fig); plt.close(fig)
                st.markdown("<span class='ab-section-label'>Posterior Beta-распределения</span>", unsafe_allow_html=True)
                fig2,a3=plt.subplots(figsize=(9,2.8),facecolor="#fff"); ax_style(a3)
                x=np.linspace(0,0.12,500)
                for ai,arm in enumerate(res["final_arms"]):
                    c=["#059669","#4F46E5"][ai]; lbl=f"{'Вариант A' if ai==0 else 'Вариант B'} (mean={arm.posterior_mean:.3%})"
                    y=stats.beta.pdf(x,arm.success_count,arm.failure_count)
                    a3.plot(x,y,color=c,linewidth=2.5,label=lbl); a3.fill_between(x,y,alpha=0.1,color=c)
                a3.set_xlabel("Конверсия θ",color="#9CA3AF",fontsize=10); a3.legend(fontsize=10,framealpha=0)
                fig2.tight_layout(pad=1.5); st.pyplot(fig2); plt.close(fig2)
        else:
            st.markdown("<div class='ab-placeholder'>◆<br><br>Задай параметры и нажми «Запустить»</div>", unsafe_allow_html=True)

# ── СТРАНИЦА 4: ОТЧЁТ ────────────────────────────────────────────────────────
elif page == "Отчёт":
    st.markdown("### Анализ результатов")
    st.markdown("<div class='ab-info-box'>Введи сырые результаты теста — система посчитает статистику и сгенерирует бизнес-резюме через LLM.</div>", unsafe_allow_html=True)
    ci,co=st.columns([1,1],gap="large")
    with ci:
        st.markdown("<span class='ab-section-label'>Результаты теста</span>", unsafe_allow_html=True)
        cv=st.number_input("Контроль — посетители",min_value=10,value=3000)
        cc=st.number_input("Контроль — конверсии",min_value=0,value=150)
        tv=st.number_input("Вариант B — посетители",min_value=10,value=3000)
        tc=st.number_input("Вариант B — конверсии",min_value=0,value=185)
        mn=st.text_input("Название метрики",value="конверсия в заказ")
        rv=st.number_input("Средняя выручка с конверсии (₽)",min_value=0,value=2500,step=100)
        rb=st.button("Рассчитать и сгенерировать отчёт",use_container_width=True)
    with co:
        if rb:
            pc=cc/cv; pt=tc/tv; la=pt-pc; lr=la/pc if pc>0 else 0
            pool=(cc+tc)/(cv+tv); se=math.sqrt(pool*(1-pool)*(1/cv+1/tv))
            z=la/se if se>0 else 0; pv=2*(1-stats.norm.cdf(abs(z)))
            cd=la/math.sqrt(pool*(1-pool)) if pool>0 else 0
            sd2=math.sqrt(pc*(1-pc)/cv+pt*(1-pt)/tv); lo=la-1.96*sd2; hi=la+1.96*sd2
            rev=la*cv*rv
            st.markdown("<span class='ab-section-label'>Результаты</span>", unsafe_allow_html=True)
            m1,m2,m3,m4=st.columns(4)
            m1.metric("p-value",f"{pv:.4f}","✅ значимо" if pv<0.05 else "❌ не значимо")
            m2.metric("Lift",f"{lr:+.1%}"); m3.metric("Cohen's d",f"{cd:.3f}")
            m4.metric("95% CI",f"[{lo:+.2%},{hi:+.2%}]")
            st.markdown(f"<div class='ab-revenue-card'><span class='ab-revenue-label'>Доп. выручка / мес.</span><span class='ab-revenue-value'>{rev:,.0f} ₽</span></div>", unsafe_allow_html=True)
            d={"метрика":mn,"конверсия_контроль":f"{pc:.3%}","конверсия_B":f"{pt:.3%}",
               "lift_абс":f"{la:+.3%}","lift_отн":f"{lr:+.1%}","p_value":round(pv,4),
               "cohens_d":round(cd,3),"CI_95":f"[{lo:+.3%},{hi:+.3%}]",
               "значимость":"да" if pv<0.05 else "нет","доп_выручка_руб":int(rev)}
            with st.spinner("LLM генерирует резюме..."):
                summ=call_llm(report_prompt(d))
            st.markdown("<span class='ab-section-label'>LLM-резюме</span>", unsafe_allow_html=True)
            st.markdown(f"<div class='ab-llm-summary'>{summ}</div>", unsafe_allow_html=True)
            st.session_state["last_rep"]={"stats":d,"summary":summ}
        elif "last_rep" in st.session_state:
            r=st.session_state["last_rep"]
            st.markdown("<span class='ab-section-label'>Последний отчёт</span>", unsafe_allow_html=True)
            st.json(r["stats"])
            st.markdown(f"<div class='ab-llm-summary'>{r['summary']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ab-placeholder'>◆<br><br>Введи данные теста и нажми «Рассчитать»</div>", unsafe_allow_html=True)

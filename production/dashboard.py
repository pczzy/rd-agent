"""持仓看板：当日风险 + 日线经典技术分析 + 半日线 price action。

    streamlit run production/dashboard.py --server.address 0.0.0.0 --server.port 8501

行情从新浪现抓（日线 scale=240、60 分钟线 scale=60），缓存 5 分钟，所以盘中刷新页面
就能拿到最新的一根。持仓、成本、现金读 state/ 下的状态文件，与 run_daily.py 同源。

这个页面**只看不下单**，也不写任何状态 —— 它是观察窗口，不是控制面板。风控的真正
执行仍在 run_daily.py 里，页面上的告警只是把同一批阈值提前显示出来。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from production.indicators import (  # noqa: E402
    add_indicators,
    break_of_structure,
    classic_read,
    levels,
    pa_volume_read,
    patterns,
    session_volume,
    structure,
    swings,
    to_half_day,
)
from production.pipeline import load_cash, load_config, load_cost_basis, load_positions  # noqa: E402

SINA = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "CN_MarketData.getKLineData?symbol={sym}&scale={scale}&ma=no&datalen={n}")
HEADERS = {"User-Agent": "Mozilla/5.0"}
COLOR = {"多": "#d62728", "空": "#2ca02c", "中性": "#7f7f7f"}   # A 股习惯：红涨绿跌

st.set_page_config(page_title="持仓看板", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- 数据


@st.cache_data(ttl=300, show_spinner=False)
def kline(code: str, scale: int, n: int) -> pd.DataFrame:
    """新浪 K 线。scale 单位是分钟：240=日线，60=小时线（A 股一天 4 根）。"""
    r = requests.get(SINA.format(sym=code.lower(), scale=scale, n=n), headers=HEADERS, timeout=20)
    text = r.text.strip()
    if not text or text[0] != "[":
        return pd.DataFrame()
    df = pd.DataFrame(json.loads(text))
    if df.empty:
        return df
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def last_price(code: str) -> float | None:
    d = kline(code, 240, 3)
    return None if d.empty else float(d["close"].iloc[-1])


def names() -> dict[str, str]:
    p = ROOT / "state/names.json"
    return json.loads(p.read_text()) if p.exists() else {}


def latest_signals() -> tuple[pd.DataFrame, str | None]:
    files = sorted((ROOT / "signals").glob("*.csv"))
    if not files:
        return pd.DataFrame(), None
    return pd.read_csv(files[-1]), files[-1].stem


def peak_equity() -> float | None:
    p = ROOT / "state/equity.csv"
    if not p.exists():
        return None
    rows = pd.read_csv(p)
    return float(rows["equity"].max()) if len(rows) else None


# --------------------------------------------------------------------------- 图


def daily_chart(d: pd.DataFrame, name: str) -> go.Figure:
    x = d["day"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.16, 0.17, 0.17],
                        subplot_titles=(f"{name} 日线", "成交量", "MACD", "RSI / KDJ"))
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="日线", increasing_line_color="#d62728",
                                 decreasing_line_color="#2ca02c"), row=1, col=1)
    for n, c in (("MA5", "#ff7f0e"), ("MA10", "#1f77b4"), ("MA20", "#9467bd"), ("MA60", "#8c564b")):
        fig.add_trace(go.Scatter(x=x, y=d[n], name=n, line=dict(width=1, color=c)), row=1, col=1)
    for n, dash in (("BOLL_UP", "dot"), ("BOLL_LOW", "dot")):
        fig.add_trace(go.Scatter(x=x, y=d[n], name=n, line=dict(width=1, dash=dash, color="#aaa"),
                                 showlegend=False), row=1, col=1)

    up = d["close"] >= d["open"]
    fig.add_trace(go.Bar(x=x, y=d["volume"], name="成交量", showlegend=False,
                         marker_color=[COLOR["多"] if u else COLOR["空"] for u in up]), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["VOL_MA5"], name="量MA5", line=dict(width=1, color="#666"),
                             showlegend=False), row=2, col=1)

    fig.add_trace(go.Bar(x=x, y=d["MACD"], name="MACD",
                         marker_color=[COLOR["多"] if v >= 0 else COLOR["空"] for v in d["MACD"]],
                         showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["DIF"], name="DIF", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["DEA"], name="DEA", line=dict(width=1)), row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=d["RSI14"], name="RSI14", line=dict(width=1.4)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["K"], name="K", line=dict(width=1, dash="dot")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["D"], name="D", line=dict(width=1, dash="dot")), row=4, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line=dict(width=1, dash="dash", color="#bbb"), row=4, col=1)

    # x 轴按类别排，不按真实时间 —— 否则周末和停牌会在图上留下空洞
    fig.update_xaxes(type="category", nticks=12)
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06))
    return fig


def half_chart(hb: pd.DataFrame, res: list[float], sup: list[float], name: str) -> go.Figure:
    s = swings(session_volume(hb))
    x = s["seq"].str.slice(5)      # 去掉年份，标签太长
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.74, 0.26],
                        subplot_titles=(f"{name} 半日线（软件里的“4 小时线”，一天两根）",
                                        "段内量比（上午比上午、下午比下午，已消掉 U 形）"))
    fig.add_trace(go.Candlestick(x=x, open=s["open"], high=s["high"], low=s["low"], close=s["close"],
                                 name="半日线", increasing_line_color="#d62728",
                                 decreasing_line_color="#2ca02c"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[s["摆高"]], y=s.loc[s["摆高"], "high"] * 1.004, mode="markers",
                             marker=dict(symbol="triangle-down", size=9, color="#2ca02c"), name="摆高"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x[s["摆低"]], y=s.loc[s["摆低"], "low"] * 0.996, mode="markers",
                             marker=dict(symbol="triangle-up", size=9, color="#d62728"), name="摆低"),
                  row=1, col=1)
    for v in res:
        fig.add_hline(y=v, line=dict(width=1, dash="dash", color="#2ca02c"), row=1, col=1,
                      annotation_text=f"阻力 {v:.2f}", annotation_position="right")
    for v in sup:
        fig.add_hline(y=v, line=dict(width=1, dash="dash", color="#d62728"), row=1, col=1,
                      annotation_text=f"支撑 {v:.2f}", annotation_position="right")

    # 画相对量而不是原始量：原始量在半日线上是"高一根低一根"的锯齿，看不出异动
    up = s["close"] >= s["open"]
    fig.add_trace(go.Bar(x=x, y=s["段内量比"], name="段内量比", showlegend=False,
                         marker_color=[COLOR["多"] if u else COLOR["空"] for u in up]), row=2, col=1)
    fig.add_hline(y=1.0, line=dict(width=1, dash="dash", color="#888"), row=2, col=1)
    fig.update_xaxes(type="category", nticks=14)
    fig.update_layout(height=620, margin=dict(l=10, r=80, t=50, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06))
    return fig


def read_badges(items: list[tuple[str, str, str]]) -> None:
    for dim, text, level in items:
        st.markdown(
            f"<div style='padding:4px 0'><b>{dim}</b> "
            f"<span style='color:{COLOR[level]}'>{text}</span></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- 页面


cfg = load_config()
nm = names()
pos = load_positions(ROOT / "state/positions.json")
cost = load_cost_basis(ROOT / "state/positions.json")
cash = load_cash(ROOT / "state/cash.json")
sig, sig_date = latest_signals()

st.sidebar.title("持仓看板")
page = st.sidebar.radio("页面", ["风险总览", "个股分析", "目标组合"], label_visibility="collapsed")
if st.sidebar.button("清缓存并重取行情"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("行情缓存 5 分钟。持仓/现金读 state/，与 run_daily.py 同源。\n\n本页只看不下单。")

held = sorted(pos)
target = sig[sig["shares"] > 0]["code"].tolist() if not sig.empty else []


# ---- 风险总览 -------------------------------------------------------------
if page == "风险总览":
    st.subheader("组合风险")
    if not pos:
        st.info("当前空仓。成交后用 `record_trades.py` 记账，这里才有内容。")
    else:
        rows = []
        for c in held:
            px = last_price(c)
            mv = (px or 0) * pos[c]
            basis = cost.get(c)
            rows.append({
                "代码": c, "名称": nm.get(c, ""), "股数": pos[c],
                "成本": basis, "现价": px, "市值": mv,
                "浮盈": (px / basis - 1) if px and basis else None,
                "权重": mv,
            })
        t = pd.DataFrame(rows)
        held_value = float(t["市值"].sum())
        capital = cfg["account"]["capital"]
        equity = held_value + (cash if cash is not None else max(0.0, capital - held_value))
        peak = max(peak_equity() or equity, equity)
        dd = 0.0 if peak <= 0 else 1 - equity / peak
        halt = cfg["risk"]["halt_on_drawdown"]
        t["权重"] = t["权重"] / capital

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("净值", f"{equity:,.0f} 元", f"{equity - capital:+,.0f}")
        c2.metric("持仓市值", f"{held_value:,.0f} 元", f"占资金 {held_value / capital:.1%}")
        c3.metric("现金", f"{(cash if cash is not None else capital - held_value):,.0f} 元")
        c4.metric("回撤", f"{dd:.1%}", f"熔断线 {halt:.0%}", delta_color="off")

        st.dataframe(
            t.style.format({"成本": "{:.2f}", "现价": "{:.2f}", "市值": "{:,.0f}",
                            "浮盈": "{:+.2%}", "权重": "{:.1%}"}),
            width="stretch", hide_index=True)

        st.subheader("告警")
        alerts = []
        if cash is None:
            alerts.append(("warning", "无 state/cash.json，现金按总资金倒推，回撤数字不可信 —— "
                                      "用 record_trades.py 记账后才有意义"))
        if dd >= halt:
            alerts.append(("error", f"回撤 {dd:.1%} 已达熔断线 {halt:.0%}：下次调仓只允许卖出"))
        cap_w = cfg["account"]["max_weight_per_stock"]
        for _, r in t.iterrows():
            if r["权重"] > cap_w:
                alerts.append(("error", f"{r['代码']} {r['名称']} 权重 {r['权重']:.1%} 超过单只上限 "
                                        f"{cap_w:.1%}（{capital * cap_w:,.0f} 元）"))
            if r["浮盈"] is not None and r["浮盈"] <= -0.30:
                alerts.append(("error", f"{r['代码']} {r['名称']} 浮亏 {r['浮盈']:.1%}，"
                                        "退市/暴雷这类事模型看不到，需人工核查基本面"))
        off_plan = [c for c in held if target and c not in target]
        for c in off_plan:
            alerts.append(("warning", f"{c} {nm.get(c, '')} 不在最新目标组合里，"
                                      "下次调仓会出清仓单（若为有意持有，记得从清单里剔除）"))
        if cash is not None and target:
            # 要比的是"还得掏多少现金"和"手上有多少现金"，不是目标市值和净值：
            # 已经持有的目标股不用再买，而不打算卖的超配仓位（比如有意持有的那只）
            # 也变不成现金。拿净值去比会永远比得过，这条告警就永远不响。
            picked = sig[sig["shares"] > 0]
            in_target = float(sum((last_price(c) or 0) * pos[c] for c in held if c in target))
            need_new = float((picked["price"] * picked["shares"]).sum()) - in_target
            if need_new > cash:
                alerts.append(("warning",
                               f"建满目标组合还需约 {need_new:,.0f} 元，可用现金只有 {cash:,.0f} 元。"
                               f"config 的 capital 仍按 {capital:,.0f} 算 —— 若不卖掉计划外的仓位，"
                               "调仓清单会出现买不起的买单"))
        if not alerts:
            st.success("无告警")
        for kind, msg in alerts:
            getattr(st, kind)(msg)


# ---- 个股分析 -------------------------------------------------------------
elif page == "个股分析":
    pool = held + [c for c in target if c not in held]
    if not pool:
        st.info("没有持仓，也没有信号文件。")
        st.stop()
    code = st.selectbox("股票", pool,
                        format_func=lambda c: f"{c} {nm.get(c, '')}" + ("（持仓）" if c in held else ""))
    name = nm.get(code, code)

    d = kline(code, 240, 260)
    if d.empty:
        st.error("新浪没返回该股日线")
        st.stop()
    d = add_indicators(d)

    st.subheader(f"{code} {name}")
    px = float(d["close"].iloc[-1])
    prev = float(d["close"].iloc[-2])
    cols = st.columns(4)
    cols[0].metric("最新价", f"{px:.2f}", f"{px / prev - 1:+.2%}")
    if code in pos:
        basis = cost.get(code)
        cols[1].metric("持仓", f"{pos[code]} 股", f"成本 {basis:.2f}" if basis else "")
        if basis:
            cols[2].metric("浮盈", f"{(px - basis) * pos[code]:+,.0f} 元", f"{px / basis - 1:+.2%}")
    if not sig.empty and code in set(sig["code"]):
        r = sig[sig["code"] == code].iloc[0]
        cols[3].metric("模型打分", f"{r['score']:.4f}", f"排名 {int(r['rank'])} / {len(sig)}")

    tab1, tab2 = st.tabs(["日线 · 经典技术分析", "半日线 · price action"])

    with tab1:
        left, right = st.columns([3, 1])
        with left:
            st.plotly_chart(daily_chart(d.tail(120), name), width="stretch")
        with right:
            st.markdown("**指标解读**")
            read_badges(classic_read(d))
            st.caption(f"数据 {d['day'].iloc[0]} ~ {d['day'].iloc[-1]}，共 {len(d)} 根日线。")

    with tab2:
        h60 = kline(code, 60, 400)
        if h60.empty:
            st.error("新浪没返回该股 60 分钟线")
        else:
            hb = to_half_day(h60)
            res, sup = levels(hb)
            trend, notes = structure(hb)
            left, right = st.columns([3, 1])
            with left:
                st.plotly_chart(half_chart(hb.tail(60), res, sup, name), width="stretch")
            with right:
                st.markdown("**结构**")
                st.markdown(f"<span style='color:{COLOR['多'] if 'HH' in trend else COLOR['空'] if 'LH' in trend else COLOR['中性']}'>{trend}</span>",
                            unsafe_allow_html=True)
                for n_ in notes:
                    st.caption(n_)
                bos = break_of_structure(hb)
                if bos:
                    st.warning(bos)
                st.markdown("**当前这一段的形态**")
                pt = patterns(hb)
                st.write("\n".join(f"- {p}" for p in pt) if pt else "无明显形态")
                st.markdown("**关键位**")
                st.write("阻力：" + "、".join(f"{v:.2f}" for v in res) if res else "阻力：暂无")
                st.write("支撑：" + "、".join(f"{v:.2f}" for v in sup) if sup else "支撑：暂无")
                st.markdown("**量能**")
                read_badges(pa_volume_read(hb))
            last = hb.iloc[-1]
            st.caption(f"最后一段：{last['seq']}（{last['段']}） "
                       f"开 {last['open']:.2f} 高 {last['high']:.2f} 低 {last['low']:.2f} 收 {last['close']:.2f}")
            st.info("半日线一天两根（上午 9:30–11:30、下午 13:00–15:00），两根拼回去等于日线。"
                    "**别在这条线上看量能指标**：A 股日内成交量呈 U 形，上午占全天 56%~66%，"
                    "量能柱天生高一根低一根，那是午休不是资金进出。同理，这里的均线周期要减半理解。")


# ---- 目标组合 -------------------------------------------------------------
else:
    if sig.empty:
        st.info("还没有信号文件，先跑 run_daily.py。")
        st.stop()
    st.subheader(f"目标组合（信号日 {sig_date}）")
    picked = sig[sig["shares"] > 0].copy()
    st.caption(f"入选 {len(picked)} 只，全池 {len(sig)} 只。持仓中的会标出来。")

    if st.checkbox("拉取技术面（30 次请求，实测约 2 秒，缓存 5 分钟）", value=True):
        rows = []
        bar = st.progress(0.0)
        for i, (_, r) in enumerate(picked.iterrows(), 1):
            d = kline(r["code"], 240, 120)
            item = {"排名": int(r["rank"]), "代码": r["code"], "名称": r["name"],
                    "打分": r["score"], "持仓": "✓" if r["code"] in pos else ""}
            if not d.empty and len(d) >= 60:
                d = add_indicators(d)
                last = d.iloc[-1]
                item |= {"现价": float(last["close"]),
                         "距MA20": float(last["close"] / last["MA20"] - 1),
                         "RSI14": float(last["RSI14"]),
                         "MACD": float(last["MACD"]),
                         "量比": float(last["量比"])}
            rows.append(item)
            bar.progress(i / len(picked))
        bar.empty()
        t = pd.DataFrame(rows)
        # 只格式化真实存在的列：某只股票取不到行情时它那几列就整列缺席，
        # 直接把完整字典交给 style.format 会 KeyError，整页白屏。
        fmt = {"打分": "{:.4f}", "现价": "{:.2f}", "距MA20": "{:+.1%}",
               "RSI14": "{:.1f}", "MACD": "{:+.3f}", "量比": "{:.2f}"}
        st.dataframe(t.style.format({k: v for k, v in fmt.items() if k in t.columns}),
                     width="stretch", hide_index=True, height=640)
    else:
        st.dataframe(picked[["rank", "code", "name", "score", "price", "shares", "status"]],
                     width="stretch", hide_index=True, height=640)

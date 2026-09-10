"""持仓看板：当日风险 + 手工记账 + 日线经典技术分析 + 半日线 price action。

    streamlit run production/dashboard.py --server.address 0.0.0.0 --server.port 8501
    （已装成 systemd 服务 rdagent-dashboard）

**页面上每一个价格都带它出现的时间。** 实时价来自新浪的逐笔快照接口（自带日期和时分秒），
K 线上的价格标它所属的那根 K 线，成本价标它由哪几笔、什么时候加权出来的。一个说不出
时间的价格没法验证，也没法用来对账。

「持仓管理」页会写 state/trades.csv（经 ledger.py，与命令行 record_trades.py 同一套实现）。
除此之外本页只读不写，**任何情况下都不下单** —— 风控的执行仍在 run_daily.py 里。
"""

from __future__ import annotations

import datetime as dt
import json
import re
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
from production.ledger import (  # noqa: E402
    add_trades,
    changes,
    drop_trade,
    duplicate_of,
    fee_of,
    read_ledger,
    replay,
)
from production.pipeline import load_cash, load_config, load_positions  # noqa: E402

KLINE = ("http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "CN_MarketData.getKLineData?symbol={sym}&scale={scale}&ma=no&datalen={n}")
REALTIME = "http://hq.sinajs.cn/list={syms}"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
COLOR = {"多": "#d62728", "空": "#2ca02c", "中性": "#7f7f7f"}   # A 股习惯：红涨绿跌
CODE_RE = re.compile(r"^S[HZ]\d{6}$")

st.set_page_config(page_title="持仓看板", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- 行情


@st.cache_data(ttl=300, show_spinner=False)
def kline(code: str, scale: int, n: int) -> pd.DataFrame:
    """新浪 K 线。scale 单位是分钟：240=日线，60=小时线（A 股一天 4 根）。"""
    r = requests.get(KLINE.format(sym=code.lower(), scale=scale, n=n), headers=HEADERS, timeout=20)
    text = r.text.strip()
    if not text or text[0] != "[":
        return pd.DataFrame()
    df = pd.DataFrame(json.loads(text))
    if df.empty:
        return df
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def quotes(codes: tuple[str, ...]) -> dict[str, dict]:
    """实时快照，一次可取多只。**这个接口自带日期和时分秒**，是页面上时间戳的来源。

    K 线接口的日线只给到日期，说不清"这个价是几点的"；盘中尤其要紧 —— 同一天里
    14:00 的价和 15:00 的价是两回事。
    """
    if not codes:
        return {}
    r = requests.get(REALTIME.format(syms=",".join(c.lower() for c in codes)),
                     headers=HEADERS, timeout=15)
    r.encoding = "gbk"
    out: dict[str, dict] = {}
    for code, line in zip(codes, r.text.strip().splitlines()):
        if '"' not in line:
            continue
        f = line.split('"')[1].split(",")
        if len(f) < 33 or not f[3]:
            continue
        try:
            out[code] = {"name": f[0], "open": float(f[1]), "prev": float(f[2]),
                         "price": float(f[3]), "high": float(f[4]), "low": float(f[5]),
                         "volume": float(f[8]), "ts": f"{f[30]} {f[31]}"}
        except ValueError:
            continue
    return out


def quote(code: str) -> dict | None:
    return quotes((code,)).get(code)


def at(value: float | None, ts: str | None, digits: int = 2) -> str:
    """页面上所有价格都走这里：价 + 它出现的时间。"""
    if value is None:
        return "—"
    return f"{value:.{digits}f}" + (f"  @ {ts}" if ts else "  @ 时间未知")


def short(ts: str | None) -> str:
    """表格里空间紧张时去掉年份。"""
    return "" if not ts else str(ts)[5:]


def names() -> dict[str, str]:
    p = ROOT / "state/names.json"
    return json.loads(p.read_text()) if p.exists() else {}


def positions_detail() -> dict[str, dict]:
    """持仓明细，含成本价的形成时间（ledger 写进 positions.json 的）。"""
    p = ROOT / "state/positions.json"
    return json.loads(p.read_text()) if p.exists() else {}


def cost_window(h: dict) -> str:
    lots, a, b = h.get("cost_lots", 1), h.get("cost_from"), h.get("cost_to")
    if not a:
        return "时间未知"
    return a if lots == 1 else f"{a} ~ {b}（{lots} 笔加权）"


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


def tone(v) -> str:
    """盈亏上色。A 股习惯红涨绿跌，与 K 线图一致 —— 页面上两处颜色含义必须一样。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, str):
        return f"color:{COLOR['多']}" if v == "买入" else f"color:{COLOR['空']}"
    return f"color:{COLOR['多']};font-weight:600" if v > 0 else (
        f"color:{COLOR['空']};font-weight:600" if v < 0 else "")


def paint(styler, cols: list[str]):
    """给存在的列上色。列可能因取价失败整列缺席，subset 传不存在的列会抛 KeyError。"""
    present = [c for c in cols if c in styler.data.columns]
    return styler.map(tone, subset=present) if present else styler


def read_badges(items: list[tuple[str, str, str]]) -> None:
    for dim, text, level in items:
        st.markdown(f"<div style='padding:4px 0'><b>{dim}</b> "
                    f"<span style='color:{COLOR[level]}'>{text}</span></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- 图


def daily_chart(d: pd.DataFrame, name: str) -> go.Figure:
    x = d["day"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.16, 0.17, 0.17],
                        subplot_titles=(f"{name} 日线（{d['day'].iloc[0]} ~ {d['day'].iloc[-1]}）",
                                        "成交量", "MACD", "RSI / KDJ"))
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 name="日线", increasing_line_color="#d62728",
                                 decreasing_line_color="#2ca02c"), row=1, col=1)
    for n, c in (("MA5", "#ff7f0e"), ("MA10", "#1f77b4"), ("MA20", "#9467bd"), ("MA60", "#8c564b")):
        fig.add_trace(go.Scatter(x=x, y=d[n], name=n, line=dict(width=1, color=c)), row=1, col=1)
    for n in ("BOLL_UP", "BOLL_LOW"):
        fig.add_trace(go.Scatter(x=x, y=d[n], name=n, line=dict(width=1, dash="dot", color="#aaa"),
                                 showlegend=False), row=1, col=1)

    up = d["close"] >= d["open"]
    fig.add_trace(go.Bar(x=x, y=d["volume"], name="成交量", showlegend=False,
                         marker_color=[COLOR["多"] if u else COLOR["空"] for u in up]), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["VOL_MA5"], name="量MA5", line=dict(width=1, color="#666"),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=d["MACD"], name="MACD", showlegend=False,
                         marker_color=[COLOR["多"] if v >= 0 else COLOR["空"] for v in d["MACD"]]),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["DIF"], name="DIF", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["DEA"], name="DEA", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["RSI14"], name="RSI14", line=dict(width=1.4)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["K"], name="K", line=dict(width=1, dash="dot")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["D"], name="D", line=dict(width=1, dash="dot")), row=4, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line=dict(width=1, dash="dash", color="#bbb"), row=4, col=1)

    # x 轴按类别排，不按真实时间 —— 否则周末和停牌会在图上留下空洞
    fig.update_xaxes(type="category", nticks=12)
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=44, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06))
    return fig


def half_chart(hb: pd.DataFrame, res: list[tuple[float, str]], sup: list[tuple[float, str]],
               name: str) -> go.Figure:
    s = swings(session_volume(hb))
    x = s["seq"].str.slice(5)      # 去掉年份，标签太长
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.74, 0.26],
                        subplot_titles=(f"{name} 半日线 {s['seq'].iloc[0]} ~ {s['seq'].iloc[-1]}"
                                        "（软件里的“4 小时线”，一天两根）",
                                        "段内量比（上午比上午、下午比下午，已消掉 U 形）"))
    fig.add_trace(go.Candlestick(x=x, open=s["open"], high=s["high"], low=s["low"], close=s["close"],
                                 name="半日线", increasing_line_color="#d62728",
                                 decreasing_line_color="#2ca02c"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[s["摆高"]], y=s.loc[s["摆高"], "high"] * 1.004, mode="markers",
                             marker=dict(symbol="triangle-down", size=9, color="#2ca02c"),
                             name="摆高"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[s["摆低"]], y=s.loc[s["摆低"], "low"] * 0.996, mode="markers",
                             marker=dict(symbol="triangle-up", size=9, color="#d62728"),
                             name="摆低"), row=1, col=1)
    # 关键位标上它是哪一根 K 线留下的
    for v, t in res:
        fig.add_hline(y=v, line=dict(width=1, dash="dash", color="#2ca02c"), row=1, col=1,
                      annotation_text=f"阻力 {v:.2f} · {short(t)}", annotation_position="right")
    for v, t in sup:
        fig.add_hline(y=v, line=dict(width=1, dash="dash", color="#d62728"), row=1, col=1,
                      annotation_text=f"支撑 {v:.2f} · {short(t)}", annotation_position="right")

    up = s["close"] >= s["open"]
    fig.add_trace(go.Bar(x=x, y=s["段内量比"], name="段内量比", showlegend=False,
                         marker_color=[COLOR["多"] if u else COLOR["空"] for u in up]), row=2, col=1)
    fig.add_hline(y=1.0, line=dict(width=1, dash="dash", color="#888"), row=2, col=1)
    fig.update_xaxes(type="category", nticks=14)
    fig.update_layout(height=640, margin=dict(l=10, r=110, t=54, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06))
    return fig


# --------------------------------------------------------------------------- 页面

cfg = load_config()
nm = names()
detail = positions_detail()
pos = load_positions(ROOT / "state/positions.json")
cash = load_cash(ROOT / "state/cash.json")
sig, sig_date = latest_signals()
capital = cfg["account"]["capital"]

st.sidebar.title("持仓看板")
page = st.sidebar.radio("页面", ["风险总览", "持仓管理", "个股分析", "目标组合"],
                        label_visibility="collapsed")
if st.sidebar.button("重取行情"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption("实时价缓存 60 秒、K 线缓存 5 分钟。页面上每个价格都标了它出现的时间。\n\n"
                   "「持仓管理」写 state/trades.csv，其余页面只读。本页任何情况下都不下单。")

held = sorted(pos)
target = sig[sig["shares"] > 0]["code"].tolist() if not sig.empty else []


# ---- 风险总览 -------------------------------------------------------------
if page == "风险总览":
    st.subheader("组合风险")
    if not pos:
        st.info("当前空仓。到「持仓管理」记一笔成交，这里才有内容。")
    else:
        q = quotes(tuple(held))
        rows = []
        for c in held:
            info = q.get(c)
            px = info["price"] if info else None
            mv = (px or 0) * pos[c]
            h = detail.get(c, {})
            basis = h.get("cost")
            rows.append({
                "代码": c, "名称": nm.get(c, ""), "股数": pos[c],
                "成本价": basis, "成本形成于": cost_window(h),
                "现价": px, "报价时间": info["ts"] if info else "取价失败",
                "市值": mv, "浮盈": (px / basis - 1) if px and basis else None,
                "权重": mv / capital,
            })
        t = pd.DataFrame(rows)
        held_value = float(t["市值"].sum())
        equity = held_value + (cash if cash is not None else max(0.0, capital - held_value))
        peak = max(peak_equity() or equity, equity)
        dd = 0.0 if peak <= 0 else 1 - equity / peak
        halt = cfg["risk"]["halt_on_drawdown"]
        as_of = max((r["报价时间"] for r in rows if r["现价"]), default="—")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("净值", f"{equity:,.0f} 元", f"{equity - capital:+,.0f}")
        c2.metric("持仓市值", f"{held_value:,.0f} 元", f"占资金 {held_value / capital:.1%}")
        c3.metric("现金", f"{(cash if cash is not None else capital - held_value):,.0f} 元")
        c4.metric("回撤", f"{dd:.1%}", f"熔断线 {halt:.0%}", delta_color="off")
        eq_rows = pd.read_csv(ROOT / "state/equity.csv") if (ROOT / "state/equity.csv").exists() else pd.DataFrame()
        span = f"{eq_rows['date'].iloc[0]} ~ {eq_rows['date'].iloc[-1]}" if len(eq_rows) else "无记录"
        st.caption(f"峰值 {peak:,.0f} 元取自 `state/equity.csv`（{span}，由 run_daily.py 写入）"
                   "与当前净值的较大者；本页不写 equity.csv，所以两次调仓之间的净值不进历史。")
        st.caption(f"净值 = 持仓市值 + 现金。持仓市值按实时价计，报价时间 **{as_of}**；"
                   f"现金来自成交流水（末次更新 {json.loads((ROOT / 'state/cash.json').read_text())['as_of']}）。"
                   if cash is not None else f"报价时间 {as_of}")

        st.dataframe(paint(t.style.format({"成本价": "{:.3f}", "现价": "{:.2f}", "市值": "{:,.0f}",
                                           "浮盈": "{:+.2%}", "权重": "{:.1%}"}, na_rep="—"), ["浮盈"]),
                     width="stretch", hide_index=True)

        st.subheader("告警")
        alerts = []
        if cash is None:
            alerts.append(("warning", "无 state/cash.json，现金按总资金倒推，回撤数字不可信 —— "
                                      "到「持仓管理」把成交记进去才有意义"))
        if dd >= halt:
            alerts.append(("error", f"回撤 {dd:.1%} 已达熔断线 {halt:.0%}：下次调仓只允许卖出"))
        cap_w = cfg["account"]["max_weight_per_stock"]
        for _, r in t.iterrows():
            if r["权重"] > cap_w:
                alerts.append(("error", f"{r['代码']} {r['名称']} 权重 {r['权重']:.1%} 超过单只上限 "
                                        f"{cap_w:.1%}（{capital * cap_w:,.0f} 元）"
                                        f"，按 {at(r['现价'], r['报价时间'])} 计"))
            if r["浮盈"] is not None and r["浮盈"] <= -0.30:
                alerts.append(("error", f"{r['代码']} {r['名称']} 浮亏 {r['浮盈']:.1%}"
                                        f"（成本 {r['成本价']:.2f}，现价 {at(r['现价'], r['报价时间'])}），"
                                        "退市/暴雷这类事模型看不到，需人工核查基本面"))
        for c in [c for c in held if target and c not in target]:
            alerts.append(("warning", f"{c} {nm.get(c, '')} 不在最新目标组合里（信号日 {sig_date}），"
                                      "下次调仓会出清仓单（若为有意持有，记得从清单里剔除）"))
        if cash is not None and target:
            picked = sig[sig["shares"] > 0]
            in_target = float(sum((q.get(c, {}).get("price") or 0) * pos[c] for c in held if c in target))
            need_new = float((picked["price"] * picked["shares"]).sum()) - in_target
            if need_new > cash:
                alerts.append(("warning",
                               f"建满目标组合还需约 {need_new:,.0f} 元（按信号日 {sig_date} 的价格估），"
                               f"可用现金只有 {cash:,.0f} 元。config 的 capital 仍按 {capital:,.0f} 算 —— "
                               "若不卖掉计划外的仓位，调仓清单会出现买不起的买单"))
        if not alerts:
            st.success("无告警")
        for kind, msg in alerts:
            getattr(st, kind)(msg)


# ---- 持仓管理 -------------------------------------------------------------
elif page == "持仓管理":
    st.subheader("持仓管理")
    st.caption("这里写的是**已经在券商成交**的事实，不是下单。流水 state/trades.csv 是唯一事实来源，"
               "positions.json 和 cash.json 都由它重放得出。")

    if pos:
        q = quotes(tuple(held))
        rows = []
        for c in held:
            info, h = q.get(c), detail.get(c, {})
            basis, px = h.get("cost"), (info or {}).get("price")
            rows.append({"代码": c, "名称": nm.get(c, ""), "股数": pos[c],
                         "成本价": basis, "成本形成于": cost_window(h),
                         "现价": px, "报价时间": (info or {}).get("ts", "取价失败"),
                         "浮盈": (px / basis - 1) if px and basis else None})
        st.dataframe(paint(pd.DataFrame(rows).style.format(
            {"成本价": "{:.3f}", "现价": "{:.2f}", "浮盈": "{:+.2%}"}, na_rep="—"), ["浮盈"]),
            width="stretch", hide_index=True)
    else:
        st.info("当前空仓。")

    st.markdown("### 记一笔成交")
    pool = sorted(set(list(nm) + held))
    pick = st.selectbox("股票", pool, format_func=lambda c: f"{c} {nm.get(c, '')}",
                        index=pool.index(held[0]) if held else 0)
    live = quote(pick)
    if live:
        st.caption(f"实时参考价：**{at(live['price'], live['ts'])}** ｜ "
                   f"今开 {live['open']:.2f} ｜ 昨收 {live['prev']:.2f} ｜ "
                   f"最高 {live['high']:.2f} ｜ 最低 {live['low']:.2f}")
    else:
        st.caption("取不到实时价，成交价请照券商成交回报手工填写。")

    with st.form("add_trade", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        side = c1.radio("方向", ["买入", "卖出"], horizontal=True)
        shares = c2.number_input("股数", min_value=1, step=100,
                                 value=100 if side == "买入" else max(pos.get(pick, 100), 1))
        price = c3.number_input("成交价", min_value=0.001, step=0.01, format="%.3f",
                                value=float(live["price"]) if live else 1.0)
        c4, c5, c6 = st.columns(3)
        date = c4.date_input("成交日期", value=dt.date.today())
        tm = c5.text_input("成交时间 HH:MM", value=dt.datetime.now().strftime("%H:%M"),
                           help="填券商回报上的成交时间。留空则只记日期，成本价将说不清是几点的价。")
        note = c6.text_input("备注", placeholder="限价单 / 分批第 2 笔 / 计划外")
        force = st.checkbox("允许与流水中完全重复的一笔（确实分两笔成交时勾选）")
        submitted = st.form_submit_button("记入流水", type="primary")

    if submitted:
        signed = int(shares) * (1 if side == "买入" else -1)
        tm = tm.strip()
        err = None
        if not CODE_RE.match(pick):
            err = f"代码 {pick} 不合法"
        elif tm and not re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", tm):
            err = f"时间 {tm!r} 格式不对，应为 HH:MM"
        elif signed < 0 and -signed > pos.get(pick, 0):
            err = f"卖出 {-signed} 股超过持有的 {pos.get(pick, 0)} 股"
        if err:
            st.error(err)
        else:
            row = {"date": date.strftime("%Y-%m-%d"), "time": tm, "code": pick,
                   "name": nm.get(pick, ""), "shares": signed, "price": float(price),
                   "fee": fee_of(abs(signed) * float(price), signed < 0, cfg), "note": note}
            if not force and duplicate_of(read_ledger(), row):
                st.error("流水里已有同日同股同量同价的一笔。确实分两笔成交请勾选上面的选项。")
            else:
                add_trades([row], cfg)
                st.success(f"已记入：{row['date']} {tm} {side} {pick} {abs(signed)} 股 "
                           f"@ {price:.3f}，费用 {row['fee']:.2f} 元")
                st.rerun()

    st.markdown("### 持仓变动明细")
    ch = changes(cfg)
    if ch.empty:
        st.info("流水为空。")
    else:
        st.dataframe(paint(ch.style.format({"成交价": "{:.3f}", "费用": "{:.2f}"}, na_rep="—"), ["方向"]),
                     width="stretch", hide_index=True)
        st.caption("每一行左边是这笔成交做了什么，右边是做完以后变成了什么。成交价的时间即该行「成交时间」。")

        st.markdown("### 改错")
        led = read_ledger()
        opts = {i: f"[{i}] {r['date']} {r['time']} {r['code']} {r['name']} "
                   f"{'买' if r['shares'] > 0 else '卖'}{abs(int(r['shares']))}股 @{r['price']}"
                for i, r in led.iterrows()}
        idx = st.selectbox("选一行删除（删完自动按剩余流水重放）", list(opts), format_func=opts.get)
        if st.button("删除这一行", type="secondary"):
            try:
                drop_trade(int(idx), cfg)
                st.success(f"已删除 {opts[idx]}，并已重放持仓与现金")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    book, ledger_cash, last_ts = replay(cfg)
    invested = sum(h["shares"] * h["cost"] for h in book.values())
    live = quotes(tuple(sorted(book)))
    market = sum((live.get(c, {}).get("price") or 0) * h["shares"] for c, h in book.items())
    mkt_ts = max((v["ts"] for v in live.values()), default="—")
    # 只有一个数配叫"净值"：现价市值 + 现金。成本市值只用来看浮盈，混着报会和风险总览对不上。
    st.markdown(f"**重放结果**（末笔成交 {last_ts or '无'}）：持仓 {len(book)} 只，"
                f"现金 {ledger_cash:,.2f} 元")
    c1, c2, c3 = st.columns(3)
    c1.metric("净值（现价口径）", f"{market + ledger_cash:,.0f} 元")
    c1.caption(f"= 现价市值 {market:,.0f} + 现金，报价时间 {mkt_ts}")
    c2.metric("现价市值", f"{market:,.0f} 元")
    c2.caption(f"报价时间 {mkt_ts}")
    c3.metric("成本市值", f"{invested:,.0f} 元")
    c3.caption(f"买入价口径，与现价市值之差 {market - invested:+,.0f} 元即当前浮盈")
    st.caption("「净值」与风险总览页是同一个数（现价市值 + 现金）。成本市值只用于算浮盈，"
               "不参与净值 —— 两处若报不同口径的合计，必然对不上。")


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
    live = quote(code)

    st.subheader(f"{code} {name}")
    cols = st.columns(4)
    if live:
        cols[0].metric("实时价", f"{live['price']:.2f}", f"{live['price'] / live['prev'] - 1:+.2%}")
        cols[0].caption(f"@ {live['ts']}")
    else:
        last = float(d["close"].iloc[-1])
        cols[0].metric("收盘价", f"{last:.2f}")
        cols[0].caption(f"@ {d['day'].iloc[-1]} 收盘（实时价取不到）")
    if code in pos:
        h = detail.get(code, {})
        basis = h.get("cost")
        cols[1].metric("持仓", f"{pos[code]} 股", f"成本 {basis:.2f}" if basis else "")
        cols[1].caption(f"成本形成于 {cost_window(h)}")
        if basis and live:
            cols[2].metric("浮盈", f"{(live['price'] - basis) * pos[code]:+,.0f} 元",
                           f"{live['price'] / basis - 1:+.2%}")
            cols[2].caption(f"按 {at(live['price'], live['ts'])}")
    if not sig.empty and code in set(sig["code"]):
        r = sig[sig["code"] == code].iloc[0]
        cols[3].metric("模型打分", f"{r['score']:.4f}", f"排名 {int(r['rank'])} / {len(sig)}")
        cols[3].caption(f"信号日 {sig_date}，当时价 {r['price']:.2f}")

    tab1, tab2 = st.tabs(["日线 · 经典技术分析", "半日线 · price action"])

    with tab1:
        left, right = st.columns([3, 1])
        with left:
            st.plotly_chart(daily_chart(d.tail(120), name), width="stretch")
        with right:
            st.markdown(f"**指标解读**　<span style='color:#888;font-size:12px'>"
                        f"截至 {d['day'].iloc[-1]} 收盘</span>", unsafe_allow_html=True)
            read_badges(classic_read(d))
            last = d.iloc[-1]
            st.caption(f"最新一根日线 {last['day']}：开 {last['open']:.2f} 高 {last['high']:.2f} "
                       f"低 {last['low']:.2f} 收 {last['close']:.2f}；"
                       f"MA20 {last['MA20']:.2f}、布林上轨 {last['BOLL_UP']:.2f}、"
                       f"下轨 {last['BOLL_LOW']:.2f}（均为该日收盘口径）。"
                       f"数据 {d['day'].iloc[0]} ~ {d['day'].iloc[-1]}，共 {len(d)} 根。")

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
                tone = COLOR["多"] if "HH" in trend else COLOR["空"] if "LH" in trend else COLOR["中性"]
                st.markdown(f"<span style='color:{tone}'>{trend}</span>", unsafe_allow_html=True)
                for n_ in notes:
                    st.caption(n_)
                bos = break_of_structure(hb)
                if bos:
                    st.warning(f"{bos}（按最后一段 {hb['seq'].iloc[-1]} 收盘价判定）")
                st.markdown("**当前这一段的形态**")
                pt = patterns(hb)
                st.write("\n".join(f"- {p}" for p in pt) if pt else "无明显形态")
                st.markdown("**关键位**　<span style='color:#888;font-size:12px'>价 · 留下它的那根 K 线</span>",
                            unsafe_allow_html=True)
                st.write("阻力：" + ("；".join(f"{v:.2f} · {t}" for v, t in res) if res else "上方无摆点"))
                st.write("支撑：" + ("；".join(f"{v:.2f} · {t}" for v, t in sup) if sup else "下方无摆点"))
                st.markdown("**量能**")
                read_badges(pa_volume_read(hb))
            last = hb.iloc[-1]
            st.caption(f"最后一段 {last['seq']}（{last['段']}）：开 {last['open']:.2f} "
                       f"高 {last['high']:.2f} 低 {last['low']:.2f} 收 {last['close']:.2f}")
            st.info("半日线一天两根（上午 9:30–11:30、下午 13:00–15:00），两根拼回去等于日线。"
                    "**别在这条线上看原始量能**：A 股日内成交量呈 U 形，上午占全天 56%~66%，"
                    "量能柱天生高一根低一根 —— 图上第二格画的是段内量比，已消掉这个节律。"
                    "同理，这里的均线周期要减半理解。")


# ---- 目标组合 -------------------------------------------------------------
else:
    if sig.empty:
        st.info("还没有信号文件，先跑 run_daily.py。")
        st.stop()
    st.subheader(f"目标组合（信号日 {sig_date}）")
    picked = sig[sig["shares"] > 0].copy()
    st.caption(f"入选 {len(picked)} 只，全池 {len(sig)} 只。表里「信号日价」是 {sig_date} 的收盘价，"
               "组合就是按它算出来的；「现价」是实时价，两者之间的漂移正是照单下单会吃到的偏差。")

    q = quotes(tuple(picked["code"]))
    ts_all = sorted({v["ts"] for v in q.values()})
    if ts_all:
        st.caption(f"实时价报价时间：{ts_all[0]}" + (f" ~ {ts_all[-1]}" if ts_all[0] != ts_all[-1] else ""))

    if st.checkbox("拉取技术面（30 次请求，实测约 2 秒，缓存 5 分钟）", value=True):
        rows = []
        bar = st.progress(0.0)
        for i, (_, r) in enumerate(picked.iterrows(), 1):
            d = kline(r["code"], 240, 120)
            info = q.get(r["code"], {})
            item = {"排名": int(r["rank"]), "代码": r["code"], "名称": r["name"],
                    "打分": r["score"], "持仓": "✓" if r["code"] in pos else "",
                    "信号日价": float(r["price"]), "现价": info.get("price"),
                    "较信号日": (info["price"] / r["price"] - 1) if info.get("price") else None,
                    "报价时间": info.get("ts", "—")}
            if not d.empty and len(d) >= 60:
                d = add_indicators(d)
                last = d.iloc[-1]
                item |= {"距MA20": float(last["close"] / last["MA20"] - 1),
                         "RSI14": float(last["RSI14"]), "量比": float(last["量比"]),
                         "指标截至": str(last["day"])}
            rows.append(item)
            bar.progress(i / len(picked))
        bar.empty()
        t = pd.DataFrame(rows)
        fmt = {"打分": "{:.4f}", "信号日价": "{:.2f}", "现价": "{:.2f}", "较信号日": "{:+.1%}",
               "距MA20": "{:+.1%}", "RSI14": "{:.1f}", "量比": "{:.2f}"}
        # 只格式化真实存在的列：某只取不到行情时它那几列会整列缺席，
        # 直接把完整字典交给 style.format 会 KeyError，整页白屏。
        st.dataframe(paint(t.style.format({k: v for k, v in fmt.items() if k in t.columns}, na_rep="—"),
                           ["较信号日", "距MA20"]),
                     width="stretch", hide_index=True, height=640)
        st.caption("「距MA20」「RSI14」「量比」按日线收盘计算，时间见「指标截至」列。")
    else:
        st.dataframe(picked[["rank", "code", "name", "score", "price", "shares", "status"]],
                     width="stretch", hide_index=True, height=640)
        st.caption(f"表中 price 为信号日 {sig_date} 的收盘价。")

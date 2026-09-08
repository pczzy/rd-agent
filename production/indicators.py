"""技术指标与 price action，纯 pandas 实现。

不用 TA-Lib：它要编译 C 库，而这里需要的十来个指标加起来不过几十行。依赖越少，
18:00 的自动流程越不容易在某次环境变动后悄悄断掉。

两套东西分开：
  - 日线跑经典技术分析（均线/MACD/RSI/KDJ/BOLL/ATR）
  - 半日线跑 price action（结构、支撑阻力、形态）

半日线 = 交易软件上那条"4 小时线"：按整点分桶，上午 9:30-11:30 一根、下午 13:00-15:00
一根，每根实际只有 2 小时。它和日线信息量相同（两根拼回去等于日线），多出来的只有
11:30 那个切点。**不要在半日线上算量能均线** —— A 股日内成交量呈 U 形，实测上午占
全天 56%~66%，量能柱天生高一根低一根，任何量能指标都会跟着两根一循环地摆动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- 经典指标


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在日线上补齐经典指标。df 需含 open/high/low/close/volume。"""
    d = df.copy()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]

    for n in (5, 10, 20, 60):
        d[f"MA{n}"] = c.rolling(n).mean()

    d["DIF"] = ema(c, 12) - ema(c, 26)
    d["DEA"] = ema(d["DIF"], 9)
    d["MACD"] = (d["DIF"] - d["DEA"]) * 2

    # RSI 用 Wilder 平滑（alpha = 1/n），不是简单均值 —— 两者在超买超卖阈值附近能差好几点
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI14"] = 100 - 100 / (1 + up / down.replace(0, np.nan))

    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    d["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["D"] = d["K"].ewm(alpha=1 / 3, adjust=False).mean()
    d["J"] = 3 * d["K"] - 2 * d["D"]

    mid, std = c.rolling(20).mean(), c.rolling(20).std()
    d["BOLL_MID"], d["BOLL_UP"], d["BOLL_LOW"] = mid, mid + 2 * std, mid - 2 * std
    d["BOLL_B"] = (c - d["BOLL_LOW"]) / (d["BOLL_UP"] - d["BOLL_LOW"]).replace(0, np.nan)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["ATR_PCT"] = d["ATR14"] / c

    d["VOL_MA5"] = v.rolling(5).mean()
    d["量比"] = v / d["VOL_MA5"]
    return d


def classic_read(d: pd.DataFrame) -> list[tuple[str, str, str]]:
    """把指标翻成人话。返回 (维度, 结论, 级别) —— 级别用于上色：多/空/中性。"""
    if len(d) < 60:
        return [("数据", f"仅 {len(d)} 根日线，不足 60 根，指标不可靠", "中性")]
    r = d.iloc[-1]
    prev = d.iloc[-2]
    out = []

    ma = [r["MA5"], r["MA10"], r["MA20"], r["MA60"]]
    if all(ma[i] > ma[i + 1] for i in range(3)):
        out.append(("均线", "MA5>10>20>60 多头排列", "多"))
    elif all(ma[i] < ma[i + 1] for i in range(3)):
        out.append(("均线", "MA5<10<20<60 空头排列", "空"))
    else:
        out.append(("均线", "均线缠绕，无明确排列", "中性"))

    pos = "上方" if r["close"] > r["MA20"] else "下方"
    out.append(("价格位置", f"收盘在 MA20 {pos}（{r['close'] / r['MA20'] - 1:+.1%}）",
                "多" if r["close"] > r["MA20"] else "空"))

    if prev["MACD"] <= 0 < r["MACD"]:
        out.append(("MACD", "红柱翻正，金叉", "多"))
    elif prev["MACD"] >= 0 > r["MACD"]:
        out.append(("MACD", "绿柱翻负，死叉", "空"))
    else:
        trend = "红柱走阔" if r["MACD"] > prev["MACD"] > 0 else (
            "红柱收敛" if r["MACD"] > 0 else ("绿柱走阔" if r["MACD"] < prev["MACD"] else "绿柱收敛"))
        out.append(("MACD", f"{trend}（DIF {r['DIF']:.2f} / DEA {r['DEA']:.2f}）",
                    "多" if r["MACD"] > 0 else "空"))

    rsi = r["RSI14"]
    out.append(("RSI14", f"{rsi:.1f}" + ("，超买" if rsi > 70 else "，超卖" if rsi < 30 else "，中性区"),
                "空" if rsi > 70 else "多" if rsi < 30 else "中性"))

    if prev["K"] <= prev["D"] and r["K"] > r["D"]:
        out.append(("KDJ", f"K 上穿 D 金叉（K {r['K']:.1f}）", "多"))
    elif prev["K"] >= prev["D"] and r["K"] < r["D"]:
        out.append(("KDJ", f"K 下穿 D 死叉（K {r['K']:.1f}）", "空"))
    else:
        out.append(("KDJ", f"K {r['K']:.1f} / D {r['D']:.1f} / J {r['J']:.1f}", "中性"))

    b = r["BOLL_B"]
    where = "上轨之上" if b > 1 else "上半区" if b > 0.5 else "下半区" if b > 0 else "下轨之下"
    out.append(("布林带", f"%B {b:.2f}，位于{where}", "空" if b > 1 else "多" if b < 0 else "中性"))

    out.append(("波动", f"ATR14 占价格 {r['ATR_PCT']:.2%}（日均振幅）", "中性"))
    vr = r["量比"]
    out.append(("量能", f"量比 {vr:.2f}" + ("，明显放量" if vr > 1.5 else "，缩量" if vr < 0.7 else "，量能平稳"),
                "中性"))
    return out


# --------------------------------------------------------------------------- price action


def to_half_day(h60: pd.DataFrame) -> pd.DataFrame:
    """60 分钟线合成半日线（即交易软件里那条一天两根的"4 小时线"）。"""
    h = h60.copy()
    h["date"] = h["day"].str[:10]
    h["段"] = h["day"].str[11:16].map(lambda x: "上午" if x in ("10:30", "11:30") else "下午")
    h["seq"] = h["date"] + h["段"].map({"上午": " 11:30", "下午": " 15:00"})
    g = (
        h.groupby("seq", sort=True)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"), date=("date", "first"), 段=("段", "first"))
        .reset_index()
    )
    return g.sort_values("seq").reset_index(drop=True)


def swings(d: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """分形摆动点：高点两侧各 k 根都更低即为摆高，反之为摆低。

    k=2 在半日线上约等于日线的 1 根 —— 再大就把两三天的结构抹平了，再小则每根都是摆点。
    """
    hi, lo = d["high"], d["low"]
    is_h = pd.Series(True, index=d.index)
    is_l = pd.Series(True, index=d.index)
    for i in range(1, k + 1):
        is_h &= (hi > hi.shift(i)) & (hi > hi.shift(-i))
        is_l &= (lo < lo.shift(i)) & (lo < lo.shift(-i))
    out = d.copy()
    out["摆高"], out["摆低"] = is_h.fillna(False), is_l.fillna(False)
    return out


def structure(d: pd.DataFrame) -> tuple[str, list[str]]:
    """用最近的摆点判断结构：HH/HL 上升、LH/LL 下降、混合为震荡。"""
    s = swings(d)
    highs = s.loc[s["摆高"], "high"].tolist()[-3:]
    lows = s.loc[s["摆低"], "low"].tolist()[-3:]
    notes = []
    if len(highs) >= 2:
        notes.append(f"最近两个摆高 {highs[-2]:.2f} → {highs[-1]:.2f}"
                     + ("（抬高 HH）" if highs[-1] > highs[-2] else "（走低 LH）"))
    if len(lows) >= 2:
        notes.append(f"最近两个摆低 {lows[-2]:.2f} → {lows[-1]:.2f}"
                     + ("（抬高 HL）" if lows[-1] > lows[-2] else "（走低 LL）"))
    if len(highs) >= 2 and len(lows) >= 2:
        up = highs[-1] > highs[-2] and lows[-1] > lows[-2]
        dn = highs[-1] < highs[-2] and lows[-1] < lows[-2]
        trend = "上升结构（HH + HL）" if up else "下降结构（LH + LL）" if dn else "震荡：高低点未同向"
    else:
        trend = "摆点不足，结构未成形"
    return trend, notes


def levels(d: pd.DataFrame, n: int = 4, price: float | None = None) -> tuple[list[float], list[float]]:
    """关键位：现价之上的摆点是阻力，之下的是支撑，各取最近的 n 个。

    要按"离现价多近"取，不是按"数值多大"取 —— 后者会把半年前那波高点端上来，
    而现价早已跌穿它下方几十个点，报出"支撑 115"而现价 87 这种废话。

    不区分摆点原本是高点还是低点：被跌破的支撑会变成阻力，反之亦然。price action
    关心的是价格在哪里被拒绝过，不是它当时叫什么名字。
    """
    s = swings(d)
    price = float(d["close"].iloc[-1]) if price is None else price
    pts = sorted(s.loc[s["摆高"], "high"].tolist() + s.loc[s["摆低"], "low"].tolist())

    def thin(vals: list[float]) -> list[float]:
        """由近及远，0.5% 以内视为同一档。"""
        keep: list[float] = []
        for v in vals:
            if not keep or abs(v - keep[-1]) / v > 0.005:
                keep.append(v)
        return keep[:n]

    res = thin([v for v in pts if v > price])                # 升序 = 由近及远
    sup = thin([v for v in pts if v < price][::-1])          # 降序 = 由近及远
    return res, sup


def patterns(d: pd.DataFrame) -> list[str]:
    """最后一根半日线上的形态。只报当根，历史形态留给图去看。"""
    if len(d) < 3:
        return []
    c, p = d.iloc[-1], d.iloc[-2]
    rng = c["high"] - c["low"]
    if rng <= 0:
        return []
    body = abs(c["close"] - c["open"])
    upper = c["high"] - max(c["close"], c["open"])
    lower = min(c["close"], c["open"]) - c["low"]
    out = []
    if body / rng < 0.34 and lower > body * 2:
        out.append("长下影（pin bar）：这一段被砸下去又拉回，低位有承接")
    if body / rng < 0.34 and upper > body * 2:
        out.append("长上影（pin bar）：冲高遇卖压回落")
    if body / rng < 0.1:
        out.append("十字星：多空僵持")
    if c["high"] < p["high"] and c["low"] > p["low"]:
        out.append("内包线（inside bar）：波动收敛，等待方向选择")
    pbody_lo, pbody_hi = min(p["open"], p["close"]), max(p["open"], p["close"])
    if c["close"] > c["open"] and c["close"] >= pbody_hi and c["open"] <= pbody_lo:
        out.append("看涨吞没：完整吞掉上一段实体")
    if c["close"] < c["open"] and c["close"] <= pbody_lo and c["open"] >= pbody_hi:
        out.append("看跌吞没：完整吞掉上一段实体")
    # 跳空只在隔夜有意义：上午根相对上一根（前一日下午）
    if c["段"] == "上午":
        if c["low"] > p["high"]:
            out.append(f"向上跳空 {c['low'] / p['high'] - 1:+.2%}，缺口未回补")
        elif c["high"] < p["low"]:
            out.append(f"向下跳空 {c['high'] / p['low'] - 1:+.2%}，缺口未回补")
    return out


def break_of_structure(d: pd.DataFrame) -> str | None:
    """收盘价是否突破了上一个摆点（BOS）。"""
    s = swings(d)
    last = d.iloc[-1]["close"]
    hi = s.loc[s["摆高"], "high"].tolist()
    lo = s.loc[s["摆低"], "low"].tolist()
    if hi and last > hi[-1]:
        return f"向上突破前摆高 {hi[-1]:.2f}（BOS）"
    if lo and last < lo[-1]:
        return f"向下跌破前摆低 {lo[-1]:.2f}（BOS）"
    return None

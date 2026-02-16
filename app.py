from datetime import datetime

import pytz
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LAT, LON = 7.925056, 81.485056
TIMEZONE = "Asia/Colombo"


@st.cache_data(ttl=600)
def fetch_weather() -> tuple[pd.DataFrame, dict]:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation,rain,uv_index",
        "daily": "sunrise,sunset",
        "past_hours": 24,
        "forecast_days": 3,
        "timezone": TIMEZONE,
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    resp.raise_for_status()
    result = resp.json()
    df = pd.DataFrame(result["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index = df.index.tz_localize(TIMEZONE)
    daily = result["daily"]
    return df, daily


def c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def compute_heat_index(temp_c: float, rh: float) -> float:
    T = c_to_f(temp_c)
    R = rh

    simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + R * 0.094)
    if simple < 80:
        return f_to_c(simple)

    HI = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * R
        - 0.22475541 * T * R
        - 0.00683783 * T * T
        - 0.05481717 * R * R
        + 0.00122874 * T * T * R
        + 0.00085282 * T * R * R
        - 0.00000199 * T * T * R * R
    )

    if R < 13 and 80 <= T <= 112:
        HI -= ((13 - R) / 4) * ((17 - abs(T - 95)) / 17) ** 0.5

    if R > 85 and 80 <= T <= 87:
        HI += ((R - 85) / 10) * ((87 - T) / 5)

    return f_to_c(HI)


def heat_index_category(hi_c: float) -> str:
    if hi_c < 27:
        return "Comfortable"
    if hi_c < 32:
        return "Caution"
    if hi_c < 39:
        return "Extreme Caution"
    if hi_c < 51:
        return "Danger"
    return "Extreme Danger"


def current_conditions_summary(temp: float, rh: float, hi: float, precip: float) -> str:
    parts = []
    if hi >= 39:
        parts.append(f"Dangerously hot ({hi:.0f}°C heat index)")
    elif hi >= 32:
        parts.append(f"Very hot ({hi:.0f}°C heat index)")
    elif hi >= 27:
        parts.append(f"Warm ({temp:.0f}°C)")
    else:
        parts.append(f"Pleasant ({temp:.0f}°C)")

    if rh >= 80:
        parts.append("very humid")
    elif rh >= 60:
        parts.append("humid")
    else:
        parts.append("dry")

    if precip > 2:
        parts.append("heavy rain")
    elif precip > 0:
        parts.append("light rain")

    return parts[0] + ", " + " and ".join(parts[1:]) + "."


def find_outdoor_windows(df: pd.DataFrame) -> list[dict]:
    comfortable = (df["heat_index"] < 27) & (df["precipitation"] <= 2)
    groups = (comfortable != comfortable.shift()).cumsum()
    windows = []
    for _, group in df[comfortable].groupby(groups[comfortable]):
        windows.append({
            "start": group.index[0],
            "end": group.index[-1] + pd.Timedelta(hours=1),
            "hours": len(group),
            "avg_hi": round(group["heat_index"].mean(), 1),
        })
    windows.sort(key=lambda w: w["start"])
    return windows


def time_of_day_label(hour: int) -> str:
    if hour < 4:
        return "Night"
    if hour < 7:
        return "Early Morning"
    if hour < 11:
        return "Morning"
    if hour < 14:
        return "Midday"
    if hour < 17:
        return "Afternoon"
    if hour < 20:
        return "Evening"
    return "Night"


def build_forecast_chart(df: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}]],
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=df["temperature_2m"],
        name="Temperature", line=dict(color="#FF6B35", width=2),
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["heat_index"],
        name="Heat Index", line=dict(color="#D62828", width=2, dash="dash"),
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["relative_humidity_2m"],
        name="Humidity", fill="tozeroy",
        line=dict(color="#457B9D", width=1),
        opacity=0.3,
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["cloud_cover"],
        name="Cloud Cover", fill="tozeroy",
        line=dict(color="#90A4AE", width=1),
        opacity=0.3,
    ), row=2, col=1, secondary_y=False)

    fig.add_trace(go.Bar(
        x=df.index, y=df["precipitation"],
        name="Rain", marker_color="#2196F3",
    ), row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["wind_speed_10m"],
        name="Wind", line=dict(color="#26A69A", width=1.5, dash="dot"),
    ), row=2, col=1, secondary_y=True)

    uv = df["uv_index"].fillna(0)
    uv_colors = [
        "#4CAF50" if v < 3 else
        "#FFC107" if v < 6 else
        "#FF9800" if v < 8 else
        "#F44336" if v < 11 else
        "#9C27B0"
        for v in uv
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=uv,
        name="UV Index", marker_color=uv_colors,
    ), row=3, col=1)

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Humidity (%)", range=[0, 100], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Cloud (%)", range=[0, 100], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Rain/Wind", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="UV", row=3, col=1)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=30, b=40),
        hovermode="x unified",
        height=650,
    )

    return fig


def main():
    st.set_page_config(page_title="Paper Town Weather", layout="wide")
    st.title("Paper Town Weather")

    df, daily = fetch_weather()

    now = datetime.now(pytz.timezone(TIMEZONE))
    current_idx = df.index.get_indexer([now], method="nearest")[0]
    current = df.iloc[current_idx]

    temp = current["temperature_2m"]
    rh = current["relative_humidity_2m"]
    precip = current["precipitation"]
    hi = compute_heat_index(temp, rh)
    category = heat_index_category(hi)

    summary = current_conditions_summary(temp, rh, hi, precip)
    st.markdown(f"*{summary}*")

    today_idx = daily["time"].index(now.strftime("%Y-%m-%d")) if now.strftime("%Y-%m-%d") in daily["time"] else 0
    sunrise = daily["sunrise"][today_idx].split("T")[1]
    sunset = daily["sunset"][today_idx].split("T")[1]

    wind = current["wind_speed_10m"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Temperature", f"{temp:.1f} °C")
    col2.metric("Heat Index", f"{hi:.1f} °C — {category}", delta=f"{hi - temp:+.1f} °C from actual")
    col3.metric("Humidity", f"{rh:.0f}%")
    col4.metric("Wind", f"{wind:.0f} km/h")
    col5.metric("Sunrise / Sunset", f"{sunrise} / {sunset}")

    df["heat_index"] = df.apply(
        lambda row: compute_heat_index(row["temperature_2m"], row["relative_humidity_2m"]),
        axis=1,
    )

    st.subheader("3-Day Forecast")
    future = df[(df.index >= now) & (df.index <= now + pd.Timedelta(hours=72))]
    fig = build_forecast_chart(future)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Best Outdoor Times")
    windows = find_outdoor_windows(future)
    today = now.date()
    tomorrow = today + pd.Timedelta(days=1)
    day_after = today + pd.Timedelta(days=2)
    days = [(today, "Today"), (tomorrow, "Tomorrow"), (day_after, day_after.strftime("%A"))]
    day_cols = st.columns(3)
    for col, (day, label) in zip(day_cols, days):
        with col:
            st.markdown(f"**{label}** — {day.strftime('%a %b %d')}")
            day_windows = [w for w in windows if w["start"].date() == day]
            if day_windows:
                for w in day_windows:
                    tod = time_of_day_label(w["start"].hour)
                    start = w["start"].strftime("%H:%M")
                    end = w["end"].strftime("%H:%M")
                    st.success(f"**{tod}:** {start}–{end} ({w['hours']}h, avg {w['avg_hi']}°C)")
            else:
                st.warning("No comfortable windows")

    st.caption(f"Last refreshed: {now.strftime('%Y-%m-%d %H:%M')} IST  ·  Location: {LAT}, {LON}")


if __name__ == "__main__":
    main()

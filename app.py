import math
from datetime import datetime

import pytz
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LAT, LON = 7.925056, 81.485056
TIMEZONE = "Asia/Colombo"


@st.cache_data(ttl=600)
def fetch_weather() -> tuple[pd.DataFrame, dict] | None:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation,rain,apparent_temperature",
        "daily": "sunrise,sunset",
        "forecast_days": 3,
        "timezone": TIMEZONE,
        "models": "ecmwf_ifs025",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    if resp.status_code == 429:
        return None
    resp.raise_for_status()
    result = resp.json()
    df = pd.DataFrame(result["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index = df.index.tz_localize(TIMEZONE)
    daily = result["daily"]
    return df, daily


@st.cache_data(ttl=600)
def fetch_air_quality() -> pd.DataFrame | None:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "us_aqi,pm2_5,pm10",
        "forecast_days": 3,
        "timezone": TIMEZONE,
    }
    resp = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params, timeout=15)
    if resp.status_code == 429:
        return None
    resp.raise_for_status()
    result = resp.json()
    df = pd.DataFrame(result["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index = df.index.tz_localize(TIMEZONE)
    return df


@st.cache_data(ttl=600)
def fetch_uv() -> pd.DataFrame | None:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "uv_index",
        "forecast_days": 3,
        "timezone": TIMEZONE,
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    if resp.status_code == 429:
        return None
    resp.raise_for_status()
    result = resp.json()
    df = pd.DataFrame(result["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index = df.index.tz_localize(TIMEZONE)
    return df


@st.cache_data(ttl=600)
def generate_ai_summary(
    temp: float, rh: float, precip: float, wind: float,
    hi: float, wb: float, aqi: float | None,
    sunrise: str, sunset: str, next_hours_summary: str,
    avg_high: float | None, avg_low: float | None,
) -> str | None:
    import anthropic

    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    client = anthropic.Anthropic(api_key=api_key)
    historical_ctx = ""
    if avg_high is not None and avg_low is not None:
        historical_ctx = f"- Monthly average high: {avg_high:.0f}°C, average low: {avg_low:.0f}°C\n"
    prompt = (
        f"Current conditions:\n"
        f"- Temperature: {temp:.1f}°C (heat index: {hi:.1f}°C, wet bulb: {wb:.1f}°C)\n"
        f"- Humidity: {rh:.0f}%, Wind: {wind:.0f} km/h, Rain: {precip:.1f} mm/h\n"
        f"- AQI: {f'{aqi:.0f}' if aqi is not None else 'N/A'}\n"
        f"- Sunrise: {sunrise}, Sunset: {sunset}\n"
        f"{historical_ctx}"
        f"\nNext 6 hours:\n{next_hours_summary}\n"
        f"\nThe audience lives here and knows what tropical weather feels like. "
        f"Do NOT describe heat, humidity, or stickiness — they already know. "
        f"Only mention rain if non-zero precipitation actually appears in the forecast. "
        f"Focus on: what's *different* or *notable* compared to the monthly average? "
        f"Specific weather events with timing? Unusual conditions?\n"
        f"If nothing notable is happening, just say so briefly.\n"
        f"Write 1-2 short sentences. No filler. Under 30 words."
    )
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except anthropic.APIError:
        return None


HISTORICAL_URL = "https://raw.githubusercontent.com/nuuuwan/weather_lk/data/data_by_place/81.70E-7.72N-Batticaloa.json"


@st.cache_data(ttl=3600)
def fetch_historical() -> pd.DataFrame | None:
    try:
        resp = requests.get(HISTORICAL_URL, timeout=10)
        resp.raise_for_status()
        records = resp.json()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        return df
    except Exception:
        return None


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


def compute_wet_bulb(temp_c: float, rh: float) -> float:
    T = temp_c
    R = rh
    return (
        T * math.atan(0.151977 * math.sqrt(R + 8.313659))
        + math.atan(T + R)
        - math.atan(R - 1.676331)
        + 0.00391838 * R ** 1.5 * math.atan(0.023101 * R)
        - 4.686035
    )


def aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy (Sensitive)"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


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
    comfortable = (df["heat_index"] < 27) & (df["wet_bulb"] < 24) & (df["precipitation"] <= 2)
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


def build_forecast_chart(df: pd.DataFrame, aqi_df: pd.DataFrame = None) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.2, 0.2, 0.2], vertical_spacing=0.04,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}], [{}]],
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
        x=df.index, y=df["apparent_temperature"],
        name="Apparent Temp", line=dict(color="#E85D04", width=2, dash="dashdot"),
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["wet_bulb"],
        name="Wet Bulb", line=dict(color="#023E8A", width=2, dash="dot"),
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

    if aqi_df is not None:
        aqi_slice = aqi_df.reindex(df.index, method="nearest")
        aqi_vals = aqi_slice["us_aqi"].fillna(0)
        aqi_colors = [
            "#4CAF50" if v <= 50 else
            "#FFC107" if v <= 100 else
            "#FF9800" if v <= 150 else
            "#F44336" if v <= 200 else
            "#9C27B0" if v <= 300 else
            "#7E0023"
            for v in aqi_vals
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=aqi_vals,
            name="AQI", marker_color=aqi_colors,
        ), row=4, col=1)

    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Humidity (%)", range=[0, 100], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Cloud (%)", range=[0, 100], row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Rain/Wind", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="UV", row=3, col=1)
    fig.update_yaxes(title_text="AQI", row=4, col=1)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=30, b=40),
        hovermode="x unified",
        height=800,
    )

    return fig


def main():
    st.set_page_config(page_title="Paper Town Weather", layout="wide")
    st.title("Paper Town Weather")

    weather = fetch_weather()
    if weather is None:
        st.warning("Weather API rate limited. Please wait a few minutes and refresh.")
        st.stop()
    df, daily = weather
    aqi_df = fetch_air_quality()
    uv_df = fetch_uv()
    if uv_df is not None:
        df["uv_index"] = uv_df["uv_index"].reindex(df.index, method="nearest")
    else:
        df["uv_index"] = 0
    hist_df = fetch_historical()

    now = datetime.now(pytz.timezone(TIMEZONE))
    current_idx = df.index.get_indexer([now], method="nearest")[0]
    current = df.iloc[current_idx]

    temp = current["temperature_2m"]
    rh = current["relative_humidity_2m"]
    precip = current["precipitation"]
    hi = compute_heat_index(temp, rh)
    wb = compute_wet_bulb(temp, rh)
    apparent = current["apparent_temperature"]

    aqi = None
    if aqi_df is not None:
        aqi_idx = aqi_df.index.get_indexer([now], method="nearest")[0]
        aqi_current = aqi_df.iloc[aqi_idx]
        aqi = aqi_current["us_aqi"]

    today_idx = daily["time"].index(now.strftime("%Y-%m-%d")) if now.strftime("%Y-%m-%d") in daily["time"] else 0
    sunrise = daily["sunrise"][today_idx].split("T")[1]
    sunset = daily["sunset"][today_idx].split("T")[1]
    wind = current["wind_speed_10m"]

    avg_high, avg_low = None, None
    if hist_df is not None:
        month_data = hist_df[hist_df["month"] == now.month]
        if len(month_data) > 0:
            avg_high = round(month_data["max_temp"].mean(), 1)
            avg_low = round(month_data["min_temp"].mean(), 1)

    summary = current_conditions_summary(temp, rh, hi, precip)
    st.markdown(f"*{summary}*")

    next_hours = df[(df.index > now) & (df.index <= now + pd.Timedelta(hours=6))]
    next_hours_lines = []
    for _, row in next_hours.iterrows():
        h = row.name.strftime("%H:%M")
        next_hours_lines.append(f"{h}: {row['temperature_2m']:.0f}°C, {row['relative_humidity_2m']:.0f}% RH, {row['precipitation']:.1f}mm rain, {row['wind_speed_10m']:.0f}km/h wind")
    next_hours_summary = "\n".join(next_hours_lines)

    ai_summary = generate_ai_summary(
        round(temp, 1), round(rh, 0), round(precip, 1), round(wind, 0),
        round(hi, 1), round(wb, 1), round(aqi, 0) if aqi is not None else None,
        sunrise, sunset, next_hours_summary, avg_high, avg_low,
    )
    if ai_summary:
        st.markdown(ai_summary)

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
    col1.metric("Temperature", f"{temp:.1f} °C")
    col2.metric("Heat Index", f"{hi:.1f} °C", delta=f"{hi - temp:+.1f} °C from actual", delta_color="inverse")
    col3.metric("Apparent Temp", f"{apparent:.1f} °C", delta=f"{apparent - temp:+.1f} °C from actual", delta_color="inverse")
    col4.metric("Wet Bulb Temp", f"{wb:.1f} °C")
    col5.metric("Humidity", f"{rh:.0f}%")
    col6.metric("Wind", f"{wind:.0f} km/h")
    col7.metric("Air Quality", f"{aqi:.0f} AQI" if aqi is not None else "N/A")
    col8.metric("Sunrise", sunrise)
    col9.metric("Sunset", sunset)

    df["heat_index"] = df.apply(
        lambda row: compute_heat_index(row["temperature_2m"], row["relative_humidity_2m"]),
        axis=1,
    )
    df["wet_bulb"] = df.apply(
        lambda row: compute_wet_bulb(row["temperature_2m"], row["relative_humidity_2m"]),
        axis=1,
    )

    st.subheader("3-Day Forecast")
    future = df[(df.index >= now) & (df.index <= now + pd.Timedelta(hours=72))]
    fig = build_forecast_chart(future, aqi_df)
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

    if hist_df is not None:
        month = now.month
        month_data = hist_df[hist_df["month"] == month]
        if len(month_data) > 0:
            avg_max = month_data["max_temp"].mean()
            avg_min = month_data["min_temp"].mean()
            avg_rain = month_data["rain"].mean()
            rain_days_pct = (month_data["rain"] > 0).mean() * 100
            record_high = month_data["max_temp"].max()
            record_low = month_data["min_temp"].min()

            st.subheader(f"Historical — {now.strftime('%B')} in Batticaloa")
            hc1, hc2, hc3, hc4, hc5, hc6 = st.columns(6)
            hc1.metric("Avg High", f"{avg_max:.1f} °C", delta=f"{temp - avg_max:+.1f} °C today", delta_color="inverse")
            hc2.metric("Avg Low", f"{avg_min:.1f} °C")
            hc3.metric("Avg Rain", f"{avg_rain:.1f} mm/day")
            hc4.metric("Rain Days", f"{rain_days_pct:.0f}%")
            hc5.metric("Record High", f"{record_high:.1f} °C")
            hc6.metric("Record Low", f"{record_low:.1f} °C")
            st.caption(f"Based on {len(month_data)} days of {now.strftime('%B')} data (2018–2026) from Sri Lanka Met Dept, Batticaloa station")

    st.caption(f"Last refreshed: {now.strftime('%Y-%m-%d %H:%M')} IST  ·  Location: {LAT}, {LON}")


if __name__ == "__main__":
    main()

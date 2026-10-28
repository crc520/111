from fastmcp import FastMCP
import httpx

# 创建MCP实例
mcp = FastMCP("weather-server")

# 注册天气工具，Open-Meteo免费，不需要API Key
@mcp.tool()
async def get_weather(city: str) -> str:
    """查询城市实时天气
    Args:
        city: 城市名，例如: HangZhou
    """
    # 简易地理编码，这里简化演示，你也可以接入高德
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    async with httpx.AsyncClient() as client:
        geo_res = await client.get(geo_url)
        geo_data = geo_res.json()
        if not geo_data.get("results"):
            return f"找不到城市：{city}"
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto"
            }
        )
        weather_data = weather_resp.json()
        curr = weather_data["current"]
        return (
            f"【{city}实时天气】\n"
            f"温度：{curr['temperature_2m']} ℃\n"
            f"湿度：{curr['relative_humidity_2m']} %\n"
            f"风速：{curr['wind_speed_10m']} km/h\n"
            f"天气代码：{curr['weather_code']}"
        )

if __name__ == "__main__":
    # stdio模式，给Agent调用
    mcp.run(transport="stdio")

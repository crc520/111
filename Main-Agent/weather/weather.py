from fastmcp import FastMCP
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
AMAP_KEY = "b7c61fcc76342049663689d2a82bda04"

mcp = FastMCP("amap-weather-server")


async def get_city_adcode(city_name: str) -> str | None:
    """高德地理编码：中文城市名 → adcode"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            url,
            params={
                "key": AMAP_KEY,
                "address": city_name
            }
        )
        data = resp.json()
        if data.get("status") != "1":
            return None
        geocodes = data.get("geocodes", [])
        if not geocodes:
            return None
        adcode = geocodes[0]["adcode"]
        return adcode


@mcp.tool()
async def get_weather(city: str) -> str:
    """查询国内城市实况天气（高德API）
    Args:
        city: 中文城市名，例如：杭州、上海、北京
    """
    if not AMAP_KEY:
        return "❌ 未配置AMAP_KEY，请在.env填写高德Web服务key"

    # 第一步：城市名称转adcode
    adcode = await get_city_adcode(city)
    if not adcode:
        return f"❌ 无法识别城市：{city}"

    # 第二步：调用高德天气
    weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    async with httpx.AsyncClient() as client:
        res = await client.get(
            weather_url,
            params={
                "key": AMAP_KEY,
                "city": adcode,
                "extensions": "base"  # base实况
            }
        )
        weather_data = res.json()
        if weather_data.get("status") != "1":
            return f"❌ 天气接口调用失败，info:{weather_data.get('info')}"

        lives = weather_data.get("lives", [])
        if not lives:
            return f"❌ 没有天气数据：{city}"

        live = lives[0]
        return (
            f"【{live['city']} 实时天气】\n"
            f"天气：{live['weather']}\n"
            f"温度：{live['temperature']} ℃\n"
            f"湿度：{live['humidity']} %\n"
            f"风向风力：{live['winddirection']} {live['windpower']} 级\n"
            f"发布时间：{live['reporttime']}"
        )

@mcp.tool()
async def get_weather_forecast(city: str) -> str:
    """查询国内城市【4天天气预报（含当天）】，高德接口上限就是4天
    Args:
        city: 中文城市名，例如：嘉兴、杭州、上海
    """
    if not AMAP_KEY:
        return "❌ 未配置AMAP_KEY，请在.env填写高德Web服务key"

    adcode = await get_city_adcode(city)
    if not adcode:
        return f"❌ 无法识别城市：{city}"

    weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
    async with httpx.AsyncClient() as client:
        res = await client.get(
            weather_url,
            params={
                "key": AMAP_KEY,
                "city": adcode,
                "extensions": "all"  # all=4天预报
            }
        )
        weather_data = res.json()
        if weather_data.get("status") != "1":
            return f"❌ 接口调用失败，info:{weather_data.get('info')}"

        forecasts = weather_data.get("forecasts", [])
        if not forecasts:
            return f"❌ 没有预报数据：{city}"

        forecast = forecasts[0]
        cast_list = forecast.get("casts", [])
        lines = [f"【{forecast['city']} 天气预报 | 发布时间：{forecast['reporttime']}】"]
        for item in cast_list:
            lines.append(
                f"\n📅 {item['date']} 星期{item['week']}\n"
                f"白天：{item['dayweather']} {item['daytemp']} ℃ | {item['daywind']}风{item['daypower']}级\n"
                f"夜间：{item['nightweather']} {item['nighttemp']} ℃ | {item['nightwind']}风{item['nightpower']}级"
            )
        return "\n".join(lines)

if __name__ == "__main__":
    # stdio 给你的Agent调用
    mcp.run(transport="stdio")

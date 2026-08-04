from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from data_fetcher import CURRENCY_PAIRS
from signal_engine import analyze_pair

app = FastAPI(title="Forex Signal Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/pairs")
async def get_pairs():
    return JSONResponse(content=CURRENCY_PAIRS)


@app.get("/api/analyze/{pair}")
async def analyze(pair: str, period: str = "6mo", interval: str = "1d"):
    # Convert URL-safe pair name (USD-INR) to display name (USD/INR)
    display_pair = pair.replace("-", "/")
    result = analyze_pair(display_pair, period=period, interval=interval)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"No data available for {display_pair}. Check pair name or try again later."},
        )
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

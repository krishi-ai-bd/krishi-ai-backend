from fastapi import APIRouter, HTTPException
from .daily_suggestion_schema import daily_suggestion_request, daily_suggestion_response
from .daily_suggestion import DailySuggestion

router = APIRouter()
daily_suggestion_agent = DailySuggestion()


@router.get("/daily_suggestion", response_model=daily_suggestion_response)
async def get_daily_suggestion():
    """
    Returns today's daily agricultural suggestion audio URL.
    - If already generated today: returns cached URL instantly
    - If not yet generated: generates on-demand and returns URL
    """
    try:
        return daily_suggestion_agent.get_today_suggestion()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/daily_suggestion", response_model=daily_suggestion_response)
async def post_daily_suggestion(request: daily_suggestion_request):
    """
    Manually trigger/override today's daily suggestion generation.
    Always regenerates even if today's audio already exists.
    """
    try:
        return daily_suggestion_agent.daily_suggestion(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

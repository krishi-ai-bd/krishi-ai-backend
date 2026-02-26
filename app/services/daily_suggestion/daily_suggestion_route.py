from fastapi import APIRouter, HTTPException
from .daily_suggestion_schema import daily_suggestion_request, daily_suggestion_response
from .daily_suggestion import DailySuggestion

router = APIRouter()
daily_suggestion_agent = DailySuggestion()


@router.get("/daily_suggestion", response_model=daily_suggestion_response)
async def get_daily_suggestion():
    """
    Returns today's daily agricultural suggestion audio URL.
    Returns 404 if today's audio hasn't been generated yet (wait for scheduler or call POST).
    """
    result = daily_suggestion_agent.get_today_suggestion()
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Today's suggestion audio not yet generated. It will be ready at the scheduled time."
        )
    return result


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

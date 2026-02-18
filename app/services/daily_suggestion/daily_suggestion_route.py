from fastapi import APIRouter, HTTPException, Header
from .daily_suggestion_schema import daily_suggestion_request, daily_suggestion_response
from .daily_suggestion import DailySuggestion

router = APIRouter()
daily_suggestion_agent = DailySuggestion()


@router.post("/daily_suggestion", response_model=daily_suggestion_response)
async def daily_suggestion(
    request: daily_suggestion_request
):
    return daily_suggestion_agent.daily_suggestion(request)

@router.post("/get_conversation", response_model=daily_suggestion_response)
async def get_conversation(
    request: daily_suggestion_request
):
    return daily_suggestion_agent.get_conversation(request)
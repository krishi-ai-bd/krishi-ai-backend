from fastapi import APIRouter, HTTPException, Header
from .chat_schema import chatbot_request, chatbot_response
from .chat import ChatbotAgent

router = APIRouter()
chatbot_agent = ChatbotAgent()


@router.post("/chat", response_model=chatbot_response)
async def chat(
    request: chatbot_request
):
    return chatbot_agent.chat(request)
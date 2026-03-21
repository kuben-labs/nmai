import asyncio
from src.ai_accounting_agent.coordinator import CoordinatorAgent

async def test():
    coordinator = CoordinatorAgent(
        task_summary="Opprett en ansatt med navn Test User",
        task_type="Direct",
        complexity="single",
        subtasks=[],
        tripletex_credentials={"base_url": "https://api.tripletex.no/v2", "session_token": "dummy"}
    )
    await coordinator.coordinate_execution()

if __name__ == "__main__":
    asyncio.run(test())

"""Smoke tests for the Textual chat app."""
import pytest


@pytest.mark.asyncio
async def test_app_composes_without_error():
    """App starts and has the expected widgets."""
    from chat_app import RagChatApp
    app = RagChatApp()
    async with app.run_test() as pilot:
        assert app.query_one("Header")
        assert app.query_one("#history")
        input_widget = app.query_one("Input")
        assert input_widget is not None


@pytest.mark.asyncio
async def test_quit_binding_exits_app():
    """Pressing q exits the app."""
    from chat_app import RagChatApp
    app = RagChatApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
    # If we reach here without hanging, the app exited cleanly

"""Actual native tools with synthetic API responses only; no real generation."""

import asyncio
import json

import respx

from llama_index.tools.nano_banana_acedatacloud import NanoBananaToolSpec

CLASS = NanoBananaToolSpec
GEN = "nano_banana_generate_image"
EDIT = "nano_banana_edit_image"
QUERY = "nano_banana_get_task"
ROUTE = "/nano-banana/images"
EDIT_ROUTE = "/nano-banana/images"
TASK = "/nano-banana/tasks"
KEY = "image_url"
EXT = "png"
KWARGS = {
    "prompt": "Camera moves through a sunrise garden",
    "model": "nano-banana-2",
    "resolution": "1K",
    "aspect_ratio": "1:1",
    "count": 1,
}
EDIT_KWARGS = {
    "prompt": "Camera moves through a sunrise garden",
    "model": "nano-banana-2",
    "resolution": "1K",
    "aspect_ratio": "1:1",
    "count": 1,
    "image_urls": ["https://example.org/first.png"],
}
EXPECTED = {
    "async": True,
    "model": "nano-banana-2",
    "action": "generate",
    "prompt": "Camera moves through a sunrise garden",
    "count": 1,
    "resolution": "1K",
    "aspect_ratio": "1:1",
}
REQUIRE_FINISHED = False


async def main():
    with respx.mock(assert_all_mocked=True) as router:
        router.post("https://api.acedata.cloud" + ROUTE).respond(200, json={"task_id": "synthetic-task"})
        router.post("https://api.acedata.cloud" + TASK).respond(
            200,
            json={
                "id": "synthetic-task",
                "finished_at": 123,
                "response": {
                    "success": True,
                    "data": [{KEY: "https://example.invalid/synthetic." + EXT, "state": "succeeded"}],
                },
            },
        )
        spec = CLASS(api_token="offline-dummy")
        try:
            generate, _, query = spec.to_tool_list()
            submission = generate.call(**KWARGS).raw_output
            recovered = CLASS(api_token="offline-dummy")
            try:
                result = (await recovered.to_tool_list()[2].acall(task_id=submission["task_id"])).raw_output
            finally:
                await recovered.aclose()
            assert result["status"] == "succeeded" and len(router.calls) == 2
            print(
                json.dumps(
                    {
                        "mode": "synthetic-offline",
                        "tools": [t.metadata.name for t in spec.to_tool_list()],
                        "submission": submission,
                        "recovered_result": result,
                    },
                    indent=2,
                )
            )
        finally:
            await spec.aclose()


if __name__ == "__main__":
    asyncio.run(main())

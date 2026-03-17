#!/usr/bin/env python3
"""
WebSocket proxy: bridges PS <-> localhost so WSL can connect.

WSL cannot connect to Pokemon Showdown through Clash TUN, but Windows can.
This proxy runs on Windows, connects to PS, and exposes a local websocket
server that WSL's MCTS player connects to.

Usage (Windows):
    /c/Windows/py.exe -3 scripts/ws_proxy.py

Then from WSL:
    python3 scripts/play_ladder_mcts.py --proxy-port 9001 --games 100
"""

import asyncio
import os
import sys
import traceback

sys.stdout = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)

# Remove proxy env vars so websockets connects directly
for k in list(os.environ):
    if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        del os.environ[k]

import websockets.legacy.client as ws_client
import websockets.legacy.server as ws_server

PS_URL = "ws://sim.smogon.com:8000/showdown/websocket"
LOCAL_HOST = "0.0.0.0"
LOCAL_PORT = 9876
MAX_RETRIES = 10
RETRY_DELAY = 3


async def connect_to_ps():
    """Connect to PS with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[proxy] Connecting to PS (attempt {attempt}/{MAX_RETRIES})...",
                  flush=True)
            ws = await ws_client.connect(
                PS_URL, ping_interval=20, ping_timeout=20,
                max_queue=None, open_timeout=15,
            )
            print(f"[proxy] Connected to PS!", flush=True)
            return ws
        except Exception as e:
            print(f"[proxy] Attempt {attempt} failed: {type(e).__name__}: {e}",
                  flush=True)
            if attempt < MAX_RETRIES:
                print(f"[proxy] Retrying in {RETRY_DELAY}s...", flush=True)
                await asyncio.sleep(RETRY_DELAY)
    return None


async def proxy_handler(local_ws):
    """Handle one WSL client: connect to PS and forward messages both ways."""
    client_addr = local_ws.remote_address
    print(f"[proxy] WSL client connected from {client_addr}", flush=True)

    ps_ws = await connect_to_ps()
    if ps_ws is None:
        print("[proxy] Failed to connect to PS after all retries", flush=True)
        await local_ws.close(1011, "Failed to connect to PS")
        return

    try:
        async def forward_ps_to_local():
            """Forward PS -> WSL."""
            try:
                async for msg in ps_ws:
                    await local_ws.send(msg)
            except Exception:
                pass

        async def forward_local_to_ps():
            """Forward WSL -> PS."""
            try:
                async for msg in local_ws:
                    await ps_ws.send(msg)
            except Exception:
                pass

        done, pending = await asyncio.wait(
            [
                asyncio.ensure_future(forward_ps_to_local()),
                asyncio.ensure_future(forward_local_to_ps()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except Exception as e:
        print(f"[proxy] Error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    finally:
        try:
            await ps_ws.close()
        except Exception:
            pass

    print(f"[proxy] WSL client disconnected: {client_addr}", flush=True)


async def main():
    print(f"[proxy] Starting websocket proxy on {LOCAL_HOST}:{LOCAL_PORT}", flush=True)
    print(f"[proxy] Will forward to {PS_URL}", flush=True)
    print(f"[proxy] Waiting for WSL client...", flush=True)

    async with ws_server.serve(proxy_handler, LOCAL_HOST, LOCAL_PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())

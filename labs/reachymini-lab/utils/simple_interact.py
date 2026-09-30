# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""
Reachy Mini Ollama Chat Controller

This app integrates Reachy Mini with Ollama LLM to create an interactive experience.
While Ollama generates responses, Reachy Mini performs random expressive movements.
"""

import math
import json
import random
import threading
import time
from typing import Dict, List, Optional


def _create_head_pose(*args, **kwargs):
    from reachy_mini.utils import create_head_pose as _chp

    return _chp(*args, **kwargs)


def check_runtime_dependencies(require_reachy: bool = False) -> bool:
    """Check that optional runtime deps are importable."""
    missing = []
    try:
        import requests  # noqa: F401
    except Exception:
        missing.append("requests")
    if require_reachy:
        try:
            import reachy_mini  # noqa: F401
        except Exception:
            missing.append("reachy-mini")
    if missing:
        print(f"❌ Missing runtime dependencies: {', '.join(missing)}")
        print("   Install: pip install -r requirements.txt")
        if "reachy-mini" in missing:
            print("   For robot: pip install 'reachy-mini[mujoco]'")
        return False
    return True


class ReachyOllamaController:
    def __init__(
        self, ollama_url: str = "http://localhost:11434", model: str = "qwen3.5:0.8b"
    ):
        """
        Initialize the Reachy Mini Ollama controller.

        Args:
            ollama_url: URL of the Ollama server
            model: Ollama model to use
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.reachy_context = None
        self.reachy = None
        self.is_moving = False
        self.stop_movement_flag = threading.Event()

        # Predefined action functions
        self.actions = {
            "nod": self.nod_head,
            "shake": self.shake_head,
            "look_left": self.look_left,
            "look_right": self.look_right,
            "look_up": self.look_up,
            "look_down": self.look_down,
            "antennas_wiggle": self.wiggle_antennas,
            "circle_head": self.circle_head,
            "excited": self.excited_movement,
            "thoughtful": self.thoughtful_movement,
        }

    def connect_reachy(self):
        """Connect to Reachy Mini simulation."""
        from reachy_mini import ReachyMini

        print("Connecting to Reachy Mini...")
        try:
            # Create and enter the context manager
            self.reachy_context = ReachyMini(media_backend="no_media")
            self.reachy = self.reachy_context.__enter__()
            print("Connected to Reachy Mini!")
        except Exception as e:
            print(f"⚠️ Failed to connect to Reachy Mini: {e}")
            self.reachy = None
            self.reachy_context = None

    def disconnect_reachy(self):
        """Disconnect from Reachy Mini."""
        if self.reachy_context:
            try:
                # Exit the context manager
                self.reachy_context.__exit__(None, None, None)
                print("Disconnected from Reachy Mini.")
            except Exception as e:
                print(f"⚠️ Error disconnecting from Reachy Mini: {e}")
            finally:
                self.reachy = None
                self.reachy_context = None

    def random_action(self, duration: float = 2.0):
        """Execute a random action from the predefined actions."""
        if not self.reachy or self.is_moving:
            return

        action_name = random.choice(list(self.actions.keys()))
        print(f"Performing random action: {action_name}")
        self.actions[action_name](duration)

    def continuous_random_actions(self):
        """Continuously perform random actions until stopped."""
        self.is_moving = True
        self.stop_movement_flag.clear()

        def movement_loop():
            while not self.stop_movement_flag.is_set():
                # Random duration between 1.5 and 3 seconds
                duration = random.uniform(1.5, 3.0)
                action_name = random.choice(list(self.actions.keys()))
                print(f"Performing action: {action_name} for {duration:.1f}s")
                self.actions[action_name](duration)
                time.sleep(0.1)  # Small pause between actions

        # Start movement thread
        self.movement_thread = threading.Thread(target=movement_loop)
        self.movement_thread.daemon = True
        self.movement_thread.start()

    def stop_continuous_actions(self):
        """Stop continuous random actions."""
        self.stop_movement_flag.set()
        self.is_moving = False
        if hasattr(self, "movement_thread"):
            self.movement_thread.join(timeout=1.0)

    # -------------------------------
    # Predefined Action Functions
    # -------------------------------

    def nod_head(self, duration: float = 1.0):
        """Nod head up and down."""
        if not self.reachy:
            return

        # Look down
        self.reachy.goto_target(
            head=_create_head_pose(pitch=20, degrees=True), duration=duration / 2
        )
        time.sleep(duration / 4)

        # Look up
        self.reachy.goto_target(
            head=_create_head_pose(pitch=-20, degrees=True), duration=duration / 2
        )

        # Return to center
        time.sleep(duration / 4)
        self.reachy.goto_target(
            head=_create_head_pose(pitch=0, degrees=True), duration=duration / 2
        )

    def shake_head(self, duration: float = 1.0):
        """Shake head left and right."""
        if not self.reachy:
            return

        cycles = 3  # Number of shakes
        for _ in range(cycles):
            # Look left
            self.reachy.goto_target(
                head=_create_head_pose(yaw=30, degrees=True),
                duration=duration / (cycles * 4),
            )
            time.sleep(duration / (cycles * 8))

            # Look right
            self.reachy.goto_target(
                head=_create_head_pose(yaw=-30, degrees=True),
                duration=duration / (cycles * 4),
            )
            time.sleep(duration / (cycles * 8))

        # Return to center
        self.reachy.goto_target(
            head=_create_head_pose(yaw=0, degrees=True), duration=duration / 4
        )

    def look_left(self, duration: float = 1.0):
        """Look to the left."""
        if not self.reachy:
            return

        self.reachy.goto_target(
            head=_create_head_pose(yaw=30, degrees=True), duration=duration / 2
        )
        time.sleep(duration)
        self.reachy.goto_target(
            head=_create_head_pose(yaw=0, degrees=True), duration=duration / 2
        )

    def look_right(self, duration: float = 1.0):
        """Look to the right."""
        if not self.reachy:
            return

        self.reachy.goto_target(
            head=_create_head_pose(yaw=-30, degrees=True), duration=duration / 2
        )
        time.sleep(duration)
        self.reachy.goto_target(
            head=_create_head_pose(yaw=0, degrees=True), duration=duration / 2
        )

    def look_up(self, duration: float = 1.0):
        """Look up."""
        if not self.reachy:
            return

        self.reachy.goto_target(
            head=_create_head_pose(pitch=-20, degrees=True), duration=duration / 2
        )
        time.sleep(duration)
        self.reachy.goto_target(
            head=_create_head_pose(pitch=0, degrees=True), duration=duration / 2
        )

    def look_down(self, duration: float = 1.0):
        """Look down."""
        if not self.reachy:
            return

        self.reachy.goto_target(
            head=_create_head_pose(pitch=20, degrees=True), duration=duration / 2
        )
        time.sleep(duration)
        self.reachy.goto_target(
            head=_create_head_pose(pitch=0, degrees=True), duration=duration / 2
        )

    def wiggle_antennas(self, duration: float = 2.0):
        """Wiggle antennas."""
        if not self.reachy:
            return

        cycles = 4
        for i in range(cycles):
            left_antenna = 0.5 * math.sin(2 * math.pi * i / cycles)
            right_antenna = 0.5 * math.cos(2 * math.pi * i / cycles)
            self.reachy.set_target(antennas=[left_antenna, right_antenna])
            time.sleep(duration / cycles)

        # Return to neutral
        self.reachy.set_target(antennas=[0, 0])

    def circle_head(self, duration: float = 3.0):
        """Move head in a circular pattern."""
        if not self.reachy:
            return

        steps = 20
        for i in range(steps):
            angle = 2 * math.pi * i / steps
            x_offset = 0.02 * math.sin(angle)
            y_offset = 0.02 * math.cos(angle)

            pose = _create_head_pose(x=x_offset * 1000, y=y_offset * 1000, mm=True)
            self.reachy.set_target(head=pose)
            time.sleep(duration / steps)

        # Return to center
        self.reachy.goto_target(head=_create_head_pose(), duration=0.5)

    def excited_movement(self, duration: float = 2.0):
        """Excited movement with quick nods and antenna wiggles."""
        if not self.reachy:
            return

        # Quick nod
        self.reachy.goto_target(
            head=_create_head_pose(pitch=15, degrees=True), duration=0.2
        )
        time.sleep(0.1)
        self.reachy.goto_target(
            head=_create_head_pose(pitch=-15, degrees=True), duration=0.2
        )
        time.sleep(0.1)

        # Wiggle antennas quickly
        for _ in range(3):
            self.reachy.set_target(antennas=[0.7, -0.7])
            time.sleep(0.1)
            self.reachy.set_target(antennas=[-0.7, 0.7])
            time.sleep(0.1)

        # Return to neutral
        self.reachy.set_target(antennas=[0, 0])
        self.reachy.goto_target(head=_create_head_pose(), duration=0.3)

    def thoughtful_movement(self, duration: float = 3.0):
        """Thoughtful movement with slow nods and gentle antenna movements."""
        if not self.reachy:
            return

        # Slow, thoughtful nods
        for _ in range(2):
            self.reachy.goto_target(
                head=_create_head_pose(pitch=10, degrees=True), duration=0.5
            )
            time.sleep(0.3)
            self.reachy.goto_target(
                head=_create_head_pose(pitch=-5, degrees=True), duration=0.5
            )
            time.sleep(0.3)

        # Gentle antenna movements
        for i in range(5):
            antenna_val = 0.3 * math.sin(2 * math.pi * i / 5)
            self.reachy.set_target(antennas=[antenna_val, antenna_val])
            time.sleep(0.2)

        # Return to neutral
        self.reachy.set_target(antennas=[0, 0])
        self.reachy.goto_target(head=_create_head_pose(), duration=0.5)

    # -------------------------------
    # Ollama Integration
    # -------------------------------

    def chat_with_ollama(self, prompt: str, stream: bool = True):
        """
        Send a prompt to Ollama and get a response.

        Args:
            prompt: User's message
            stream: Whether to stream the response

        Returns:
            The full response text
        """
        import requests

        url = f"{self.ollama_url}/api/generate"

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": 0.7, "num_predict": 200},
        }

        print(f"\n🤖 Sending to Ollama ({self.model}): {prompt[:50]}...")

        try:
            response = requests.post(url, json=data, stream=stream)

            if not response.ok:
                print(f"\n⚠️ Ollama HTTP {response.status_code}: {response.text[:200]}")
                return None

            if stream:
                full_response = ""
                print("🤖 Ollama Response (streaming): ", end="", flush=True)

                # Start random movements during response
                self.continuous_random_actions()

                for line in response.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        try:
                            chunk = json.loads(line)
                            if chunk.get("error"):
                                print(f"\n⚠️ Ollama error: {chunk['error']}")
                                self.stop_continuous_actions()
                                return None
                            response_text = (
                                chunk.get("response") or chunk.get("thinking") or ""
                            )
                            if response_text:
                                print(response_text, end="", flush=True)
                                full_response += response_text
                        except json.JSONDecodeError:
                            continue

                # Stop movements after response is complete
                self.stop_continuous_actions()
                print()  # New line after streaming

                return full_response
            else:
                # Non-streaming response
                result = response.json()
                full_response = result.get("response", "")
                print(f"\n🤖 Ollama Response: {full_response}")
                return full_response

        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to Ollama at {self.ollama_url}")
            print("Please make sure Ollama is running: `ollama serve`")
            return None
        except Exception as e:
            print(f"❌ Error communicating with Ollama: {e}")
            return None

    def interactive_chat(self):
        """Run an interactive chat session with Ollama and Reachy Mini."""
        print("=" * 60)
        print("🤖 Reachy Mini Ollama Chat")
        print("=" * 60)
        print(f"Model: {self.model}")
        print(f"Ollama URL: {self.ollama_url}")
        print("\nCommands:")
        print("  /exit or /quit - Exit the chat")
        print("  /help - Show this help")
        print("  /actions - List available actions")
        print("  /do <action> - Perform a specific action")
        print("\nStart chatting! (Reachy Mini will move during responses)")
        print("=" * 60)

        # Connect to Reachy Mini
        try:
            self.connect_reachy()
        except Exception as e:
            print(f"⚠️ Could not connect to Reachy Mini: {e}")
            print("⚠️ Continuing without robot control...")
            self.reachy = None

        eof_count = 0
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue
                eof_count = 0

                if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                    print("\n👋 Goodbye!")
                    break

                elif user_input.lower() == "/help":
                    print("\nAvailable commands:")
                    print("  /exit or /quit - Exit the chat")
                    print("  /help - Show this help")
                    print("  /actions - List available actions")
                    print("  /do <action> - Perform a specific action")
                    print("\nAny other input will be sent to Ollama for a response.")

                elif user_input.lower() == "/actions":
                    print("\n🤖 Available Reachy Mini actions:")
                    for action_name in self.actions.keys():
                        print(f"  - {action_name}")
                    print("\nUse: /do <action_name> to perform an action")

                elif user_input.lower().startswith("/do "):
                    action_name = user_input[4:].strip()
                    if action_name in self.actions:
                        print(f"🤖 Performing action: {action_name}")
                        self.actions[action_name](2.0)
                    else:
                        print(f"❌ Unknown action: {action_name}")
                        print("Use /actions to see available actions")

                else:
                    # Send to Ollama
                    response = self.chat_with_ollama(user_input)
                    if response:
                        # You could add additional processing here
                        pass

            except EOFError:
                eof_count += 1
                if eof_count >= 3:
                    print("\n👋 EOF received, exiting chat.")
                    break
                print("\n⚠️ Empty input (EOF). Type '/exit' to exit.")
                continue
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

        # Cleanup
        self.disconnect_reachy()


def main():
    """Main entry point for the application."""
    import argparse

    parser = argparse.ArgumentParser(description="Reachy Mini Ollama Chat Controller")
    parser.add_argument(
        "--url", default="http://localhost:11434", help="Ollama server URL"
    )
    parser.add_argument("--model", default="qwen3.5:0.8b", help="Ollama model to use")
    parser.add_argument(
        "--test", action="store_true", help="Test Reachy Mini actions without chat"
    )

    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        return

    if not check_runtime_dependencies(require_reachy=True):
        return

    controller = ReachyOllamaController(ollama_url=args.url, model=args.model)

    if args.test:
        # Test mode: connect and run some actions
        print("🧪 Test mode: Testing Reachy Mini actions...")
        try:
            controller.connect_reachy()
            if controller.reachy:
                print("Testing random actions for 10 seconds...")
                controller.continuous_random_actions()
                time.sleep(10)
                controller.stop_continuous_actions()
                controller.disconnect_reachy()
                print("✅ Test completed successfully!")
            else:
                print("❌ Test failed: Could not connect to Reachy Mini")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            controller.disconnect_reachy()


if __name__ == "__main__":
    main()

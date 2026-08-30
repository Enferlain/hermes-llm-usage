import importlib.util
import sys
import types
from pathlib import Path
from unittest import TestCase


API_PATH = Path(__file__).parents[1] / "plugins" / "llm-usage" / "dashboard" / "plugin_api.py"


class _Router:
    def get(self, _path):
        return lambda fn: fn


# The standalone plugin tests only need route decoration; avoid importing the
# dashboard's optional FastAPI/Pydantic stack just to exercise provider mapping.
if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")
    setattr(fastapi_stub, "APIRouter", _Router)
    sys.modules["fastapi"] = fastapi_stub
spec = importlib.util.spec_from_file_location("llm_usage_plugin_api", API_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load plugin API module")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class ZaiQuotaTests(TestCase):
    def test_parse_live_response_shape(self):
        payload = {
            "code": 200,
            "msg": "Operation successful",
            "data": {
                "limits": [
                    {
                        "type": "TIME_LIMIT",
                        "unit": 5,
                        "number": 1,
                        "usage": 100,
                        "currentValue": 0,
                        "remaining": 100,
                        "percentage": 0,
                        "nextResetTime": 1789323426998,
                        "usageDetails": [
                            {"modelCode": "search-prime", "usage": 0},
                            {"modelCode": "web-reader", "usage": 0},
                            {"modelCode": "zread", "usage": 0},
                        ],
                    },
                    {
                        "type": "TOKENS_LIMIT",
                        "unit": 3,
                        "number": 5,
                        "percentage": 29,
                        "nextResetTime": 1788060606395,
                    },
                ],
                "level": "lite",
            },
            "success": True,
        }
        windows, capacity = api.parse_zai_quota(payload)

        self.assertEqual([w["id"] for w in windows], ["mcp_monthly", "five_hour"])
        self.assertEqual(windows[1]["used_pct"], 29.0)
        self.assertEqual(windows[1]["resets_at"], 1788060606395 / 1000)
        self.assertIn("5-hour", windows[1]["label"])
        self.assertEqual(windows[1]["detail"], "Plan: lite")
        self.assertEqual(capacity["plan_level"], "lite")
        self.assertEqual(len(capacity["mcp_tool_usage"]), 3)

    def test_parse_flat_payload_without_data_envelope(self):
        payload = {
            "limits": [{"type": "TOKENS_LIMIT", "number": 5, "percentage": 42}],
            "level": "pro",
        }
        windows, capacity = api.parse_zai_quota(payload)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["used_pct"], 42.0)
        self.assertEqual(capacity["plan_level"], "pro")

    def test_error_envelope_maps_to_auth_error(self):
        # HTTP 200 with {"code":1000,"msg":"Authentication Failed"} — the fetcher
        # surfaces this as a rejected key, and availability classification must
        # read it as an auth error, not a transient unknown.
        error = "Z.ai rejected this key (Authentication Failed)"
        self.assertTrue(api._looks_like_auth_error(error))
        self.assertEqual(api._availability("error", error), "auth_error")

    def test_no_limits_yields_no_windows(self):
        windows, capacity = api.parse_zai_quota({"data": {"limits": []}, "level": "lite"})
        self.assertEqual(windows, [])
        self.assertIsNone(capacity)

    def test_cache_completeness_requires_zai_and_antigravity(self):
        payload = {
            "providers": [
                {"id": p}
                for p in ("anthropic", "grok", "codex", "nous", "venice")
            ]
        }
        self.assertFalse(api._providers_look_complete(payload))
        payload["providers"].append({"id": "zai"})
        payload["providers"].append({"id": "antigravity"})
        self.assertTrue(api._providers_look_complete(payload))


class AntigravityUsageTests(TestCase):
    def test_parse_live_tsv_output(self):
        text = (
            "Gemini Models\tWeekly Limit Remaining\t97%\t2026-09-04T21:30:19Z\n"
            "Gemini Models\tFive Hour Limit Remaining\t88%\t2026-08-30T03:23:33Z\n"
            "Claude and GPT models\tWeekly Limit Remaining\t100%\t2026-09-06T00:44:13Z\n"
            "Claude and GPT models\tFive Hour Limit Remaining\t100%\t2026-08-30T05:44:13Z\n"
        )
        windows = api.parse_antigravity_usage(text)

        self.assertEqual(len(windows), 4)
        by_id = {w["id"]: w for w in windows}
        self.assertEqual(set(by_id), {"gemini_weekly", "gemini_five_hour", "claude_gpt_weekly", "claude_gpt_five_hour"})
        # CLI reports remaining → panel shows used (100 - remaining).
        self.assertEqual(by_id["gemini_weekly"]["used_pct"], 3.0)
        self.assertEqual(by_id["gemini_five_hour"]["used_pct"], 12.0)
        self.assertEqual(by_id["claude_gpt_weekly"]["used_pct"], 0.0)
        self.assertIsNotNone(by_id["gemini_weekly"]["resets_at"])
        self.assertEqual(by_id["gemini_five_hour"]["detail"], "Gemini Models")

    def test_garbage_yields_no_windows(self):
        self.assertEqual(api.parse_antigravity_usage(""), [])
        self.assertEqual(api.parse_antigravity_usage("some agent chatter\nno tabs here"), [])

    def test_missing_timestamp_still_parses(self):
        text = "Gemini Models\tWeekly Limit Remaining\t50%\n"
        windows = api.parse_antigravity_usage(text)
        self.assertEqual(len(windows), 1)
        self.assertIsNone(windows[0]["resets_at"])

    def test_window_ids_map_to_frontend_labels(self):
        # The desktop plugin's shortLabel() keys off these exact ids; keep the
        # contract explicit so a rename here can't silently degrade the panel.
        text = "Gemini Models\tFive Hour Limit Remaining\t10%\t2026-08-30T03:23:33Z\n"
        (window,) = api.parse_antigravity_usage(text)
        self.assertEqual(window["id"], "gemini_five_hour")
        self.assertIn("5-hour", window["label"])

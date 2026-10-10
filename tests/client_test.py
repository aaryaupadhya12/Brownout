import unittest
from types import SimpleNamespace as NS
from Clients.openai_compact import parse_response


def fake_resp(content=None, calls=None, usage=(10, 5, 15)):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))],
              usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2]))

def fake_call(id, name, args, extra=None):
    tc = NS(id=id, function=NS(name=name, arguments=args))
    if extra:
        tc.extra_content = extra
    return tc


class TestParse(unittest.TestCase):
    def test_tool_call(self):
        r = parse_response(fake_resp(calls=[fake_call("c1", "get_metrics", '{"zone": "B"}')]), 12.0)
        print(r)
        self.assertEqual(r.tool_calls[0].args, {"zone": "B"})
        self.assertEqual(r.usage.input_tokens, 10)

    def test_missing_id(self):
        r = parse_response(fake_resp(calls=[fake_call("", "submit", "{}")]), 1.0)
        self.assertEqual(r.tool_calls[0].id, "call_0")

    def test_malformed_args(self):
        r = parse_response(fake_resp(calls=[fake_call("c1", "shift_traffic", "{zone: B")]), 1.0)
        self.assertEqual(r.tool_calls[0].args, {})
        self.assertEqual(r.tool_calls[0].raw_args, "{zone: B")

    def test_plain_text(self):
        r = parse_response(fake_resp(content="all fine"), 1.0)
        self.assertEqual(r.tool_calls, [])
        self.assertEqual(r.text, "all fine")

    def test_real_gemini_shape(self):
        sig = {"google": {"thought_signature": "abc"}}
        calls = [fake_call("call_220042", "get_metrics", '{"zone":"A"}', sig),
                 fake_call("call_220043", "get_metrics", '{"zone":"B"}'),
                 fake_call("call_220044", "get_metrics", '{"zone":"C"}')]
        r = parse_response(fake_resp(calls=calls, usage=(338, 48, 534)), 5.0)
        print(r.usage)
        self.assertEqual(len(r.tool_calls), 3)
        self.assertEqual(r.tool_calls[0].extra, sig)
        self.assertEqual(r.tool_calls[1].extra, {})
        self.assertEqual(r.usage.reasoning_tokens, 148)


if __name__ == "__main__":
    unittest.main()
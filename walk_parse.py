import json
from openai.types.chat import ChatCompletion
from Clients.openai_compact import parse_response

# load the saved response as a real SDK object, with no API call
text = open("raw.json", encoding="utf-8").read()
resp = ChatCompletion.model_validate_json(text)

# the same steps parse_response does, one at a time
msg = resp.choices[0].message
print("STEP 1  message content:", msg.content)
print("STEP 2  number of tool calls:", len(msg.tool_calls))
print()

for i, tc in enumerate(msg.tool_calls):
    print("--- tool call", i, "---")
    print("id:", tc.id)
    print("name:", tc.function.name)

    raw = tc.function.arguments
    print("raw arguments (text):", repr(raw), type(raw).__name__)

    args = json.loads(raw)
    print("after json.loads (dict):", args, type(args).__name__)

    extra = getattr(tc, "extra_content", None)
    if extra is None:
        print("extra_content: NOT FOUND")
    else:
        signature = extra["google"]["thought_signature"]
        print("extra_content: found, signature starts with", signature[:20], "... length", len(signature))
    print()

u = resp.usage
print("usage:", u.prompt_tokens, u.completion_tokens, u.total_tokens)
print("reasoning estimate:", u.total_tokens - u.prompt_tokens - u.completion_tokens)
print()

# now the real function
reply = parse_response(resp, 0.0)
print("parse_response gave:")
print("  text:", repr(reply.text))
for c in reply.tool_calls:
    if c.extra:
        has_extra = "yes"
    else:
        has_extra = "empty"
    print("  ", c.id, c.name, c.args, "extra:", has_extra)
print("  usage:", reply.usage)
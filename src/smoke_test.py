"""Phase 1: verify the local model can (a) chat and (b) emit a valid tool call."""
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MODEL = "gemma4-judge:latest"


@tool
def lookup_order(order_id: str) -> str:
    """Look up the status of a customer order by its ID."""
    return f"Order {order_id}: shipped, delivered 2026-09-18, item=ceramic vase"


def main() -> None:
    llm = ChatOllama(model=MODEL, temperature=0)

    print("[1/3] plain chat ...")
    r = llm.invoke("Reply with exactly one word: hello")
    print("   ->", repr(r.content[:120]))

    print("[2/3] binding tool ...")
    bound = llm.bind_tools([lookup_order])
    print("   -> bound ok")

    print("[3/3] tool call ...")
    r = bound.invoke("My order #4471 arrived smashed. Look it up for me.")
    print("   content   ->", repr(r.content[:120]))
    print("   tool_calls->", r.tool_calls)

    if r.tool_calls:
        tc = r.tool_calls[0]
        ok = tc["name"] == "lookup_order" and "4471" in str(tc["args"])
        print("\nRESULT:", "PASS - model drives tools correctly" if ok
              else f"PARTIAL - called {tc['name']} with {tc['args']}")
    else:
        print("\nRESULT: FAIL - no tool call emitted")


if __name__ == "__main__":
    main()

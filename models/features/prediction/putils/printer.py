def print_loop_message(loop_count: int, process: str, *args):
    message = " ".join(map(str, args))
    print(f"[{process} Loop: {loop_count}] {message}")

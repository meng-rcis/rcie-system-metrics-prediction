def print_loop_message(loop_count, *args):
    message = " ".join(map(str, args))
    print(f"[Current Loop - {loop_count}] {message}")

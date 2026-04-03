def sort_by_time(items):
    return sorted(items, key=lambda x: x.start)


def validate_monotonic(items):
    for i in range(len(items)):
        if items[i].start > items[i].end:
            raise ValueError("start > end detectado")

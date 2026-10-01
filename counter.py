class LineCounter:
    """Counts objects crossing a horizontal line (defined by y-coordinate)."""

    def __init__(self, line_y, direction="down"):
        self.line_y = line_y
        self.direction = direction      # "down" or "up"
        self.counted_ids = set()
        self.count = 0
        self._prev_positions = {}       # id -> previous cy

    def update(self, tracked_objects):
        for obj_id, (cx, cy) in tracked_objects.items():
            prev_cy = self._prev_positions.get(obj_id)
            if prev_cy is not None and obj_id not in self.counted_ids:
                crossed = (
                    (self.direction == "down" and prev_cy < self.line_y <= cy) or
                    (self.direction == "up"   and prev_cy > self.line_y >= cy)
                )
                if crossed:
                    self.counted_ids.add(obj_id)
                    self.count += 1
            self._prev_positions[obj_id] = cy

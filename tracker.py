from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> centroid
        self.disappeared = OrderedDict()   # id -> frames missing
        self.max_disappeared = max_disappeared

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, boxes):
        if not boxes:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        new_centroids = [self._centroid(b) for b in boxes]

        if not self.objects:
            for c in new_centroids:
                self.register(c)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        # distance matrix
        D = np.linalg.norm(
            np.array(obj_centroids)[:, None] - np.array(new_centroids)[None, :],
            axis=2
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            obj_id = obj_ids[r]
            self.objects[obj_id] = new_centroids[c]
            self.disappeared[obj_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(obj_ids))) - used_rows:
            obj_id = obj_ids[r]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for c in set(range(len(new_centroids))) - used_cols:
            self.register(new_centroids[c])

        return self.objects

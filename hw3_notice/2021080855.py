import sys
import math

input_file = sys.argv[1]
n = int(sys.argv[2])
eps = float(sys.argv[3])
min_pts = int(sys.argv[4])

points = {}
label = {}
cluster_id = 0

with open(input_file, 'r') as f:
    for line in f:
        obj_id, x, y = line.strip().split()
        points[int(obj_id)] = (float(x), float(y))


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def range_query(points, tid, eps):
    neighbors = []
    for obj_id, xy in points.items():
        if dist(points[tid], xy) <= eps:
            neighbors.append(obj_id)    
    return neighbors

# dbscan
for p in points:
    if p in label: # processed before
        continue

    neighbors = range_query(points, p, eps)
    if len(neighbors) < min_pts: # p is not core point
        label[p] = -1 # noise
        continue

    # start cluster
    label[p] = cluster_id # next cluseter id
    seed_set = set(neighbors)

    while seed_set:
        q = seed_set.pop()

        if label.get(q) == -1: # noise인데 border point
            label[q] = cluster_id
        
        if q in label:
            continue

        q_neighbors = range_query(points, q, eps)
        label[q] = cluster_id

        if len(q_neighbors) < min_pts: # core point check
            continue
        
        seed_set |= set(q_neighbors)

    cluster_id += 1


# cluster 묶기
clusters = {}
for p, c in label.items():
    if c == -1: # noise
        continue
    clusters.setdefault(c, []).append(p)

cluster_list = sorted(clusters.values(), key=len, reverse=True)[:n]

for i, cluster in enumerate(cluster_list):
    with open(f"{input_file.replace('.txt', '')}_cluster_{i}.txt", 'w') as f:
        for pid in cluster:
            f.write(f"{pid}\n")

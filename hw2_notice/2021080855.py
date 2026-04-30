import sys
from math import log2

train_set = sys.argv[1]
test_set = sys.argv[2]
classification_result = sys.argv[3]

with open(train_set, 'r') as f:
    lines = [i.strip().split('\t') for i in f.readlines()]
    headers = lines[0]
    train_data = lines[1:]

attr_idx = {name: i for i, name in enumerate(headers)}
# class label idx
target_attr_idx = len(headers)-1
attributes = headers[:-1]

with open(test_set, 'r') as f:
    test_lines = [i.strip().split('\t') for i in f.readlines()]
    test_headers = test_lines[0]
    test_data = test_lines[1:]

# info = sum of -plog2(p)
def info(rows):
    total = len(rows)
    if total == 0:
        return 0.0
    
    # class label 별 개수
    counts = {}
    for row in rows:
        label = row[target_attr_idx]
        counts[label] = counts.get(label, 0) + 1

    # sum of -plog2(p)
    result = 0.0
    for c in counts.values():
        p = c / total
        result -= p * log2(p)
    return result

def split_by_attr(rows, attr_name):
    groups = {}
    for row in rows:
        value = row[attr_idx[attr_name]]
        groups.setdefault(value, []).append(row)
    return groups

# gain = info - infoA
# gain ratio = gain / splitinfo
def gain_ratio(rows, attr_name, base_info):
    total = len(rows)
    groups = split_by_attr(rows, attr_name)
    info_a = 0.0
    split_info = 0.0
    for subset in groups.values():
        ratio = len(subset) / total
        info_a += ratio * info(subset)
        split_info -= ratio * log2(ratio)
    gain = base_info - info_a

    if split_info == 0:
        return 0.0
    return gain / split_info

# decision tree
def build_tree(rows, attributes, parent_majority):
    if not rows:
        return {'leaf': True, 'class': parent_majority}
    
    counts = {}
    for row in rows:
        label = row[target_attr_idx]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    # 가장 많은 클래스 찾기
    majority = None
    max_count = -1
    for label, c in counts.items():
        if c > max_count:
            max_count = c
            majority = label
    
    labels = [row[target_attr_idx] for row in rows]
    
    # 모든 row가 같은 class label -> leaf
    if len(set(labels)) == 1:
        return {'leaf': True, 'class': labels[0]}
    
    # no more attribute available -> majority를 leaf
    if not attributes:
        return {'leaf': True, 'class': majority}
    
    # gain ratio max 선택
    base_info = info(rows)
    best_gr, best_attr = max((gain_ratio(rows, a, base_info), a) for a in attributes)

    if best_gr == 0:
        return {'leaf': True, 'class': majority}

    # 내부 노드 생성
    node = {'leaf': False, 'attr': best_attr, 'majority': majority, 'children': {}}
    remaining = [a for a in attributes if a != best_attr]
    
    # attribute 별 children
    for value, subset in split_by_attr(rows, best_attr).items():
        node['children'][value] = build_tree(subset, remaining, majority)
    
    return node

# decision tree 따라 내려가서 예측
def predict(node, sample):
    if node['leaf']:
        return node['class']
    value = sample[attr_idx[node['attr']]]

    # 못 본 속성이면 majority class로 처리
    if value not in node['children']:
        return node['majority']
    
    return predict(node['children'][value], sample)

# decision tree 학습
tree = build_tree(train_data, attributes, None)

predictions = [predict(tree, row) for row in test_data]

with open(classification_result, 'w') as f:
    f.write('\t'.join(headers) + '\n')
    for row, pred in zip(test_data, predictions):
        f.write('\t'.join(row + [pred]) + '\n')

from itertools import combinations
import sys

min_support = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

transactions = []

with open(input_file, 'r') as f:
    for line in f:
        items = frozenset(map(int, line.strip().split()))
        transactions.append(items)

total = len(transactions)

freq = {}
for t in transactions:
    for i in t:
        freq[i] = freq.get(i, 0) + 1

# min support를 만족하는 item만 L1에 포함
L1 = {k: v for k, v in freq.items() if v / total >= min_support / 100}

# 크기 k인 frequent set
Lk = {frozenset([k]): v for k, v in L1.items()}
freq_k = dict(Lk)

k = 2
while Lk:
    # Lk-1 frequent set에서 모든 item 추출
    items = set()
    for fs in Lk.keys():
        items.update(fs)
    
    # 크기가 k인 조합 중에서 k-1 부분집합이 Lk-1에 존재하는것만 후보로 선정
    Ck = []
    for combo in combinations(items, k):
        if all(frozenset(sub) in Lk for sub in combinations(combo, k-1)):
            Ck.append(frozenset(combo))
    
    # min support 체크
    Lk = {}
    for c in Ck:
        count = sum(1 for t in transactions if c <= t)
        if count / total >= min_support / 100:
            Lk[c] = count
    
    freq_k.update(Lk)
    k += 1

with open(output_file, 'w') as f:
    for xy, xy_count in freq_k.items():
        for size in range(1, len(xy)):
            for x in combinations(xy, size):
                x = frozenset(x)
                y = xy - x

                # 출력: {X} \t {Y} \t support \t confidence \n
                f.write("\t".join([
                    "{" + ",".join(map(str, sorted(x))) + "}",
                    "{" + ",".join(map(str, sorted(y))) + "}",
                    f"{xy_count / total * 100:.2f}", # X U Y / total
                    f"{xy_count / freq_k[x] * 100:.2f}" # X U Y / X
                ]) + "\n")

cases = int(input())

def n_gen(inputs):
    inputs = inputs.split(' ')
    for i in inputs:
        yield i
        
for i in range(1, cases + 1):
    print('Case ' + str(i) + ':')
    g = n_gen(input())

    n = int(next(g))
    q = int(next(g))

    database = []
    for i in range(n):
        database.append(set(input().split(' ')))

    for i in range(q):
        count = 0
        full = input()
        query = set(full.split(' '))
        for d in database:
            if len(query.intersection(d)) != 0 and full in ' '.join(d):
                count += 1
        print(count)
        
        

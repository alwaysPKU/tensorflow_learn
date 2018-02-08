l1 = ['23',23,"mac"]
print(l1)


dicr = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
for (x, y) in dicr.items():
    print(x,y)
    # print('key:',x)
    # print('value',y)

l = sorted(dicr.items(), key=lambda dicr:dicr[1])
l2 = sorted(dicr.items(), key=lambda  x:x[0])

print(l)
print(l2)

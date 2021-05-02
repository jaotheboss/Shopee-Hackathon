def n_gen(inputs):
    inputs = inputs.split(' ')
    for i in inputs:
        yield int(i)
g = n_gen(input())
n_items = next(g)
stock = next(g)

class item():
       def __init__(self, index, fd, stock = None, qty = None):
              self.index = index
              self.fd = fd
              self.stock = stock   # all but Parent will be None
              self.qty = qty       # all Parent will be None
              self.parent = []
              self.children = []
              
       def stock_count(self):
              if self.fd == 'dynamic':
                     limiting_factor = min(i.stock_count() for i in self.parent)
                     pre_stock = limiting_factor // self.qty
                     fixed_child = [i.stock_count() for i in self.children if i.fd == 'fixed']
                     fixed_child = 0 + sum(fixed_child)
                     return pre_stock + fixed_child
              else:
                     return self.stock
              
       def add_children(self, child):
              self.children.append(child)
              child.parent.append(self)
              if child.fd == 'fixed' and self.fd == 'fixed':
                     self.stock -= (child.stock * child.qty)

inventory = {
       1: item(1, 'fixed', stock)
}
for i in range(2, n_items + 1):
       g = n_gen(input())
       index = i
       fd = next(g)
       if fd == 1:
              fd = 'dynamic'
       else:
              fd = 'fixed'
       parent_index = next(g)
       qty = next(g)
       if fd == 'fixed':
              si = next(g)
              inventory[index] = item(index, fd, si, qty)
              inventory[parent_index].add_children(inventory[index])
       else:
              inventory[index] = item(index, fd, None, qty)
              inventory[parent_index].add_children(inventory[index])

for i in inventory:
       print(inventory[i].stock_count())
       
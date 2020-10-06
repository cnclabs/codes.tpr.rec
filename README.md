# codes.tpr.rec

## usage
ref. https://github.com/cnclabs/smore
```
git clone https://github.com/cnclabs/codes.tpr.rec
cd codes.tpr.rec
make
./tpr
```

## example data format
for `-train_ui` (user-item)
```
userA itemA 1.0
userA itemB 2.0
userA itemC 1.0
userB itemB 1.0
userB itemD 2.0
...
```
for `-train_iw` (item-word)
```
itemA text 1.0
itemA preference 1.0
itemA embedding 1.0
itemB matrix 1.0
itemB factorization 2.0
...
```

def StockPicker(arr):
  cek = 0
  for i in range(len(arr)):      
    try:
      if arr[i+1] > arr[i]:
        cek = 1
    except:
      continue

  if cek == 0:
    return -1

  orig_arr = arr
  if sorted(arr)[0] == arr[-1]:
    cant_pick = sorted(arr)[0]
    arr.remove(cant_pick)
    cur_pick = sorted(arr)[0]
  else:
    cur_pick = sorted(arr)[0]

  idx = arr.index(cur_pick)
  sell = sorted(arr[idx::],reverse=True)[0]

  return sell - cur_pick

# keep this function call here 
print(StockPicker(input()))
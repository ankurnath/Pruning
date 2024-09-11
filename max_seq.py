

# arr = [-2,1,-3,4,-1,2,1,-5,4] 
arr =[-2,11,-4,2,-3,-10]



n = len (arr)
dp = [0] * len(arr)




dp[0] = arr[0]
subseq = [arr[0]]

for i in range(1,n):
    dp[i] = max(arr[i],dp[i-1]+arr[i])

    

    # subseq.append(arr[i])


print(subseq)

print(max(dp))
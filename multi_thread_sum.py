import concurrent.futures
def range_sum(start, end):
    return sum(range(start, end+1))

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
futures = {executor.submit(range_sum, i*100+1, (i+1)*100): i for i in range(10)}
res = 0
for future in concurrent.futures.as_completed(futures):
    data = future.result()
    res += data
print(res)
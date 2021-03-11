import requests
import concurrent.futures
def send_message(start, end):
    for i in range(start, end):
        r = requests.post('http://localhost:5000/',data={'key':i})
        print(r.text)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        future = {executor.submit(send_message, i*1000, (i+1)*1000) : i for i in range(10)}
        # for f in concurrent.futures.as_completed(future):
        #     res = f.result()
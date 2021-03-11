import redis
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import numpy as np
import json
import os
import pickle
import requests
from bs4 import BeautifulSoup
from http import cookiejar
import re
import time
import concurrent.futures

def cookie_login(cookie_path):
    if os.path.exists(cookie_path):
        with open(cookie_path,'r', encoding='utf-8') as f:
            cookies = json.load(f)
        browser.get('https://www.zhihu.com/')
        for c in cookies:
            if 'sameSite' in c:
                c.pop('sameSite')
            browser.add_cookie(c)
        browser.refresh()
        if is_logged_in():
            return True
        else:
            return False
    else:
        return False

def is_logged_in():
    try:
        browser.find_element_by_css_selector('img.AppHeader-profileAvatar')
    except Exception as e:
        return False
    else:
        return True

def get_following_list_from_single_page(url, visited, idx):
    print(url)
    browser = browsers[idx-1]
    browser.get(url + '/following')
    time.sleep(0.5)
    number_of_following = int(browser.find_elements_by_css_selector('strong.NumberBoard-itemValue')[0].get_attribute('title'))
    following_list = set()
    page = 1
    count = 0
    if url in visited:
        return following_list
    else:
        visited.add(url)
        while count < number_of_following:
            page_url = url + '/following' + '?page=' + str(page)
            browser.get(page_url)
            time.sleep(0.5)
            people_cards = browser.find_elements_by_css_selector('a.UserLink-link')
            for card in people_cards:
                link = card.get_attribute('href')
                # if the link is already visited, then skip (set O(1), list O(n) ), so use set to store
                if link in visited:
                    continue
                following_list.add(link)
            count += len(following_list)
            page += 1
        print('following list',following_list)
        return following_list

# get all the followings using BFS
# def get_following_list(user, degrees, number):
#     depth = 1
#     visited = set()
#     following_list = get_following_list_from_single_page(user, visited, 1)
#     following_list = list(following_list)
#     count = len(following_list)
#     container = []
#     container.extend(following_list)
#     # starting BFS
#     while len(following_list) != 0 and depth < degrees and count < number:
#         # max_workers = number of threads
#         with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#             future_to_following = {executor.submit(chunk_list, following_list, idx, visited): idx for idx in range(1,9)}
#             for url in following_list:
#                 print('url',url)
#             # future_to_following = {executor.submit(get_following_list_from_single_page, url, visited): url for url in following_list}
#             following_list = []
#             for future in concurrent.futures.as_completed(future_to_following):
#                 browser_id = future_to_following[future]
#                 print('browser_id', browser_id)
#             #     print('future_to_following', future_to_following[future])
#                 print('sub_following_list', future.result())
#                 sub_following_list = future.result()
#                 following_list.extend(sub_following_list)
#             count += len(following_list)
#             print('count', count)
#             container.extend(following_list)
#         # since we already visited the whole layer, then connection depth increase
#         depth += 1
#     return container

# def chunk_list(following_list, idx , visited):
#     chunks = len(following_list) // 7
#     if idx <= 7:
#         following_list_chunk = following_list[(idx-1)*chunks:idx*chunks]
#     else:
#         following_list_chunk = following_list[chunks*7:]
#     sub_following = []
#     for link in following_list_chunk:
#         sub_following.extend(get_following_list_from_single_page(link, visited, idx))
#     return sub_following

def collect_all_urls(start_user, degree, initial_list):
    def bfs_get_following_list_recursive(init_user, degree, user_urls, visited, count, idx):
        if init_user in visited:
            return user_urls
        if degree < 1:
            visited.add(start_user)
            return user_urls
        elif degree == 1:
            following_list = get_following_list_from_single_page(init_user, visited,idx)
            visited.add(init_user)
            for u in following_list:
                if count[0] < 300:
                    user_urls.add(u)
                    if u not in visited:
                        count[0] += 1
                else:
                    break
            return user_urls
        else:
            following_list = get_following_list_from_single_page(init_user, visited, idx)
            print('following_list recursion:', following_list)
            visited.add(init_user)
            for u in following_list:
                if count[0] < 10:
                    user_urls.add(u)
                    print('u', u)
                    if u not in visited:
                        bfs_get_following_list_recursive(u, degree-1, user_urls, visited, count, idx)
                        count[0] += 1
                    print('count', count[0])
        return user_urls

    visited = set()
    user_urls = set()
    # count?
    count = [0]
    res = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(bfs_get_following_list_recursive, initial_list[i], degree,user_urls, visited, count, i+1): i for i in range(8)}
        for f in concurrent.futures.as_completed(futures):
            res.extend(list(f.result()))
    return res

# def collect_all_urls_multithread(start_user, degree):
#     def bfs_get_following_list_recursive(init_user, degree, user_urls, visited, count):
#         if init_user in visited:
#             return
#         if degree < 1:
#             visited.add(start_user)
#             return
#         elif degree == 1:
#             following_list = get_following_list_from_single_page_single_browser(init_user, visited)
#             visited.add(init_user)
#             for u in following_list:
#                 if count[0] < 300:
#                     user_urls.add(u)
#                     if u not in visited:
#                         count[0] += 1
#                 else:
#                     break
#             return
#         else:
#             following_list = get_following_list_from_single_page_single_browser(init_user, visited)
#             # how to count the number when using multithreads?
#             visited.add(init_user)
#             for u in following_list:
#                 if count[0] < 300:
#                     user_urls.add(u)
#                     print('u', u)
#                     if u not in visited:
#                         bfs_get_following_list_recursive(u, degree-1, user_urls, visited, count)
#                         count[0] += 1
#                     print('count', count[0])
#
#     visited = set()
#     user_urls = set()
#     count = [0]
#     bfs_get_following_list_recursive(start_user, degree,user_urls, visited, count)
#     return user_urls
#
# def get_following_list_from_single_page_single_browser(url, visited):
#     print(url)
#     browser = browsers[0]
#     browser.get(url + '/following')
#     time.sleep(0.5)
#     number_of_following = int(browser.find_elements_by_css_selector('strong.NumberBoard-itemValue')[0].get_attribute('title'))
#     following_list = set()
#     page = 1
#     count = 0
#     if url in visited:
#         return following_list
#     else:
#         visited.add(url)
#         while count < number_of_following:
#             page_url = url + '/following' + '?page=' + str(page)
#             browser.get(page_url)
#             time.sleep(0.5)
#             people_cards = browser.find_elements_by_css_selector('a.UserLink-link')
#             for card in people_cards:
#                 link = card.get_attribute('href')
#                 # if the link is already visited, then skip (set O(1), list O(n) ), so use set to store
#                 if link in visited:
#                     continue
#                 following_list.add(link)
#             count += len(following_list)
#             page += 1
#         print('following list',following_list)
#         return following_list
#
# def chunk_list_recursive(following_list, idx , user_urls, visited, count):
#     chunks = len(following_list) // 7
#     if idx <= 7:
#         following_list_chunk = following_list[(idx-1)*chunks:idx*chunks]
#     else:
#         following_list_chunk = following_list[chunks*7:]
#     for u in following_list_chunk:
#         bfs_get_following_list_recursive(u, degree - 1, user_urls, visited, count)
#     return

if __name__=='__main__':
    browsers = []
    for i in range(8):
        PATH  = 'C:\webdrivers\chromedriver.exe'
        options = Options()
        options.add_argument(r"user-data-dir=C:\Users\10782\AppData\Local\Google\Chrome\User Data\Default")
        options.add_argument(r'user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36"')
        url = 'https://www.zhihu.com/'
        browser = webdriver.Chrome(PATH)
        browser.get(url)
        cookie_login('./Cookies.json')
        browsers.append(browser)
    user = 'https://www.zhihu.com/people/e-chung'
    initial_list = ['https://www.zhihu.com/people/huo-hua-de-41', 'https://www.zhihu.com/org/cheng-xu-yuan-ke-zhan', 'https://www.zhihu.com/people/julyedu.com',
                    'https://www.zhihu.com/org/liang-zi-wei-48', 'https://www.zhihu.com/people/su-sheng-han-24', 'https://www.zhihu.com/people/McDoge',
                    'https://www.zhihu.com/people/li-xiao-zhi-37', 'https://www.zhihu.com/people/cuan-ding-24', 'https://www.zhihu.com/people/highlandpark',
                    'https://www.zhihu.com/people/sinya']
    # following_list, visited = get_following_list_from_single_page('https://www.zhihu.com/people/e-chung', visited)
    # container = get_following_list(user, 3, 500)
    user_urls = collect_all_urls(user, 3, initial_list)
    print(len(user_urls))
    print(user_urls)

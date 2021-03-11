import gensim
from gensim import models
import requests
import os
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
import json
import numpy as np
import torch
from transformers import BertTokenizer,BertModel
import  pandas as pd
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine
import math
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
# import web_driver
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


def get_text(url):
    browser.get(url)
    # 点击“阅读全文”按钮操作
    buttons_read_all = browser.find_elements_by_css_selector('button.Button.ContentItem-more.Button--plain')
    for button in buttons_read_all:
        try:
            button.click()
        except:
            print('unable to clik')
    # 提取全部文本
    document = browser.find_elements_by_css_selector('span.RichText.ztext.CopyrightRichText-richText')
    text_whole = []
    for doc in document:
        text = doc.get_attribute('innerText')
        text_split = text.split('\n\n')
        for t in text_split:
            chunk_text = list(get_split(t, 100))
            text_whole.extend(chunk_text)
    df = pd.DataFrame(text_whole, columns=['sentences'])
    sentences = df.sentences.values
    # max_len = 0
    # for sent in sentences:
    #     print('sent', sent)
    #     input_ids = tokenizer.encode(sent, add_special_tokens=True)
    #     max_len = max(max_len, len(input_ids))
    return sentences

def get_split(text, length):
    return (text[0+i:length+i] for i in range(0, len(text), length))

def user_vec_func(url, topics):
    user_vec = torch.empty(768).to(device)
    for t in range(len(topics)):
        sentences = get_text(url + topics[t])
        if len(sentences) == 0:
            continue
        input_ids = []
        attention_masks = []
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=150, pad_to_max_length=True,
                                                 return_token_type_ids=True, return_attention_mask=True, return_tensors='pt')
            # marked_text = "[CLS] " + sent + " [SEP]"
            # tokenized_text = tokenizer.tokenize(marked_text)
            # print(tokenized_text)
            # print(len(encoded_dict['input_ids'][0]), encoded_dict['input_ids'])
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        with torch.no_grad():
            i = 1
            for id, mask in zip(input_ids, attention_masks):
                print('sentence id:', i)
                id = torch.unsqueeze(id, 0)
                mask = torch.unsqueeze(mask, 0)
                outputs = model(id, mask)
                if i == 1:
                    # ？从第几层提取特征更好？ 为什么选择倒数第二层？ -1 - -4 平均
                    hidden_states = outputs[-2]
                else:
                    new_hidden_states = outputs[-2]
                    hidden_states = torch.cat((hidden_states, new_hidden_states), 0)
                i += 1
            # assign weight to different features based on a exp function
            user_vec_temp = torch.mean(hidden_states, dim=0)*math.exp(-0.8*(t))
        user_vec += user_vec_temp
    user_vec = user_vec.cpu()
    user_vec = user_vec.reshape(1, -1)
    user_vec = user_vec.tolist()
    return user_vec

def collect_data(url, topics):
    data_all = []
    for u in url:
        u_vec = user_vec_func(u, topics)
        data_all.extend(u_vec)
    return data_all

def cosine_similarity(vec1, vec2):
    similarity = 1 - cosine(vec1, vec2)
    return similarity

# def collect_all_urls(start_user, degree):
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
#                 if count[0] < 50:
#                     user_urls.add(u)
#                     if u not in visited:
#                         count[0] += 1
#                 else:
#                     break
#             return
#         else:
#             following_list = get_following_list_from_single_page_single_browser(init_user, visited)
#             visited.add(init_user)
#             for u in following_list:
#                 if count[0] < 50:
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

# def get_following_list_from_single_page_single_browser(url, visited):
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
#         return following_list

def data_dimensionReduction(data_all):
    # standardize all the data
    data_all = StandardScaler().fit_transform(data_all)
    # pca dimension reduction
    ppc = pca.fit_transform(data_all)
    return ppc

def cluster_func(data_all):
    ppc = datadata_dimensionReduction(data_all)
    # uncomment this section to decide the number of clusters brfore K-means
    # find the elbow point of the graph, wcss: within cluster sum of squares VS number of clusters
    # wcss = []
    # for i in range(1,21):
    #     kmeans_pca = KMeans(n_clusters = i, init='k-means++', random_state=42)
    #     kmeans_pca.fit(ppc)
    #     wcss.append(kmeans_pca.inertia_)
    # plt.figure(figsize = (10,8))
    # plt.plot(range(1,21), wcss, marker = 'o', linestyle = '--')
    # plt.xlabel('Number of Clustering')
    # plt.ylabel('wcss')
    # plt.title('k-means with PCA clustering')
    # plt.show()
    kmeans_pca = KMeans(n_clusters = 12, init='k-means++', random_state=42)
    kmeans_pca.fit(ppc)
    df_pca_kmeans = pd.DataFrame(ppc)
    df_pca_kmeans['cluster labels'] = kmeans_pca.labels_
    # choose the first two components to plot
    x_axis = df_pca_kmeans[0]
    y_axis = df_pca_kmeans[1]
    plt.figure(figsize=(10,8))
    sns.scatterplot(x_axis, y_axis, hue = df_pca_kmeans['cluster labels'], palette = ['g', 'r', 'c'])
    plt.title('Clusters by PCA Components')
    plt.show()

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

if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    # os.popen('D:\Jobs\信息聚合SDE Project course\web_driver\web_driver.py')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
    pca = PCA(n_components = 10)
    # model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True, from_tf=True)
    model.to(device)
    model.eval()
    browsers = []
    for i in range(8):
        PATH = 'C:\webdrivers\chromedriver.exe'
        url = 'https://www.zhihu.com/'
        browser = webdriver.Chrome(PATH)
        cookie_login('./Cookies.json')
        browsers.append(browser)
    # ? can not launch the page answers, posts
    topics = ['', '/answers', '/posts', '/pins']
    # topics = ['', '/pins']
    # user1 = 'https://www.zhihu.com/people/huo-hua-de-41'
    # user2 = 'https://www.zhihu.com/people/crackinterview'
    # user3 = 'https://www.zhihu.com/people/yu-shan-xiao-chu-niang'
    # user4 = 'https://www.zhihu.com/people/li-ze-hua-24'
    user5 = 'https://www.zhihu.com/people/e-chung'
    # user_vec1 = user_vec_func(user4, topics)
    # user_all = web_driver.collect_all_urls(user5, 3)
    initial_list = ['https://www.zhihu.com/people/huo-hua-de-41', 'https://www.zhihu.com/org/cheng-xu-yuan-ke-zhan',
                    'https://www.zhihu.com/people/julyedu.com',
                    'https://www.zhihu.com/org/liang-zi-wei-48', 'https://www.zhihu.com/people/su-sheng-han-24',
                    'https://www.zhihu.com/people/McDoge',
                    'https://www.zhihu.com/people/li-xiao-zhi-37', 'https://www.zhihu.com/people/cuan-ding-24',
                    'https://www.zhihu.com/people/highlandpark',
                    'https://www.zhihu.com/people/sinya']
    user_all = collect_all_urls(user5, 3, initial_list)
    print(user_all)
    # data_all = collect_data(user_all, topics)
    # print(data_all)
    # user_vec2 = user_vec_func(user2)
    # user_vec3 = user_vec_func(user3)
    # similarity1 = cosine_similarity(user_vec1, user_vec2)
    # similarity2 = cosine_similarity(user_vec1, user_vec3)
    # similarity3 = cosine_similarity(user_vec2, user_vec3)
    # print(similarity1, similarity2, similarity3)



